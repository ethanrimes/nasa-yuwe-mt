"""Provision / deprovision the H100 VM via the `az` CLI, with idempotent teardown.

This is the cost-critical module. Guarantees:
  * `down()` is idempotent and also deletes the disk + NIC + public IP so we never
    leak billable resources.
  * `up()` records the provision time so `uptime_hours()` / budget tripwires work.
  * Everything is a thin wrapper over `az` so there is no SDK auth ceremony — it
    reuses your `az login` session.

NOTHING here spends money unless you actually call `up()`. Dry-run with `plan`.

CLI:
    python -m nymt_shared.h100 plan          # show what `up` would do (no spend)
    python -m nymt_shared.h100 up            # provision (spot H100) -- SPENDS MONEY
    python -m nymt_shared.h100 status        # show power state + uptime + est cost
    python -m nymt_shared.h100 ip            # print public IP
    python -m nymt_shared.h100 ssh -- <cmd>  # run a remote command
    python -m nymt_shared.h100 down          # delete VM + disk + NIC + IP (idempotent)
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config

_STATE_FILE = config.REPO_ROOT / ".h100_state.json"


# --------------------------------------------------------------------------- #
# small az helpers
# --------------------------------------------------------------------------- #
# Resolve the `az` executable once. On Windows the CLI is `az.cmd`, which
# subprocess (shell=False) cannot launch via the bare name "az" because
# CreateProcess does not honor PATHEXT; shutil.which returns the full path.
_AZ = shutil.which("az") or "az"


def _az(args: list[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    cmd = [_AZ, *args]
    return subprocess.run(
        cmd, check=check, text=True,
        capture_output=capture,
    )


def _az_json(args: list[str], check: bool = True):
    cp = _az([*args, "-o", "json"], check=check)
    out = (cp.stdout or "").strip()
    return json.loads(out) if out else None


def _load_state() -> dict:
    if _STATE_FILE.exists():
        return json.loads(_STATE_FILE.read_text())
    return {}


def _save_state(d: dict) -> None:
    _STATE_FILE.write_text(json.dumps(d, indent=2))


def _set_sub() -> None:
    _az(["account", "set", "--subscription", config.SUBSCRIPTION_ID])


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #
def cloud_init() -> str:
    """cloud-init that installs uv + clones nothing (we rsync the repo over SSH).

    The DSVM ubuntu-hpc image already ships CUDA drivers, so we only add uv and a
    working dir. Training deps are installed by the run script after rsync.
    """
    return """#cloud-config
package_update: true
runcmd:
  - [ bash, -lc, "curl -LsSf https://astral.sh/uv/install.sh | sh" ]
  - [ bash, -lc, "mkdir -p /home/%s/nymt && chown -R %s:%s /home/%s/nymt" ]
""" % (config.ADMIN_USER, config.ADMIN_USER, config.ADMIN_USER, config.ADMIN_USER)


def plan() -> dict:
    """Return (and print) exactly what `up()` will create. No spend."""
    spec = {
        "subscription": config.SUBSCRIPTION_ID,
        "resource_group": config.RESOURCE_GROUP,
        "location": config.LOCATION,
        "vm_name": config.VM_NAME,
        "vm_size": config.VM_SIZE,
        "image": config.VM_IMAGE,
        "spot": config.USE_SPOT,
        "os_disk_gb": config.OS_DISK_GB,
        "admin_user": config.ADMIN_USER,
        "ssh_pubkey": config.SSH_PUBKEY,
        "est_cost_per_hour": config.VM_COST_PER_HOUR_SPOT if config.USE_SPOT
        else config.VM_COST_PER_HOUR_ONDEMAND,
        "max_budget_hours": config.MAX_BUDGET_HOURS,
    }
    print(json.dumps(spec, indent=2))
    return spec


def _ensure_rg() -> None:
    exists = _az_json(["group", "exists", "--name", config.RESOURCE_GROUP], check=False)
    if exists is not True:
        _az(["group", "create", "--name", config.RESOURCE_GROUP,
             "--location", config.LOCATION])


def _grant_blob_access(principal_id: str) -> None:
    """Grant the VM's managed identity 'Storage Blob Data Contributor' on the storage
    account so on-VM mirroring can auth via managed identity (this account forbids key
    auth). Idempotent: an existing assignment is treated as success."""
    if not principal_id:
        print("[h100] WARNING: no principalId; skipping blob role grant.")
        return
    acct_id = None
    cp = _az(["storage", "account", "show", "--name", config.STORAGE_ACCOUNT,
              "--query", "id", "-o", "tsv"], check=False)
    if cp.returncode == 0:
        acct_id = (cp.stdout or "").strip()
    if not acct_id:
        print(f"[h100] WARNING: could not resolve storage account id for "
              f"{config.STORAGE_ACCOUNT}; grant Blob Data Contributor manually.")
        return
    cp = _az(["role", "assignment", "create",
              "--assignee-object-id", principal_id,
              "--assignee-principal-type", "ServicePrincipal",
              "--role", "Storage Blob Data Contributor",
              "--scope", acct_id], check=False)
    if cp.returncode == 0:
        print("[h100] granted managed identity 'Storage Blob Data Contributor'.")
    elif "RoleAssignmentExists" in ((cp.stderr or "") + (cp.stdout or "")):
        print("[h100] blob role already assigned (ok).")
    else:
        print(f"[h100] WARNING: blob role grant failed: {(cp.stderr or cp.stdout).strip()[:300]}")


def up(wait: bool = True) -> dict:
    """Provision the H100 VM. SPENDS MONEY. Idempotent-ish: if the VM already
    exists it just returns its info."""
    _set_sub()
    _ensure_rg()

    # Already there?
    existing = _az_json(
        ["vm", "show", "-g", config.RESOURCE_GROUP, "-n", config.VM_NAME],
        check=False,
    )
    if existing:
        print(f"[h100] VM {config.VM_NAME} already exists; reusing.")
    else:
        if not Path(config.SSH_PUBKEY).exists():
            raise SystemExit(
                f"SSH public key not found: {config.SSH_PUBKEY}. "
                "Generate one with `ssh-keygen` or set H100_SSH_PUBKEY."
            )
        ci = config.REPO_ROOT / ".cloud-init.yaml"
        ci.write_text(cloud_init())
        args = [
            "vm", "create",
            "-g", config.RESOURCE_GROUP,
            "-n", config.VM_NAME,
            "--image", config.VM_IMAGE,
            "--size", config.VM_SIZE,
            "--location", config.LOCATION,
            "--admin-username", config.ADMIN_USER,
            "--ssh-key-values", config.SSH_PUBKEY,
            "--os-disk-size-gb", str(config.OS_DISK_GB),
            "--custom-data", str(ci),
            "--assign-identity", "[system]",
            "--public-ip-sku", "Standard",
            "--nic-delete-option", "Delete",
            "--os-disk-delete-option", "Delete",
            "--public-ip-address-dns-name", f"{config.VM_NAME}-{int(time.time())}",
        ]
        if config.USE_SPOT:
            args += ["--priority", "Spot", "--eviction-policy", "Delete",
                     "--max-price", "-1"]
        print(f"[h100] creating {config.VM_SIZE} (spot={config.USE_SPOT}) ...")
        _az(args)

    info = _az_json(["vm", "show", "-g", config.RESOURCE_GROUP, "-n", config.VM_NAME,
                     "-d"])
    # Ensure the VM has a system-assigned identity with Blob Data Contributor so on-VM
    # mirroring can authenticate (this storage account forbids key-based auth).
    principal_id = (info or {}).get("identity", {}).get("principalId") if info else None
    if not principal_id:
        ident = _az_json(["vm", "identity", "assign", "-g", config.RESOURCE_GROUP,
                          "-n", config.VM_NAME], check=False)
        principal_id = (ident or {}).get("systemAssignedIdentity") if ident else None
    _grant_blob_access(principal_id)

    state = _load_state()
    state.update({
        "vm_name": config.VM_NAME,
        "provisioned_at": state.get("provisioned_at") or datetime.now(timezone.utc).isoformat(),
        "public_ip": info.get("publicIps") if info else None,
        "size": config.VM_SIZE,
        "spot": config.USE_SPOT,
    })
    _save_state(state)
    print(f"[h100] up. public_ip={state['public_ip']}")
    return state


def _detect_egress_cidrs(samples: int = 8) -> list[str]:
    """Sample this host's public egress IP a few times and return covering /29 CIDRs.

    The orchestrator host may sit behind an Azure SNAT pool that rotates the source
    IP across a small contiguous block (observed: a /29). NRMS policy (below) blocks
    broad "Internet" SSH but tolerates narrow, specific source prefixes — so we punch
    the allow rule for exactly the /29(s) we egress from.
    """
    import ipaddress
    import urllib.request

    urls = ["https://api.ipify.org", "https://ifconfig.me/ip", "https://icanhazip.com"]
    seen: set[str] = set()
    for i in range(samples):
        url = urls[i % len(urls)]
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=8) as r:
                ip = r.read().decode().strip()
            net = ipaddress.ip_network(f"{ip}/29", strict=False)
            seen.add(str(net))
        except Exception:
            continue
    return sorted(seen)


def ensure_ssh_access(cidrs: list[str] | None = None) -> None:
    """Punch a high-priority inbound TCP/22 allow into the VM's NSG.

    Microsoft NRMS policy injects Deny-from-Internet rules (priority ~105-109) into
    BOTH the auto-created NIC `<vm>NSG` AND a separate subnet-level NSG a few minutes
    after provisioning, which override the default allow-ssh (priority 1000) and silently
    black-hole the orchestrator's SSH (manifests as connection timeouts). We add a narrow
    allow at priority 100 (above the NRMS denies) scoped to this host's specific egress
    /29(s) on every NSG in the resource group. Idempotent.
    """
    if cidrs is None:
        cidrs = _detect_egress_cidrs()
    if not cidrs:
        print("[h100] WARNING: could not detect egress IP; SSH may be blocked by NRMS NSG.")
        return
    # The allow must be punched into EVERY NSG in the traffic path. The auto-created
    # `<vm>NSG` sits on the NIC, but NRMS also associates a separate, unpredictably-named
    # NSG (e.g. `NRMS-...<vnet>`) on the SUBNET. A packet must pass BOTH, so an allow on
    # only the NIC NSG still gets black-holed by the subnet NSG's NRMS deny. Patch all NSGs
    # in the (dedicated) resource group to be safe.
    nsgs = [n.get("name") for n in (_az_json(
        ["network", "nsg", "list", "-g", config.RESOURCE_GROUP], check=False) or [])
        if n.get("name")]
    if not nsgs:
        nsgs = [f"{config.VM_NAME}NSG"]
    ok = []
    for nsg in nsgs:
        if _ensure_ssh_rule_on_nsg(nsg, cidrs):
            ok.append(nsg)
    if ok:
        print(f"[h100] ensured SSH allow rule on {ok} for {cidrs}.")
    else:
        print(f"[h100] WARNING: failed to set SSH allow rule on any NSG ({nsgs}).")


def _ensure_ssh_rule_on_nsg(nsg: str, cidrs: list[str]) -> bool:
    """Idempotently create/update the priority-100 inbound TCP/22 allow on one NSG."""
    name = "allow-ssh-orchestrator-host"
    create = [
        "network", "nsg", "rule", "create",
        "-g", config.RESOURCE_GROUP, "--nsg-name", nsg, "-n", name,
        "--priority", "100", "--direction", "Inbound", "--access", "Allow",
        "--protocol", "Tcp", "--source-address-prefixes", *cidrs,
        "--destination-port-ranges", "22",
        "--description", "orchestrator SSH (specific egress /29s) above NRMS deny",
    ]
    cp = _az(create, check=False)
    if cp.returncode == 0:
        return True
    update = [
        "network", "nsg", "rule", "update",
        "-g", config.RESOURCE_GROUP, "--nsg-name", nsg, "-n", name,
        "--priority", "100", "--access", "Allow", "--direction", "Inbound",
        "--protocol", "Tcp", "--source-address-prefixes", *cidrs,
        "--destination-port-ranges", "22",
    ]
    cp2 = _az(update, check=False)
    if cp2.returncode != 0:
        msg = ((cp2.stderr or "") + (cp2.stdout or "")).strip()[:200]
        print(f"[h100] WARNING: could not set SSH allow rule on {nsg}: {msg}")
        return False
    return True


def get_ip() -> str | None:
    info = _az_json(
        ["vm", "show", "-g", config.RESOURCE_GROUP, "-n", config.VM_NAME, "-d"],
        check=False,
    )
    return (info or {}).get("publicIps") or None


def uptime_hours() -> float | None:
    state = _load_state()
    ts = state.get("provisioned_at")
    if not ts:
        return None
    started = datetime.fromisoformat(ts)
    return (datetime.now(timezone.utc) - started).total_seconds() / 3600.0


def status() -> dict:
    _set_sub()
    info = _az_json(
        ["vm", "show", "-g", config.RESOURCE_GROUP, "-n", config.VM_NAME, "-d"],
        check=False,
    )
    up_h = uptime_hours()
    rate = config.VM_COST_PER_HOUR_SPOT if config.USE_SPOT else config.VM_COST_PER_HOUR_ONDEMAND
    out = {
        "exists": bool(info),
        "power_state": (info or {}).get("powerState"),
        "public_ip": (info or {}).get("publicIps"),
        "uptime_hours": round(up_h, 3) if up_h else None,
        "est_cost_usd": round(up_h * rate, 2) if up_h else None,
        "budget_hours": config.MAX_BUDGET_HOURS,
        "over_budget": (up_h is not None and up_h > config.MAX_BUDGET_HOURS),
    }
    print(json.dumps(out, indent=2))
    return out


def ssh(remote_cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    ip = get_ip()
    if not ip:
        raise SystemExit("no public IP; is the VM up?")
    target = f"{config.ADMIN_USER}@{ip}"
    known_hosts = "NUL" if os.name == "nt" else "/dev/null"
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", f"UserKnownHostsFile={known_hosts}",
        "-i", config.SSH_KEY, target, remote_cmd,
    ]
    return subprocess.run(cmd, check=check, text=True)


def down() -> None:
    """Delete VM + all attached resources. IDEMPOTENT — safe to call any time,
    even if nothing exists. This is what guarantees we stop paying."""
    _set_sub()
    rg = config.RESOURCE_GROUP
    name = config.VM_NAME

    # Capture attached resource IDs before deleting the VM (best effort).
    nic_ids: list[str] = []
    disk_name = None
    pip_ids: list[str] = []
    info = _az_json(["vm", "show", "-g", rg, "-n", name], check=False)
    if info:
        nic_ids = [n["id"] for n in info.get("networkProfile", {}).get("networkInterfaces", [])]
        disk_name = (info.get("storageProfile", {}).get("osDisk", {}) or {}).get("name")

    if info:
        print(f"[h100] deleting VM {name} ...")
        _az(["vm", "delete", "-g", rg, "-n", name, "--yes"], check=False)

    # NICs (and their public IPs)
    for nic_id in nic_ids:
        nic = _az_json(["network", "nic", "show", "--ids", nic_id], check=False)
        if nic:
            for ipcfg in nic.get("ipConfigurations", []):
                pip = (ipcfg.get("publicIPAddress") or {}).get("id")
                if pip:
                    pip_ids.append(pip)
        _az(["network", "nic", "delete", "--ids", nic_id], check=False)
    for pip in pip_ids:
        _az(["network", "public-ip", "delete", "--ids", pip], check=False)

    # OS disk
    if disk_name:
        _az(["disk", "delete", "-g", rg, "-n", disk_name, "--yes"], check=False)

    # Clear local state so uptime accounting resets.
    if _STATE_FILE.exists():
        _STATE_FILE.unlink()
    print("[h100] down. (VM + NIC + public IP + disk deleted; idempotent)")


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="nymt_shared.h100")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    sub.add_parser("up")
    sub.add_parser("status")
    sub.add_parser("ip")
    sub.add_parser("down")
    p = sub.add_parser("ssh")
    p.add_argument("remote", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)

    if args.cmd == "plan":
        plan()
    elif args.cmd == "up":
        up()
    elif args.cmd == "status":
        status()
    elif args.cmd == "ip":
        print(get_ip() or "")
    elif args.cmd == "down":
        down()
    elif args.cmd == "ssh":
        remote = args.remote
        if remote and remote[0] == "--":
            remote = remote[1:]
        ssh(" ".join(remote))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
