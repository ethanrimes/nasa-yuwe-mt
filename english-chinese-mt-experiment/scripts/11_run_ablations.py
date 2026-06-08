#!/usr/bin/env python3
"""Local orchestrator for the En-Zh ablation matrix on one rented H100.

Subcommands
-----------
plan   (default, NO SPEND)  Load ablations.yaml, build the packing schedule, print
                            GPU-hours + $ estimate and the exact H100 `up` spec.
run    (SPENDS MONEY)       Provision the H100, sync code + reconstructed subsets,
                            build splits, extend each model size, launch the packed
                            training waves with the snapshot mirror, then ALWAYS
                            tear the VM down (idempotent) — even on error/SIGINT.

The cost-critical guarantee: `run` wraps everything in try/finally so
`nymt_shared.h100.down()` fires no matter what, and a --max-budget-hours tripwire
kills the VM if the matrix overruns. Nothing here spends until you call `run`.

Examples
--------
    python scripts/11_run_ablations.py                 # plan (dry-run, no spend)
    python scripts/11_run_ablations.py plan --spot
    python scripts/11_run_ablations.py run --yes       # provision + train + teardown
    python scripts/11_run_ablations.py run --keep-up   # leave VM up (debug; you pay!)
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
MONOREPO = Path(__file__).resolve().parents[2]  # nasa-yuwe-mt root (holds shared/ + root pyproject)
sys.path.insert(0, str(REPO / "src"))

from nymt_shared import config, h100, schedule  # noqa: E402

ABLATIONS = REPO / "configs" / "ablations.yaml"
SPLITS_MANIFEST = REPO / "data" / "splits" / "splits_manifest.json"
# The optional es<->nasa NLLB fine-tunes that fill spare H100 capacity (separate pkg).
SNMT_DIR = config.ES_NASA_DIR
NASA_JSONL = REPO / "data-en-zh" / "source" / "nasa_yuwe_parallel_dataset.jsonl"

# Directories that must NOT be shipped to the VM (huge / regenerated remotely).
# Paths are relative to the monorepo root (the rsync source).
RSYNC_EXCLUDES = [
    ".git", ".venv", "__pycache__", "models", "data/splits",
    "data-en-zh/raw", "data-en-zh/queue", "data-en-zh/source", "*.parquet",
    # The other experiment + its 93 GB results are irrelevant to the En-Zh GPU run.
    "english-spanish-mt-experiment",
    # Vendored reference checkouts (~3.4 GB) — never needed on the training VM.
    "_external_repos",
]


def load_runs(ablations_path: Path) -> tuple[list[schedule.Run], dict]:
    cfg = yaml.safe_load(ablations_path.read_text(encoding="utf-8"))
    subsets = cfg["subsets"]
    # Prefer the exact post-dev-holdout scale from the splits manifest if present.
    manifest = {}
    if SPLITS_MANIFEST.exists():
        manifest = json.loads(SPLITS_MANIFEST.read_text(encoding="utf-8"))
    scale_for = {
        "sentence_only": manifest.get("scale_sentence_only"),
        "sentence_plus_vocab": manifest.get("scale_sentence_plus_vocab"),
    }
    runs: list[schedule.Run] = []
    for r in cfg["runs"]:
        sub = r["subset"]
        pairs = scale_for.get(sub) or subsets[sub]["pairs"]
        runs.append(
            schedule.Run(
                run_id=r["id"],
                model_key=r["model_key"],
                subset=sub,
                pairs=int(pairs),
                epochs=float(cfg.get("epochs", 3)),
                effective_batch=int(cfg.get("effective_batch", 64)),
            )
        )
    return runs, cfg


def load_nllb_runs(no_nllb: bool) -> list[schedule.Run]:
    """Build schedule.Runs for the es<->nasa NLLB co-runs (for the plan estimate).

    Computes the real train-pair count from the local parallel jsonl via the same
    deterministic split snmt uses on the VM, so the estimate matches the actual run.
    Robust: returns [] if disabled, snmt is unavailable, or the jsonl is absent.
    """
    if no_nllb:
        return []
    try:
        sys.path.insert(0, str(SNMT_DIR / "src"))
        from snmt import data as snmt_data  # noqa: PLC0415
        from snmt import runs as nllb_runs  # noqa: PLC0415

        cfg = nllb_runs.load_config()
        if NASA_JSONL.exists():
            rows = snmt_data.load_jsonl(NASA_JSONL)
            splits = snmt_data.assign_splits(rows)
            train_pairs = len(splits["train"])
        else:
            # Nominal: full corpus minus default dev+test fractions.
            train_pairs = int(24229 * 0.94)
        return nllb_runs.load_nllb_runs(train_pairs, cfg)
    except Exception as e:
        print(f"NOTE: NLLB co-runs not included in estimate ({e}).")
        return []


def cmd_plan(args) -> int:
    runs, cfg = load_runs(ABLATIONS)
    nllb = load_nllb_runs(getattr(args, "no_nllb", False))
    runs = runs + nllb
    est = schedule.estimate(runs, spot=args.spot)
    print("=== H100 provision spec (NOT executed) ===")
    h100.plan()
    print("\n=== Ablation packing schedule ===")
    print(json.dumps(est.to_dict(), indent=2))
    n_enzh = len(runs) - len(nllb)
    print(
        f"\nSUMMARY: {est.to_dict()['n_runs']} runs "
        f"({n_enzh} en-zh + {len(nllb)} nllb es-nasa) in "
        f"{est.to_dict()['n_waves']} wave(s) -> "
        f"~{est.gpu_hours:.1f} GPU-h packed (~${est.cost_usd:.0f} "
        f"{'spot' if args.spot else 'on-demand'}); "
        f"serial would be ~{est.serial_gpu_hours:.1f} GPU-h."
    )
    if not SPLITS_MANIFEST.exists():
        print(
            "\nNOTE: data/splits/splits_manifest.json not found — using nominal "
            "row counts from ablations.yaml. Run 13_prepare_ablation_subsets.py "
            "(after the dataset is fully built) for exact scales."
        )
    return 0


def _ssh(ip: str, cmd: str, check: bool = True):
    import os
    import subprocess

    known_hosts = "NUL" if os.name == "nt" else "/dev/null"
    full = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", f"UserKnownHostsFile={known_hosts}",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=10",
        "-i", config.SSH_KEY, f"{config.ADMIN_USER}@{ip}", cmd,
    ]
    return subprocess.run(full, check=check, text=True)


def _rsync(ip: str) -> None:
    """Ship the monorepo (minus RSYNC_EXCLUDES) to ~/nymt on the VM.

    Uses rsync when available (Linux/macOS, exercised by the tests). On Windows
    rsync does not ship with the OS, so we fall back to an equivalent
    tar -> scp -> remote-untar pipeline using the bundled tar.exe (bsdtar),
    scp, and ssh — all of which ARE present on modern Windows. The same
    RSYNC_EXCLUDES are honored so the two paths transfer an identical tree.
    """
    import os
    import shutil
    import subprocess
    import tempfile

    known_hosts = "NUL" if os.name == "nt" else "/dev/null"
    dest = f"{config.ADMIN_USER}@{ip}:/home/{config.ADMIN_USER}/nymt/"

    if shutil.which("rsync"):
        excludes = []
        for e in RSYNC_EXCLUDES:
            excludes += ["--exclude", e]
        ssh_cmd = (
            f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile={known_hosts} "
            f"-i {config.SSH_KEY}"
        )
        cmd = [
            "rsync", "-az", "--delete", *excludes, "-e", ssh_cmd,
            str(MONOREPO) + "/", dest,
        ]
        subprocess.run(cmd, check=True, text=True)
        return

    # --- Windows / no-rsync fallback: tar (bsdtar) -> scp -> remote untar ---
    remote_home = f"/home/{config.ADMIN_USER}"
    tar_excludes = [f"--exclude={e}" for e in RSYNC_EXCLUDES]
    with tempfile.TemporaryDirectory() as td:
        tarball = os.path.join(td, "nymt.tgz")
        # -C MONOREPO . => archive entries are ./shared/..., ./english-chinese-...,
        # which untar to ~/nymt/shared, matching the rsync `MONOREPO/ -> ~/nymt/` layout.
        subprocess.run(
            ["tar", "-czf", tarball, *tar_excludes, "-C", str(MONOREPO), "."],
            check=True, text=True,
        )
        scp = [
            "scp", "-o", "StrictHostKeyChecking=no",
            "-o", f"UserKnownHostsFile={known_hosts}",
            "-i", config.SSH_KEY, tarball,
            f"{config.ADMIN_USER}@{ip}:{remote_home}/nymt.tgz",
        ]
        subprocess.run(scp, check=True, text=True)
        _ssh(
            ip,
            f"rm -rf {remote_home}/nymt && mkdir -p {remote_home}/nymt && "
            f"tar -xzf {remote_home}/nymt.tgz -C {remote_home}/nymt && "
            f"rm -f {remote_home}/nymt.tgz",
            check=True,
        )


def cmd_run(args) -> int:
    runs, cfg = load_runs(ABLATIONS)
    nllb = load_nllb_runs(getattr(args, "no_nllb", False))
    runs = runs + nllb
    est = schedule.estimate(runs, spot=args.spot)
    print(json.dumps(est.to_dict(), indent=2))
    budget = args.max_budget_hours or config.MAX_BUDGET_HOURS
    print(
        f"\nAbout to PROVISION {config.VM_SIZE} (spot={args.spot}) and run "
        f"{len(runs)} training runs (~{est.gpu_hours:.1f} GPU-h, "
        f"~${est.cost_usd:.0f}). Budget tripwire: {budget} h."
    )
    if not args.yes:
        try:
            if input("type 'GO' to provision and spend: ").strip() != "GO":
                print("aborted.")
                return 130
        except EOFError:
            print("non-interactive; pass --yes to proceed.")
            return 130

    torn = {"done": False}

    def _teardown(*_):
        if torn["done"]:
            return
        torn["done"] = True
        if args.keep_up:
            print("[run] --keep-up set; NOT tearing down (you are paying!).")
            return
        print("[run] tearing down H100 (idempotent) ...")
        try:
            h100.down()
        except Exception as e:  # never leave a VM running
            print(f"[run] WARN during teardown: {e}; retrying once")
            try:
                h100.down()
            except Exception as e2:
                print(f"[run] ERROR: teardown failed twice: {e2}")

    signal.signal(signal.SIGINT, lambda *_: (_teardown(), sys.exit(130)))
    signal.signal(signal.SIGTERM, lambda *_: (_teardown(), sys.exit(143)))

    try:
        state = h100.up()
        ip = state["public_ip"]
        if not ip:
            raise SystemExit("no public IP after provision")
        print(f"[run] VM up at {ip}; ensuring NSG SSH access (NRMS workaround) ...")
        try:
            h100.ensure_ssh_access()
        except Exception as e:
            print(f"[run] WARN: ensure_ssh_access failed ({e}); continuing.")
        print(f"[run] waiting for SSH ...", flush=True)
        ssh_ok = False
        for attempt in range(60):
            try:
                _ssh(ip, "echo ready", check=True)
                ssh_ok = True
                break
            except Exception:
                if attempt and attempt % 6 == 0:
                    print(f"[run] still waiting for SSH (attempt {attempt}/60) ...", flush=True)
                time.sleep(10)
        if not ssh_ok:
            raise SystemExit("SSH never became ready (check NSG / key)")

        print("[run] syncing repo ...", flush=True)
        _rsync(ip)

        remote_root = f"/home/{config.ADMIN_USER}/nymt"
        # Build the on-VM command: install deps (incl. the snmt pkg for the NLLB
        # co-runs), prepare splits, extend models, then launch packed training waves
        # with the mirror. All on the VM.
        budget_arg = f"--max-budget-hours {budget}"
        spot_flag = "--spot" if args.spot else ""
        nllb_flag = "--no-nllb" if getattr(args, "no_nllb", False) else ""
        # The snmt package ships under the monorepo (not excluded by RSYNC_EXCLUDES);
        # install it (with its GPU extra) so run_matrix_on_vm.py can pack NLLB runs.
        # '|| true' keeps the En<->Zh matrix resilient if the snmt install hiccups.
        snmt_install = "uv pip install -e '../spanish-nasa-mt-experiment[gpu]' || true"
        # Robust bootstrap: SSH can become ready BEFORE cloud-init finishes installing
        # uv, so we (a) wait for cloud-init, (b) poll for uv up to ~5 min and self-install
        # if still missing, (c) use `set -e` so a real failure aborts (no silent 2>/dev/null
        # swallow that previously let a missing venv slip through to exit 127).
        remote = (
            "set -e; "
            'export PATH="$HOME/.local/bin:$PATH"; '
            "cloud-init status --wait >/dev/null 2>&1 || true; "
            "for i in $(seq 1 60); do command -v uv >/dev/null 2>&1 && break; sleep 5; done; "
            "command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh); "
            'export PATH="$HOME/.local/bin:$PATH"; '
            f"cd {remote_root}/english-chinese-mt-experiment; "
            "uv venv --python 3.12 .venv; "
            ". .venv/bin/activate; "
            "uv pip install -e '..'; "
            "uv pip install -e .; "
            f"{snmt_install}; "
            f"python -u scripts/run_matrix_on_vm.py {spot_flag} {budget_arg} {nllb_flag}"
        )
        print("[run] launching matrix on VM (this is the billed part) ...", flush=True)
        _ssh(ip, remote, check=True)
        print("[run] matrix finished.", flush=True)
    finally:
        _teardown()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("plan")
    p.add_argument("--spot", action="store_true", default=config.USE_SPOT)
    p.add_argument("--no-nllb", action="store_true",
                   help="exclude the es<->nasa NLLB co-runs from the estimate")
    p.set_defaults(func=cmd_plan)
    r = sub.add_parser("run")
    r.add_argument("--spot", action="store_true", default=config.USE_SPOT)
    r.add_argument("--yes", action="store_true")
    r.add_argument("--keep-up", action="store_true", help="do NOT teardown (debug)")
    r.add_argument("--max-budget-hours", type=float, default=None)
    r.add_argument("--no-nllb", action="store_true",
                   help="do NOT pack the es<->nasa NLLB fine-tunes onto the GPU")
    r.set_defaults(func=cmd_run)
    args = ap.parse_args()
    if not args.cmd:
        args.func = cmd_plan
        args.spot = config.USE_SPOT
        args.no_nllb = False
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
