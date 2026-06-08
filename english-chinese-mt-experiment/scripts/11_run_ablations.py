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
import base64
import json
import os
import shlex
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
        # Per-subset train-pair counts so the sentence_only runs are costed on the
        # smaller split. sentence_plus_vocab = whole corpus; sentence_only is filtered
        # by the committed content-hash level map (same as the on-VM prepare step).
        keys_path = SNMT_DIR / "data" / "sentence_keys.json"
        sentence_keys: set[str] = set()
        if keys_path.exists():
            sentence_keys = set(json.loads(keys_path.read_text(encoding="utf-8"))["keys"])

        def _train_pairs(rows: list[dict]) -> int:
            return len(snmt_data.assign_splits(rows)["train"])

        if NASA_JSONL.exists():
            rows = snmt_data.load_jsonl(NASA_JSONL)
            full_pairs = _train_pairs(rows)
            sent_pairs = (
                _train_pairs(snmt_data.filter_by_keys(rows, sentence_keys))
                if sentence_keys else full_pairs
            )
        else:
            # Nominal: full corpus minus default dev+test fractions.
            full_pairs = int(24229 * 0.94)
            sent_pairs = int(16446 * 0.94)
        train_pairs_by_subset = {
            "sentence_only": sent_pairs,
            "sentence_plus_vocab": full_pairs,
        }
        return nllb_runs.load_nllb_runs(train_pairs_by_subset, cfg)
    except Exception as e:
        print(f"NOTE: NLLB co-runs not included in estimate ({e}).")
        return []


def cmd_plan(args) -> int:
    only_nllb = getattr(args, "only_nllb", False)
    nllb = load_nllb_runs(getattr(args, "no_nllb", False))
    if only_nllb:
        runs = list(nllb)
    else:
        runs, cfg = load_runs(ABLATIONS)
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


def _ssh_capture(ip: str, cmd: str, timeout: int = 45):
    """Run a short SSH command, capturing stdout, NEVER raising.

    Used by the completion-polling loop. A transient SSH failure (NRMS
    remediation resetting the connection, a network blip, the sshd not yet
    reachable) must NOT propagate — the caller simply retries on the next
    poll. Returns (rc, stdout). rc is the ssh process exit code; on timeout
    or spawn failure rc is 255 and stdout is "".
    """
    import os
    import subprocess

    known_hosts = "NUL" if os.name == "nt" else "/dev/null"
    full = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", f"UserKnownHostsFile={known_hosts}",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15",
        "-i", config.SSH_KEY, f"{config.ADMIN_USER}@{ip}", cmd,
    ]
    try:
        p = subprocess.run(
            full, check=False, text=True,
            capture_output=True, timeout=timeout,
        )
        return p.returncode, (p.stdout or "")
    except Exception:
        return 255, ""


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
    only_nllb = getattr(args, "only_nllb", False)
    if only_nllb:
        runs, cfg = [], {}
    else:
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
        # Spot `vm create` allocation is non-deterministic — H100 spot capacity in
        # the region can momentarily be unavailable (an allocation that fails now
        # often succeeds a minute later). Retry a few times instead of aborting the
        # whole run on the first transient failure. h100.down() between attempts
        # clears any partially-created resources so each retry starts clean.
        state = None
        last_err = None
        for attempt in range(1, 7):
            try:
                state = h100.up()
                break
            except Exception as e:
                last_err = e
                print(f"[run] provision attempt {attempt}/6 failed "
                      f"({type(e).__name__}); cleaning + retrying in 60s ...",
                      flush=True)
                try:
                    h100.down()
                except Exception:
                    pass
                time.sleep(60)
        if state is None:
            raise SystemExit(f"provision failed after 6 attempts: {last_err}")
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
        # rsync is a ~1GB scp; NRMS can reset it mid-transfer (-> teardown if it
        # raised). Retry a few times, re-asserting NSG access between attempts.
        synced = False
        for attempt in range(1, 5):
            try:
                _rsync(ip)
                synced = True
                break
            except Exception as e:
                print(f"[run] repo sync attempt {attempt}/4 failed ({e}); "
                      f"re-asserting NSG + retrying ...", flush=True)
                try:
                    h100.ensure_ssh_access()
                except Exception:
                    pass
                time.sleep(15)
        if not synced:
            raise SystemExit("repo sync to VM failed after 4 attempts (NSG/NRMS?)")

        remote_root = f"/home/{config.ADMIN_USER}/nymt"
        # Build the on-VM script: install deps (incl. the snmt pkg for the NLLB
        # co-runs), prepare splits, extend models, then launch packed training waves
        # with the mirror. All on the VM.
        budget_arg = f"--max-budget-hours {budget}"
        spot_flag = "--spot" if args.spot else ""
        nllb_flag = "--no-nllb" if getattr(args, "no_nllb", False) else ""
        onlynllb_flag = "--only-nllb" if getattr(args, "only_nllb", False) else ""
        only_val = getattr(args, "only", None)
        only_flag = f"--only {shlex.quote(only_val)}" if only_val else ""
        # The snmt package ships under the monorepo (not excluded by RSYNC_EXCLUDES);
        # install it (with its GPU extra) so run_matrix_on_vm.py can pack NLLB runs.
        # '|| true' keeps the En<->Zh matrix resilient if the snmt install hiccups.
        snmt_install = "uv pip install -e '../spanish-nasa-mt-experiment[gpu]' || true"
        # Propagate a Weights & Biases API key (if the operator exported one locally)
        # into the detached VM run so en<->zh training logs to wandb ONLINE. The key is
        # only ever written into this runtime-generated, base64-shipped run_all.sh on a
        # transient VM that we delete at teardown -- it is NEVER committed to source.
        # Absent a key, run_matrix_on_vm.launch_run cleanly disables wandb instead.
        _wandb_key = os.environ.get("WANDB_API_KEY", "").strip()
        wandb_export = f"export WANDB_API_KEY={shlex.quote(_wandb_key)}\n" if _wandb_key else ""
        # Propagate an optional run-subset filter so the on-VM matrix only runs the
        # selected NLLB runs (e.g. re-doing just the small models). Comma-separated
        # substrings of run ids; honored by snmt.runs.run_specs(). Absent => all runs.
        _snmt_only = os.environ.get("SNMT_ONLY_RUNS", "").strip()
        snmt_only_export = (
            f"export SNMT_ONLY_RUNS={shlex.quote(_snmt_only)}\n" if _snmt_only else ""
        )
        # CRITICAL: training must NOT be tied to this persistent SSH session. NRMS
        # periodically re-remediates the NSGs and can RESET an established SSH
        # connection ("client_loop: send disconnect: Connection reset"). When that
        # killed the single foreground `_ssh(...)` that used to run the whole matrix,
        # subprocess raised -> finally -> teardown -> the VM (and hours of in-progress
        # training) were destroyed mid-wave. So we now:
        #   1. write a bootstrap script to the VM,
        #   2. launch it DETACHED with setsid (survives any/all SSH drops),
        #   3. POLL for a completion sentinel using short, fault-tolerant SSH calls.
        # A dropped poll is harmless (retry next tick); teardown only fires on a
        # confirmed sentinel, budget exhaustion, or genuine give-up.
        #
        # Robust bootstrap: SSH can become ready BEFORE cloud-init finishes installing
        # uv, so we (a) wait for cloud-init, (b) poll for uv up to ~5 min and self-install
        # if still missing, (c) use `set -e` so a real failure aborts. A bash EXIT trap
        # ALWAYS writes the script's exit code to .matrix_exit so the poller can detect
        # both success and failure.
        script = (
            "#!/usr/bin/env bash\n"
            'finish() { echo "$?" > "$HOME/nymt/.matrix_exit"; }\n'
            "trap finish EXIT\n"
            "set -e\n"
            'export PATH="$HOME/.local/bin:$PATH"\n'
            "cloud-init status --wait >/dev/null 2>&1 || true\n"
            "for i in $(seq 1 60); do command -v uv >/dev/null 2>&1 && break; sleep 5; done\n"
            "command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh)\n"
            'export PATH="$HOME/.local/bin:$PATH"\n'
            f"cd {remote_root}/english-chinese-mt-experiment\n"
            "uv venv --python 3.12 .venv\n"
            ". .venv/bin/activate\n"
            "uv pip install -e '..'\n"
            "uv pip install -e .\n"
            f"{snmt_install}\n"
            f"{wandb_export}"
            f"{snmt_only_export}"
            f"python -u scripts/run_matrix_on_vm.py {spot_flag} {budget_arg} {nllb_flag} {onlynllb_flag} {only_flag}\n"
        )
        # Ship the script via base64 to dodge all quoting hazards, then launch it
        # fully detached (setsid + nohup, stdin from /dev/null) so it has no
        # controlling terminal and outlives every SSH session.
        #
        # CRITICAL NRMS lesson: NRMS can RESET *any* SSH connection at any moment
        # ("client_loop: send disconnect: Connection reset", ssh exit 255) — even a
        # 1-second command that already ran on the VM. So we must NEVER trust a
        # single SSH exit code here: a reset on the launch command would otherwise
        # raise -> finally -> teardown and destroy a VM whose training actually
        # started. Every step below is therefore best-effort + VERIFIED BY POLLING.
        b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
        write_cmd = (
            f"mkdir -p {remote_root} && "
            f"rm -f {remote_root}/.matrix_exit {remote_root}/run_all.log && "
            f"echo '{b64}' | base64 -d > {remote_root}/run_all.sh && "
            f"chmod +x {remote_root}/run_all.sh && echo WROTE_OK"
        )
        wrote = False
        for _ in range(8):
            _rc, out = _ssh_capture(ip, write_cmd, timeout=45)
            if "WROTE_OK" in out:
                wrote = True
                break
            # The write may have completed even if NRMS reset the SSH return;
            # verify independently before retrying.
            _rc2, out2 = _ssh_capture(
                ip, f"test -s {remote_root}/run_all.sh && echo HAVE_SCRIPT",
                timeout=30)
            if "HAVE_SCRIPT" in out2:
                wrote = True
                break
            try:
                h100.ensure_ssh_access()
            except Exception:
                pass
            time.sleep(10)
        if not wrote:
            raise SystemExit("could not write run_all.sh to VM (NSG/NRMS?)")

        # Launch detached. The idempotency guard is the run_all.log FILE, not a
        # pgrep: `pgrep -f run_all.sh` SELF-MATCHES the launching shell (whose own
        # command line contains "run_all.sh"), so the `||` always short-circuited and
        # setsid never ran -- the VM sat idle while we falsely believed training had
        # started. write_cmd rm's run_all.log first, so `[ -f run_all.log ]` is false
        # on the first attempt (we launch, which creates the log via the redirect) and
        # true on any NRMS-retry re-issue (we skip -> no double-start). No process
        # self-match is possible because we test a file, not a command line.
        launch_cmd = (
            f"cd {remote_root} && "
            f"([ -f {remote_root}/run_all.log ] || "
            f"setsid nohup bash run_all.sh > {remote_root}/run_all.log 2>&1 "
            f"< /dev/null &) ; echo launch_issued"
        )
        # HAVE_LOG (the redirect target) is the authoritative "a launch happened"
        # signal. The pgrep liveness checks use the `[.]` bracket trick so the
        # pattern text in pgrep's OWN argv can't match the regex (classic
        # `ps | grep '[p]attern'` self-match avoidance).
        verify_cmd = (
            f"test -f {remote_root}/run_all.log && echo HAVE_LOG; "
            f"pgrep -f 'run_matrix_on_vm[.]py' >/dev/null 2>&1 && echo MATRIX_PROC; "
            f"pgrep -f 'run_all[.]sh' >/dev/null 2>&1 && echo BOOT_PROC"
        )
        launched = False
        for _ in range(8):
            _ssh_capture(ip, launch_cmd, timeout=45)  # best-effort; reset is fine
            time.sleep(8)
            _rc, out = _ssh_capture(ip, verify_cmd, timeout=30)
            if any(tok in out for tok in ("HAVE_LOG", "MATRIX_PROC", "BOOT_PROC")):
                launched = True
                break
            try:
                h100.ensure_ssh_access()
            except Exception:
                pass
            time.sleep(10)
        if not launched:
            raise SystemExit("detached training did not start on VM")
        print("[run] matrix launched DETACHED on VM (billed part) ; polling ...",
              flush=True)

        # ---- completion-polling loop (resilient to SSH drops) ----
        start_wall = time.time()
        poll_secs = 60
        exit_code = None
        last_tail = ""
        while True:
            time.sleep(poll_secs)
            elapsed_h = (time.time() - start_wall) / 3600.0
            # Re-assert SSH access every tick: NRMS re-remediates the NSGs and would
            # otherwise black-hole our polls (and could reset flows). Cheap insurance.
            try:
                h100.ensure_ssh_access()
            except Exception:
                pass
            rc, out = _ssh_capture(
                ip,
                f"cat {remote_root}/.matrix_exit 2>/dev/null; echo '==='; "
                f"tail -n 3 {remote_root}/run_all.log 2>/dev/null",
                timeout=45,
            )
            if rc == 0:
                head, _, tail = out.partition("===")
                code_str = head.strip()
                tail = tail.strip()
                if tail and tail != last_tail:
                    for ln in tail.splitlines():
                        print(f"[vm] {ln}", flush=True)
                    last_tail = tail
                if code_str != "":
                    try:
                        exit_code = int(code_str.split()[0])
                    except ValueError:
                        exit_code = 1
                    print(f"[run] matrix finished on VM (exit={exit_code}) "
                          f"after ~{elapsed_h:.2f} h.", flush=True)
                    break
            else:
                print(f"[run] poll: SSH not reachable this tick (~{elapsed_h:.2f} h "
                      f"elapsed); retrying ...", flush=True)
            if elapsed_h >= budget:
                print(f"[run] BUDGET {budget} h reached (elapsed ~{elapsed_h:.2f} h) "
                      f"without completion; tearing down to stop spend.", flush=True)
                break
        if exit_code not in (0, None):
            print(f"[run] WARNING: matrix exited non-zero ({exit_code}); see "
                  f"run_all.log on (now-deleted) VM / mirrored Blob logs.", flush=True)
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
    p.add_argument("--only-nllb", action="store_true",
                   help="estimate ONLY the es<->nasa NLLB fine-tunes (skip En<->Zh)")
    p.set_defaults(func=cmd_plan)
    r = sub.add_parser("run")
    r.add_argument("--spot", action="store_true", default=config.USE_SPOT)
    r.add_argument("--yes", action="store_true")
    r.add_argument("--keep-up", action="store_true", help="do NOT teardown (debug)")
    r.add_argument("--max-budget-hours", type=float, default=None)
    r.add_argument("--no-nllb", action="store_true",
                   help="do NOT pack the es<->nasa NLLB fine-tunes onto the GPU")
    r.add_argument("--only-nllb", action="store_true",
                   help="run ONLY the es<->nasa NLLB fine-tunes (skip the En<->Zh "
                        "matrix). Used to re-run just the NLLB axis.")
    r.add_argument("--only", type=str, default=None,
                   help="comma-separated en-zh run id(s) to run (e.g. '1.7b-sentvocab'); "
                        "others are skipped. Useful for re-running a single failed cell.")
    r.set_defaults(func=cmd_run)
    args = ap.parse_args()
    if not args.cmd:
        args.func = cmd_plan
        args.spot = config.USE_SPOT
        args.no_nllb = False
        args.only_nllb = False
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
