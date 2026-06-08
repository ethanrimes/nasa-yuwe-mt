"""Background snapshot mirror — runs ON the H100 VM, pushes to Azure Blob.

Every `--interval` seconds it uploads any new/changed files under the watched
directories (checkpoints + metrics + logs + samples) to
`experiments/en-zh/<run-id>/...`. Because uploads are idempotent (size-checked),
an early `h100 down` loses at most one interval of progress.

Auth is AAD-only via `DefaultAzureCredential` (this storage account forbids key auth):
on the VM it resolves to the VM's system-assigned **managed identity** (granted
"Storage Blob Data Contributor" by `h100.up()`); locally it resolves to the Azure CLI
login. No connection string is used.

CLI (on the VM):
    python -m nymt_shared.mirror --run-id 360m-sentence --root /home/azureuser/nymt \
        --interval 300
    python -m nymt_shared.mirror --run-id 360m-sentence --root . --once   # one pass
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from . import blob, config

# Subdirs of a run that we mirror, mapped to their blob sub-prefix.
WATCHED = ["checkpoints", "metrics", "logs", "samples"]


def run_prefix(run_id: str) -> str:
    return f"{config.BLOB_LAYOUT['exp_en_zh']}{run_id}/"


def mirror_once(run_id: str, root: Path, verbose: bool = True) -> int:
    """One mirroring pass over all watched subdirs of `root`. Returns files sent."""
    total = 0
    base = run_prefix(run_id)
    for sub in WATCHED:
        local = root / sub
        if not local.exists():
            continue
        total += blob.mirror_dir(local, base + sub, only_new=True, verbose=verbose)
    return total


def watch(run_id: str, root: Path, interval: int = 300) -> None:
    stop = {"now": False}

    def _handler(signum, frame):
        stop["now"] = True
        print(f"[mirror] signal {signum}; final flush then exit")

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    print(f"[mirror] watching {root} -> {run_prefix(run_id)} every {interval}s")
    while not stop["now"]:
        try:
            n = mirror_once(run_id, root, verbose=False)
            print(f"[mirror] flushed {n} file(s) at {time.strftime('%H:%M:%S')}")
        except Exception as e:  # never let the mirror die silently
            print(f"[mirror] WARN: {e}")
        for _ in range(interval):
            if stop["now"]:
                break
            time.sleep(1)
    # final flush
    n = mirror_once(run_id, root, verbose=False)
    print(f"[mirror] final flush {n} file(s); bye")


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="nymt_shared.mirror")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--root", default=".", help="run dir holding checkpoints/metrics/...")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    if args.once:
        n = mirror_once(args.run_id, root)
        print(f"uploaded {n} file(s)")
    else:
        watch(args.run_id, root, args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
