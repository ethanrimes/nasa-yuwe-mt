#!/usr/bin/env python3
"""Back up the en-spanish experiment-results to Azure Blob (experiments/en-es/).

The local results tree is ~93 GB, dominated by per-checkpoint ``optimizer.pt``
(~1.4 GB each x 42 checkpoints). Uploading all of it is wasteful: optimizer state
is only needed to *resume* a specific checkpoint, and the local copy on Q:\\ is
never deleted, so any single optimizer.pt can be pushed on demand later.

Selective policy (the default):
  * ALL small artifacts (metrics/ results/ logs/ configs/ runs/ + loose top-level
    files) — full fidelity, tiny.
  * EVERY checkpoint's weights + tokenizer + trainer_state (everything except
    optimizer.pt) — so any snapshot can be evaluated or "taken".
  * optimizer.pt ONLY for the final checkpoint of each tier — so the latest
    snapshot per tier is fully resumable.  (~28.5 GB weights + ~7 GB optimizers.)

Use --all to push the full 93 GB, or --include-all-optimizers to keep every
optimizer.pt. Idempotent: skips blobs that already exist with the same size
(unless --overwrite). Dry-run by default; pass --execute to actually upload.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]            # english-spanish-mt-experiment
sys.path.insert(0, str((REPO.parent / "shared").resolve().parent))

from nymt_shared import blob, config  # noqa: E402

RESULTS = REPO / "experiment-results"
PREFIX = config.BLOB_LAYOUT["exp_en_es"]               # "experiments/en-es/"
SMALL_DIRS = ["metrics", "results", "logs", "configs", "runs", "src"]
CKPT_RE = re.compile(r"checkpoint-(\d+)$")


def final_checkpoints() -> set[Path]:
    """Latest checkpoint-N dir per tier under checkpoints/<tier>/."""
    finals: set[Path] = set()
    ckpt_root = RESULTS / "checkpoints"
    if not ckpt_root.exists():
        return finals
    for tier in ckpt_root.iterdir():
        if not tier.is_dir():
            continue
        cks = [d for d in tier.iterdir() if d.is_dir() and CKPT_RE.search(d.name)]
        if cks:
            finals.add(max(cks, key=lambda d: int(CKPT_RE.search(d.name).group(1))))
    return finals


def plan_uploads(args) -> list[tuple[Path, str]]:
    finals = final_checkpoints()
    uploads: list[tuple[Path, str]] = []

    def add(p: Path):
        rel = p.relative_to(RESULTS).as_posix()
        uploads.append((p, PREFIX + rel))

    # Small dirs + loose top-level files (full).
    for d in SMALL_DIRS:
        dp = RESULTS / d
        if dp.exists():
            for f in dp.rglob("*"):
                if f.is_file():
                    add(f)
    for f in RESULTS.iterdir():
        if f.is_file():
            add(f)

    # Checkpoints (selective).
    ckpt_root = RESULTS / "checkpoints"
    if ckpt_root.exists():
        for f in ckpt_root.rglob("*"):
            if not f.is_file():
                continue
            if f.name == "optimizer.pt" and not args.all and not args.include_all_optimizers:
                if f.parent not in finals:
                    continue  # skip non-final optimizer states
            add(f)
    return uploads


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="actually upload (default: dry-run)")
    ap.add_argument("--all", action="store_true", help="upload the full 93 GB tree")
    ap.add_argument("--include-all-optimizers", action="store_true",
                    help="keep every optimizer.pt (not just final-per-tier)")
    ap.add_argument("--overwrite", action="store_true", help="re-upload even if same-size blob exists")
    args = ap.parse_args()

    if not RESULTS.exists():
        print(f"ERROR: {RESULTS} not found")
        return 1

    uploads = plan_uploads(args)
    total = sum(p.stat().st_size for p, _ in uploads)
    print(f"planned {len(uploads)} files, {total/1024**3:.1f} GB -> {PREFIX}")
    if not args.execute:
        for p, name in uploads[:8]:
            print(f"  {p.stat().st_size/1024**2:8.1f}MB  {name}")
        print("  ... (dry-run; pass --execute to upload)")
        return 0

    existing = {}
    try:
        for b in blob.list_blobs(PREFIX):
            existing[b] = True
    except Exception:
        pass

    done = skipped = 0
    sent_bytes = 0
    for i, (p, name) in enumerate(uploads, 1):
        if not args.overwrite and name in existing:
            skipped += 1
            continue
        blob.upload_file(p, name, overwrite=True)
        done += 1
        sent_bytes += p.stat().st_size
        if done % 20 == 0:
            print(f"  [{i}/{len(uploads)}] uploaded {done} ({sent_bytes/1024**3:.1f} GB), skipped {skipped}",
                  flush=True)
    print(f"DONE: uploaded {done}, skipped {skipped}, {sent_bytes/1024**3:.1f} GB -> {PREFIX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
