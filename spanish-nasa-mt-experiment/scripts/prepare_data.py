#!/usr/bin/env python3
"""Build the Spanish<->Nasa-Yuwe train/dev/test splits for the NLLB fine-tunes.

Source resolution order (first that exists wins):
  1. ``--source`` path, if given.
  2. The local en-zh source copy
     (``english-chinese-mt-experiment/data-en-zh/source/nasa_yuwe_parallel_dataset.jsonl``)
     — present locally, but NOT rsynced to the H100.
  3. Azure Blob ``data/nasa_yuwe_parallel_dataset.jsonl`` (``config.NASA_DATASET_BLOB``),
     downloaded via AAD. This is the path used ON the VM, where (2) is absent.

Outputs parquet splits + ``splits_manifest.json`` under ``--out-dir``
(default ``<pkg>/data/splits``). Pure data prep — no GPU, safe to run locally.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]          # spanish-nasa-mt-experiment/
sys.path.insert(0, str(PKG / "src"))

from snmt import data as snmt_data  # noqa: E402

try:
    from nymt_shared import config as shared_config
except Exception:  # nymt_shared not installed (rare in pure-local dev)
    shared_config = None  # type: ignore


def _local_source() -> Path | None:
    if shared_config is not None:
        p = shared_config.EN_ZH_DIR / "data-en-zh" / "source" / "nasa_yuwe_parallel_dataset.jsonl"
        if p.exists():
            return p
    # Fallback relative guess if shared config is unavailable.
    guess = PKG.parent / "english-chinese-mt-experiment" / "data-en-zh" / "source" / "nasa_yuwe_parallel_dataset.jsonl"
    return guess if guess.exists() else None


def _download_from_blob(dest: Path) -> Path:
    if shared_config is None:
        raise RuntimeError("nymt_shared not available; cannot download from Blob")
    from nymt_shared import blob

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[prepare] downloading {shared_config.NASA_DATASET_BLOB} -> {dest}", flush=True)
    blob.download_file(shared_config.NASA_DATASET_BLOB, dest)
    return dest


def resolve_source(explicit: str | None, cache_dir: Path) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    local = _local_source()
    if local is not None:
        print(f"[prepare] using local source {local}", flush=True)
        return local
    # On the VM: pull from Blob.
    return _download_from_blob(cache_dir / "nasa_yuwe_parallel_dataset.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="explicit path to the parallel jsonl")
    ap.add_argument("--out-dir", default=str(PKG / "data" / "splits"))
    ap.add_argument("--cache-dir", default=str(PKG / "data" / "source"))
    ap.add_argument("--dev-fraction", type=float, default=0.03)
    ap.add_argument("--test-fraction", type=float, default=0.03)
    args = ap.parse_args()

    src = resolve_source(args.source, Path(args.cache_dir))
    rows = snmt_data.load_jsonl(src)
    cfg = snmt_data.SplitConfig(dev_fraction=args.dev_fraction, test_fraction=args.test_fraction)
    manifest = snmt_data.build_splits(rows, args.out_dir, cfg)

    print("[prepare] splits:", manifest["counts"], flush=True)
    print(f"[prepare] train pairs={manifest['train_pairs']:,} "
          f"(bidir examples={manifest['train_examples_bidir']:,})", flush=True)
    print(f"[prepare] manifest -> {Path(args.out_dir) / 'splits_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
