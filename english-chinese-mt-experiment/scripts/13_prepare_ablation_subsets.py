#!/usr/bin/env python3
"""Convert the reconstructed En-Zh subsets into the parquet splits the trainer reads.

``src/ecmt/training/trainer.py`` loads training data as parquet via
``splits_root / scale_filename_template.format(n=scale)`` plus a shared
``dev.parquet`` and ``english_probe.parquet``. This adapter produces those files
from the JSONL subsets emitted by ``12_build_enzh_dataset.py`` so the EXISTING
trainer runs unchanged on our level-based ablations.

Layout produced (under --splits-root, default data/splits):

    dev.parquet              held-out eval pairs (excluded from every train split)
    english_probe.parquet    English-only sentences for the forgetting probe
    scale_16391.parquet      sentence_only  minus dev
    scale_24174.parquet      sentence_plus_vocab minus dev

Columns: en, zh, source. Deterministic (fixed seed) so reruns are stable. No GPU.

Usage:
    python scripts/13_prepare_ablation_subsets.py
    python scripts/13_prepare_ablation_subsets.py --dev-size 500 --probe-size 2000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
SUBSETS = REPO / "data-en-zh" / "subsets"
DEFAULT_SPLITS = REPO / "data" / "splits"

SUBSET_FILES = {
    "sentence_only": SUBSETS / "sentence_only.jsonl",
    "sentence_plus_vocab": SUBSETS / "sentence_plus_vocab.jsonl",
}


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {
        "en": [r.get("en", "") for r in rows],
        "zh": [r.get("zh", "") for r in rows],
        "source": [r.get("source", "unknown") for r in rows],
    }
    pq.write_table(pa.table(cols), path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-root", default=str(DEFAULT_SPLITS))
    ap.add_argument("--dev-size", type=int, default=500)
    ap.add_argument("--probe-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    splits_root = Path(args.splits_root)
    rng = random.Random(args.seed)

    full = _read_jsonl(SUBSET_FILES["sentence_plus_vocab"])
    sent_only = _read_jsonl(SUBSET_FILES["sentence_only"])
    if not full:
        print("ERROR: sentence_plus_vocab.jsonl empty; run 12_build_enzh_dataset.py first.")
        return 2

    # Hold out dev from sentence-level rows (full-sentence eval), exclude from train.
    sentence_rows = [r for r in full if r.get("level") == "sentence"]
    dev_pool = sentence_rows[:]
    rng.shuffle(dev_pool)
    dev_n = min(args.dev_size, max(0, len(dev_pool) // 4))
    dev = dev_pool[:dev_n]
    dev_ids = {r.get("nasa_id") for r in dev}

    def _drop_dev(rows: list[dict]) -> list[dict]:
        return [r for r in rows if r.get("nasa_id") not in dev_ids]

    train_full = _drop_dev(full)
    train_sent = _drop_dev(sent_only)

    # English-only forgetting probe: English side of held-out + extra sentences.
    probe_pool = [{"en": r["en"]} for r in sentence_rows if r.get("en")]
    rng.shuffle(probe_pool)
    probe = probe_pool[: args.probe_size]

    _write_parquet(dev, splits_root / "dev.parquet")
    _write_parquet(probe, splits_root / "english_probe.parquet")
    _write_parquet(train_sent, splits_root / f"scale_{len(train_sent)}.parquet")
    _write_parquet(train_full, splits_root / f"scale_{len(train_full)}.parquet")

    manifest = {
        "splits_root": str(splits_root),
        "dev": len(dev),
        "english_probe": len(probe),
        "scale_sentence_only": len(train_sent),
        "scale_sentence_plus_vocab": len(train_full),
        "scale_filename_sentence_only": f"scale_{len(train_sent)}.parquet",
        "scale_filename_sentence_plus_vocab": f"scale_{len(train_full)}.parquet",
        "dev_excluded_from_train": True,
        "seed": args.seed,
    }
    (splits_root / "splits_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(
        "\nNOTE: pass these scale numbers to 06_train.py / ablations.yaml:\n"
        f"  sentence_only       -> --scale {len(train_sent)}\n"
        f"  sentence_plus_vocab -> --scale {len(train_full)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
