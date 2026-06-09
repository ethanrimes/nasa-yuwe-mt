"""Held-out chrF + BLEU for the 6 es<->nasa NLLB runs (CPU, reference-based).

The training harness computes only *loss* during the packed H100 waves; the
reference-based generation metric (``snmt.train.evaluate_bleu``) is a separate,
slower post-hoc pass that was never run. This script runs it on CPU against the
SAME deterministic held-out ``test`` split each model was trained to exclude, for
BOTH directions (spa->pbb, pbb->spa).

This is the objective "how good is my data" signal: real Nasa-Yuwe references,
corpus chrF/BLEU via sacrebleu. nasa->es (Spanish output) is the human-verifiable
direction.

CPU is slow for 3.3B, so we evaluate a fixed-size subset (LIMIT pairs, uniform
across all models for a fair cross-size comparison). chrF over ~100+ sentences is
a stable corpus estimate; raise LIMIT for the final number once a GPU is free.

Outputs analysis/heldout_bleu.json . Prints only numeric scores (never Nasa text,
which the cp1252 console can't encode).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "spanish-nasa-mt-experiment" / "src"))

from snmt.train import evaluate_bleu  # noqa: E402

CACHE = REPO / "analysis" / "models_cache"
SPLITS_ROOT = REPO / "spanish-nasa-mt-experiment" / "data" / "splits"
OUT = REPO / "analysis" / "heldout_bleu.json"

# run id (== cached model dir) -> split subset dir
RUNS = {
    "nllb-600m-sent": "sentence_only",
    "nllb-600m-sentvocab": "sentence_plus_vocab",
    "nllb-1.3b-sent": "sentence_only",
    "nllb-1.3b-sentvocab": "sentence_plus_vocab",
    "nllb-3.3b-sent": "sentence_only",
    "nllb-3.3b-sentvocab": "sentence_plus_vocab",
}

LIMIT = int(os.environ.get("HELDOUT_LIMIT", "120"))
MAX_NEW = int(os.environ.get("HELDOUT_MAX_NEW", "128"))
BATCH = int(os.environ.get("HELDOUT_BATCH", "8"))
# optional substring filter so we can run small models first, 3.3b later
ONLY = [t for t in os.environ.get("HELDOUT_ONLY", "").split(",") if t]


def _selected(run_id: str) -> bool:
    return (not ONLY) or any(t in run_id for t in ONLY)


def main() -> None:
    # merge with any prior partial results so re-runs accumulate
    results: dict = {}
    if OUT.exists():
        results = json.loads(OUT.read_text(encoding="utf-8"))
    results.setdefault("_meta", {})
    results["_meta"].update({"limit": LIMIT, "max_new_tokens": MAX_NEW, "batch_size": BATCH})
    results.setdefault("runs", {})

    for run_id, subset in RUNS.items():
        if not _selected(run_id):
            continue
        model_dir = CACHE / run_id
        splits_dir = SPLITS_ROOT / subset
        if not (model_dir / "config.json").exists():
            print(f"[skip] {run_id}: no cached model", flush=True)
            continue
        if not (splits_dir / "test.parquet").exists():
            print(f"[skip] {run_id}: no test split at {splits_dir}", flush=True)
            continue

        print(f"[eval] {run_id} (subset={subset}, limit={LIMIT}) ...", flush=True)
        t0 = time.time()
        scores = evaluate_bleu(
            model_dir=model_dir,
            splits_dir=splits_dir,
            split="test",
            max_new_tokens=MAX_NEW,
            batch_size=BATCH,
            limit=LIMIT,
        )
        dt = round(time.time() - t0, 1)
        results["runs"][run_id] = {"subset": subset, "wall_s": dt, "scores": scores}
        # print numeric only (safe for cp1252)
        for direction, sc in scores.items():
            print(f"    {direction:18s} chrF={sc['chrf']:.2f}  BLEU={sc['bleu']:.2f}  n={sc['n']}", flush=True)
        print(f"    done in {dt}s", flush=True)
        OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
