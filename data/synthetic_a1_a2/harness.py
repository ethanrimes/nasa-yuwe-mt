"""Build per-model generation batches from the PCIC A1/A2 item list.

Assigns each coverage item to a generation model round-robin (so model is not
confounded with concept/level), and writes one batch file per model under
batches/. Each batch line is a concept the assigned model must turn into N
diverse Spanish example sentences.

  python harness.py                 # full run: all 1,935 items
  python harness.py --pilot 10      # pilot: 10 items spanning levels/kinds
  python harness.py --pilot 10 --n 5

Scales unchanged from pilot to full run — only --pilot differs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
ITEMS = HERE / "items_a1_a2.csv"
BATCH_DIR = HERE / "batches"

MODELS = [
    "claude-sonnet-4.6",
    "gpt-5.4",
    "gpt-5.5",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
]


def load_items() -> list[dict]:
    with ITEMS.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def pilot_sample(items: list[dict], k: int) -> list[dict]:
    """Spread the pilot across (cefr_level, kind, section) strata."""
    buckets: dict[tuple, list[dict]] = {}
    for it in items:
        buckets.setdefault((it["cefr_level"], it["kind"], it["section"]), []).append(it)
    keys = sorted(buckets)
    out: list[dict] = []
    i = 0
    while len(out) < k:
        b = buckets[keys[i % len(keys)]]
        if b:
            out.append(b.pop(0))
        i += 1
        if i > k * len(keys):
            break
    return out[:k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0, help="limit to K items (0=full)")
    ap.add_argument("--n", type=int, default=5, help="sentences per item")
    args = ap.parse_args()

    items = load_items()
    if args.pilot:
        items = pilot_sample(items, args.pilot)

    BATCH_DIR.mkdir(exist_ok=True)
    for m in MODELS:
        (BATCH_DIR / f"{m}.jsonl").write_text("", encoding="utf-8")

    per_model: dict[str, int] = {m: 0 for m in MODELS}
    handles = {m: (BATCH_DIR / f"{m}.jsonl").open("a", encoding="utf-8") for m in MODELS}
    for idx, it in enumerate(items):
        model = MODELS[idx % len(MODELS)]
        rec = {
            "item_id": it["item_id"],
            "cefr_level": it["cefr_level"],
            "kind": it["kind"],
            "section": it["section"],
            "concept": it["concept"],
            "concept_group": it["concept_group"],
            "model": model,
            "n": args.n,
        }
        handles[model].write(json.dumps(rec, ensure_ascii=False) + "\n")
        per_model[model] += 1
    for h in handles.values():
        h.close()

    print(f"items: {len(items)}  N: {args.n}  -> {len(items) * args.n} target sentences")
    for m in MODELS:
        print(f"  {m:24} {per_model[m]} items -> {per_model[m] * args.n} sentences")
    print(f"batches written to {BATCH_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
