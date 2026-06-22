"""Split the 1,935 PCIC items into per-model subagent chunks for the full run.

Model assignment is round-robin by item index (idx % len(MODELS)), IDENTICAL to
harness.py, so model is not confounded with concept/level. Each model's items are
then sliced into chunks of <= CHUNK items. One JSONL file per chunk is written to
batches/chunks/, plus a manifest CSV the orchestrator tracks in SQL.

    python chunk_build.py                  # CHUNK=50, N=5
    python chunk_build.py --chunk 40 --n 5
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
ITEMS = HERE / "items_a1_a2.csv"
CHUNK_DIR = HERE / "batches" / "chunks"
MANIFEST = HERE / "chunks_manifest.csv"

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=50, help="max items per chunk")
    ap.add_argument("--n", type=int, default=5, help="sentences per item")
    args = ap.parse_args()

    items = load_items()

    # Same round-robin assignment as harness.py.
    by_model: dict[str, list[dict]] = {m: [] for m in MODELS}
    for idx, it in enumerate(items):
        by_model[MODELS[idx % len(MODELS)]].append(it)

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    for old in CHUNK_DIR.glob("*.jsonl"):
        old.unlink()

    manifest: list[dict] = []
    for m in MODELS:
        rows = by_model[m]
        short = m.replace(".", "").replace("-", "")
        for ci, start in enumerate(range(0, len(rows), args.chunk), start=1):
            part = rows[start:start + args.chunk]
            chunk_id = f"{short}__{ci:02d}"
            path = CHUNK_DIR / f"{chunk_id}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for it in part:
                    rec = {
                        "item_id": it["item_id"],
                        "cefr_level": it["cefr_level"],
                        "kind": it["kind"],
                        "section": it["section"],
                        "concept": it["concept"],
                        "concept_group": it["concept_group"],
                        "model": m,
                        "n": args.n,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            manifest.append({
                "chunk_id": chunk_id,
                "model": m,
                "n_items": len(part),
                "n_target": len(part) * args.n,
                "batch_path": str(path),
                "out_path": str(HERE / "generated_full" / f"{chunk_id}.jsonl"),
            })

    with MANIFEST.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    total_items = sum(r["n_items"] for r in manifest)
    total_target = sum(r["n_target"] for r in manifest)
    print(f"items: {total_items}  chunks: {len(manifest)}  target sentences: {total_target}")
    for m in MODELS:
        cs = [r for r in manifest if r["model"] == m]
        print(f"  {m:24} {sum(c['n_items'] for c in cs):4} items  {len(cs)} chunks")
    print(f"manifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
