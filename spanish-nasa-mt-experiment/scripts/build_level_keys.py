#!/usr/bin/env python3
"""Build the committed ``data/sentence_keys.json`` level map (run locally, once).

The Spanish<->Nasa-Yuwe corpus has no native sentence/vocabulary label, but the
En<->Zh proxy ablation already split the *same* 24,229 pairs into ``sentence`` vs
``vocabulary`` levels (``data-en-zh/subsets/sentence_plus_vocab.jsonl`` carries a
``level`` field + a ``nasa_id`` back-reference to the original pair). We reuse that
authoritative labelling so the NLLB ``sentence_only`` / ``sentence_plus_vocab``
subsets line up exactly with the En<->Zh ones.

Rather than key on ``nasa_id`` (row order, fragile if the Blob copy is ever
re-emitted in a different order), we key on the *content hash* of each pair
(:func:`snmt.data.level_hash`). The output is a small JSON listing the content
hashes of every ``sentence``-level pair; ``sentence_plus_vocab`` is simply the whole
corpus, so only the sentence set needs storing.

Inputs (local only — these live under the en-zh experiment, NOT rsynced to the VM):
  - en-zh subset  : data-en-zh/subsets/sentence_plus_vocab.jsonl  (level + nasa_id)
  - nasa source   : data-en-zh/source/nasa_yuwe_parallel_dataset.jsonl (nasa_id order)

Output (committed, shipped to the VM with the snmt pkg):
  - spanish-nasa-mt-experiment/data/sentence_keys.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]            # spanish-nasa-mt-experiment/
sys.path.insert(0, str(PKG / "src"))

from snmt import data as snmt_data  # noqa: E402

REPO = PKG.parent
EN_ZH_SUBSET = REPO / "english-chinese-mt-experiment" / "data-en-zh" / "subsets" / "sentence_plus_vocab.jsonl"
NASA_SOURCE = REPO / "english-chinese-mt-experiment" / "data-en-zh" / "source" / "nasa_yuwe_parallel_dataset.jsonl"
OUT = PKG / "data" / "sentence_keys.json"

SENTENCE_LEVEL = "sentence"


def main() -> int:
    if not EN_ZH_SUBSET.exists():
        raise SystemExit(f"missing en-zh subset file: {EN_ZH_SUBSET}")
    if not NASA_SOURCE.exists():
        raise SystemExit(f"missing nasa source file: {NASA_SOURCE}")

    # nasa_id -> level, from the authoritative en-zh subset labelling.
    level_by_nasa_id: dict[int, str] = {}
    for r in snmt_data.load_jsonl(EN_ZH_SUBSET):
        nid = r.get("nasa_id")
        lvl = r.get("level")
        if nid is None or lvl is None:
            continue
        level_by_nasa_id[int(nid)] = str(lvl)

    nasa_rows = snmt_data.load_jsonl(NASA_SOURCE)

    sentence_keys: set[str] = set()
    levels_seen: dict[str, int] = {}
    n_labelled = 0
    for i, row in enumerate(nasa_rows):
        lvl = level_by_nasa_id.get(i)
        if lvl is None:
            continue  # unlabelled row (shouldn't happen for the full corpus)
        n_labelled += 1
        levels_seen[lvl] = levels_seen.get(lvl, 0) + 1
        if lvl == SENTENCE_LEVEL:
            sentence_keys.add(snmt_data.level_hash(row))

    payload = {
        "description": "Content-hash keys of sentence-level pairs (sentence_only subset). "
                       "sentence_plus_vocab = the whole corpus.",
        "hash_len": snmt_data.LEVEL_HASH_LEN,
        "sentence_level": SENTENCE_LEVEL,
        "n_nasa_rows": len(nasa_rows),
        "n_labelled": n_labelled,
        "levels_seen": levels_seen,
        "n_sentence_keys": len(sentence_keys),
        "keys": sorted(sentence_keys),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=0, ensure_ascii=True), encoding="utf-8")

    print(f"[build-keys] nasa_rows={len(nasa_rows):,} labelled={n_labelled:,} "
          f"levels={levels_seen} sentence_keys={len(sentence_keys):,}")
    print(f"[build-keys] coverage={n_labelled/len(nasa_rows):.4f}")
    print(f"[build-keys] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
