"""E0: per-tier word-frequency tables from each tier's training set.

Used by the error-attribution experiment to decide whether a source word the
model mistranslated was actually *present* in that tier's training data
(data-gap evidence) vs. well-attested (so the error is capability/forgetting).

Streams train.jsonl so memory stays flat even for the 1M-line tier.
Writes assessment/data/freq_<tier>.json = {"en": {...}, "es": {...}, "n_pairs": N}
Only keeps words with count >= MIN_KEEP to bound file size.
"""
from __future__ import annotations

import json
import time
from collections import Counter

from common import OUT_DIR, PROC_DIR, TIERS, tokenize_words, write_json

MIN_KEEP = 2  # drop hapax-of-noise to keep files small; 1-count looked up as absent


def build_for_tier(tier_dir: str, tier: str):
    path = PROC_DIR / tier_dir / "train.jsonl"
    en_c: Counter = Counter()
    es_c: Counter = Counter()
    n = 0
    t0 = time.time()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            en_c.update(tokenize_words(r.get("en", "")))
            es_c.update(tokenize_words(r.get("es", "")))
            n += 1
            if n % 100000 == 0:
                print(f"  [{tier}] {n} pairs  ({time.time()-t0:.0f}s)", flush=True)
    en_keep = {w: c for w, c in en_c.items() if c >= MIN_KEEP}
    es_keep = {w: c for w, c in es_c.items() if c >= MIN_KEEP}
    out = {
        "tier": tier,
        "n_pairs": n,
        "en_vocab": len(en_c),
        "es_vocab": len(es_c),
        "en": en_keep,
        "es": es_keep,
    }
    write_json(out, OUT_DIR / f"freq_{tier}.json")
    print(
        f"[{tier}] pairs={n}  en_vocab={len(en_c)} (kept {len(en_keep)})  "
        f"es_vocab={len(es_c)} (kept {len(es_keep)})  {time.time()-t0:.0f}s",
        flush=True,
    )


def main():
    for tier_dir, tier in TIERS.items():
        build_for_tier(tier_dir, tier)


if __name__ == "__main__":
    main()
