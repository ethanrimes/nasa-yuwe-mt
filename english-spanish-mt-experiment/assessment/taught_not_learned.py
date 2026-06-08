"""E6 - "Taught but not learned": direct evidence for a model-capacity ceiling.

For each tier's BEST checkpoint we aggregate the teacher-forced (TF) per-word
records: for every TARGET content word we count how often it appears as a
reference token and how often the model's top-1 prediction recalled it *under
teacher forcing* (gold source + gold left-context given). We join that with the
word's TRAINING frequency.

A word that is FREQUENT in training (taught), appears MANY times as a reference,
and is recalled ~0% of the time under teacher forcing is the cleanest possible
"taught but not learned" instance: the data was present, nothing was forgotten,
the model is being handed the gold prefix and still cannot emit the next word.
That isolates a capacity/optimisation limit from data and from decoding/exposure
bias.

Outputs:
  data/taught_not_learned.json   - per-tier counts + ranked word lists
  (printed) a compact table per tier
Run: uv run python assessment/taught_not_learned.py
"""
from __future__ import annotations

import json
from collections import defaultdict

from common import OUT_DIR, TIER_PAIRS, BEST_STEP, write_json

# A word is "well taught" if it occurs at least this many times in training.
TAUGHT_FREQ = 100
# Need at least this many TF reference occurrences to trust the recall rate.
MIN_OCC = 4
# "Not learned" = TF recall rate at or below this.
NOT_LEARNED_RATE = 0.0
# Softer band for reporting.
WEAK_RATE = 0.10

TIER_ORDER = ["T10k", "T50k", "T100k", "T500k", "T1M"]
DIRS = (("en2es", "es"), ("es2en", "en"))


def lj(p):
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def aggregate(tf_records):
    """word -> [occurrences, recalled] over all sentences."""
    agg = defaultdict(lambda: [0, 0])
    for rec in tf_records:
        if not rec:
            continue
        for w, ok in rec["words"]:
            a = agg[w]
            a[0] += 1
            if ok:
                a[1] += 1
    return agg


def main():
    report = {
        "params": {
            "taught_freq_threshold": TAUGHT_FREQ,
            "min_tf_occurrences": MIN_OCC,
            "not_learned_rate": NOT_LEARNED_RATE,
            "weak_rate": WEAK_RATE,
            "note": "TF = teacher forcing (gold source + gold prefix). recall_rate "
                    "is fraction of reference occurrences where the model's top-1 "
                    "was the gold word.",
        },
        "tiers": {},
    }

    for tier in TIER_ORDER:
        step = BEST_STEP[tier]
        tf = lj(OUT_DIR / "tf" / f"{tier}_step{step}.json")
        freq = lj(OUT_DIR / f"freq_{tier}.json")
        if not tf or not freq:
            continue
        tier_out = {"step": step, "pairs": TIER_PAIRS[tier], "directions": {}}
        for direction, tgt in DIRS:
            agg = aggregate(tf[direction])
            fmap = freq[tgt]
            # restrict to well-taught words with enough TF samples
            cand = []
            for w, (occ, rec) in agg.items():
                f = fmap.get(w, 0)
                if f >= TAUGHT_FREQ and occ >= MIN_OCC:
                    cand.append({
                        "word": w, "train_freq": f, "tf_occ": occ,
                        "tf_recalled": rec, "recall_rate": round(rec / occ, 3),
                    })
            taught = [c for c in cand if c["recall_rate"] <= NOT_LEARNED_RATE]
            weak = [c for c in cand if c["recall_rate"] <= WEAK_RATE]
            taught.sort(key=lambda c: (-c["train_freq"], -c["tf_occ"]))
            weak.sort(key=lambda c: (-c["train_freq"], -c["tf_occ"]))
            tier_out["directions"][direction] = {
                "n_taught_words_evaluated": len(cand),
                "n_taught_not_learned": len(taught),       # rate == 0
                "n_taught_weak": len(weak),                # rate <= 10%
                "pct_taught_not_learned": round(100 * len(taught) / max(len(cand), 1), 1),
                "top_taught_not_learned": taught[:25],
                "top_weak": weak[:25],
            }
        report["tiers"][tier] = tier_out

        # console summary
        for direction, _ in DIRS:
            d = tier_out["directions"][direction]
            ex = ", ".join(
                f"{c['word']}({c['train_freq']}x,{c['tf_recalled']}/{c['tf_occ']})"
                for c in d["top_taught_not_learned"][:6]
            )
            print(f"{tier} {direction}: {d['n_taught_not_learned']}/"
                  f"{d['n_taught_words_evaluated']} well-taught words NEVER recalled "
                  f"under TF ({d['pct_taught_not_learned']}%) | {ex}")

    write_json(report, OUT_DIR / "taught_not_learned.json")
    print(f"\nwrote {OUT_DIR / 'taught_not_learned.json'}")


if __name__ == "__main__":
    main()
