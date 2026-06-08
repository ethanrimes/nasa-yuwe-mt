"""E3: attribute translation gaps to (A) data, (B) forgetting, (C) capability, (D) decoding.

For each tier's BEST checkpoint we look at the words the model FAILED to produce
in real generation (reference content words absent from the hypothesis), and ask
*why* each one was missed, combining three independent signals:

  * freq_{tier}.json   -> how often the target word appears in that tier's TRAIN
                          data (data-availability signal).
  * tf/{tier}_step*.json-> whether the model could emit the word under teacher
                          forcing at the first / best / last snapshot
                          (knowledge + within-run forgetting signal).
  * gen/{tier}_step{best}.json -> the actual generation miss (the observed gap).

Taxonomy for every generation-missed target content word w:
  A  data_gap     : w is rare/absent in the tier's train target side (freq < LOW).
  B  forgetting   : w well-attested, recalled (TF top-1) at an earlier snapshot
                    but NOT at the latest snapshot -> the model lost it.
  D  decoding     : w well-attested, recalled (TF) at the BEST snapshot, yet
                    generation still missed it -> knowledge present, decoding/
                    exposure-bias gap (not a knowledge gap).
  C  capability   : w well-attested, never recalled (TF) at any snapshot -> the
                    data was there, it was never forgotten, the model simply
                    cannot produce it (optimization / capacity limit).

Priority A > B > D > C (root-cause ordering).

Writes assessment/data/attribution.json and a per-word CSV per tier.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

from common import (
    BEST_STEP,
    OUT_DIR,
    TIERS,
    representative_checkpoints,
    word_recall,
)

LOW = 5  # target-word train count below this == "data gap"


def load_json(p):
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def tf_recall_maps(tier_dir, tier):
    """Return {step: [ {word: recalled_bool} per sentence ]} for each direction.

    Structure: maps[direction][step] = list aligned to eval subset; each element
    is a dict word -> bool (OR-combined if a word repeats in the sentence).
    """
    reps = representative_checkpoints(tier_dir, tier)
    steps = [s for s, _ in reps]
    out = {"en2es": {}, "es2en": {}}
    for step in steps:
        f = OUT_DIR / "tf" / f"{tier}_step{step}.json"
        d = load_json(f)
        if d is None:
            continue
        for direction in ("en2es", "es2en"):
            per = []
            for cell in d.get(direction, []):
                wm = {}
                if cell:
                    for w, r in cell.get("words", []):
                        wm[w] = wm.get(w, False) or bool(r)
                per.append(wm)
            out[direction][step] = per
    return out, steps


def classify_word(w, freq_tgt, recall_series, best_step, last_step):
    """recall_series: {step: bool|None} for this (sentence,word)."""
    freq = freq_tgt.get(w, 0)
    if freq < LOW:
        return "A_data_gap", freq
    early_true = any(
        v is True for s, v in recall_series.items() if s < last_step
    )
    last_v = recall_series.get(last_step)
    if early_true and last_v is False:
        return "B_forgetting", freq
    if recall_series.get(best_step) is True:
        return "D_decoding", freq
    if all(v is not True for v in recall_series.values()):
        return "C_capability", freq
    # well-attested, recalled somewhere but not at best/last cleanly -> decoding-ish
    return "D_decoding", freq


def main():
    gen_best = {}
    summ = load_json(OUT_DIR / "gen_metrics_best.json")
    if summ is None:
        raise SystemExit("Run generation (--selection best) first.")

    report = {"LOW_freq_threshold": LOW, "tiers": {}}
    for tier_dir, tier in TIERS.items():
        best = BEST_STEP[tier]
        gen = load_json(OUT_DIR / "gen" / f"{tier}_step{best}.json")
        if gen is None:
            print(f"skip {tier}: no gen file")
            continue
        freq = load_json(OUT_DIR / f"freq_{tier}.json")
        tf_maps, steps = tf_recall_maps(tier_dir, tier)
        last_step = max(steps) if steps else best

        tier_rep = {}
        for direction, tgt_lang, freq_key in (
            ("en2es", "es", "es"),
            ("es2en", "en", "en"),
        ):
            d = gen["directions"][direction]
            hyps, refs = d["hyps"], d["ref"]
            freq_tgt = freq[freq_key] if freq else {}
            tf_dir = tf_maps[direction]

            cats = Counter()
            rows = []
            total_missed = 0
            for i, (h, r) in enumerate(zip(hyps, refs)):
                _, _, miss = word_recall(h, r, tgt_lang)
                for w in miss:
                    total_missed += 1
                    series = {}
                    for step in steps:
                        per = tf_dir.get(step)
                        series[step] = per[i].get(w) if (per and i < len(per)) else None
                    cat, fr = classify_word(w, freq_tgt, series, best, last_step)
                    cats[cat] += 1
                    rows.append({"sent": i, "word": w, "freq": fr, "category": cat})
            tier_rep[direction] = {
                "total_missed_word_tokens": total_missed,
                "categories": dict(cats),
                "category_pct": {
                    k: round(100 * v / max(total_missed, 1), 1) for k, v in cats.items()
                },
            }
            # CSV of every missed word for transparency
            cpath = OUT_DIR / "attribution" / f"{tier}_{direction}.csv"
            cpath.parent.mkdir(parents=True, exist_ok=True)
            with open(cpath, "w", newline="", encoding="utf-8") as fh:
                wr = csv.DictWriter(fh, fieldnames=["sent", "word", "freq", "category"])
                wr.writeheader()
                wr.writerows(rows)
        report["tiers"][tier] = tier_rep
        print(f"{tier}: " + " | ".join(
            f"{dir_}={tier_rep[dir_]['category_pct']}" for dir_ in ("en2es", "es2en")
        ))

    json.dump(report, open(OUT_DIR / "attribution.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\nwrote attribution.json")


if __name__ == "__main__":
    main()
