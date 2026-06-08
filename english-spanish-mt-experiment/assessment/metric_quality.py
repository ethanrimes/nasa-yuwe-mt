"""E4: how well do BLEU/chrF actually measure translation ability?

Consumes the generation stores (best ckpt, out-of-domain FLORES) and, when
present, the in-domain val-set generations, and quantifies the metrics'
trustworthiness:

  1. Corpus table   : BLEU vs chrF vs chrF++ vs content-word-recall, per tier /
                      direction / domain.
  2. Agreement      : Pearson + Spearman between sentence-level BLEU, chrF and
                      word-recall (do the metrics even agree with each other?).
  3. Single-ref tax : fraction of sentences that are semantically adequate
                      (word_recall high) yet BLEU-penalised (sent BLEU low) ->
                      direct evidence of the single-reference limitation, with
                      curated examples.
  4. Domain effect  : in-domain (val) vs out-domain (FLORES) BLEU gap -> shows
                      the score reflects domain match, not only ability.
  5. Asymmetry      : en2es vs es2en, and whether low en2es BLEU understates a
                      model that still recalls most content words.

Writes assessment/data/metric_quality.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from common import OUT_DIR, TIERS, BEST_STEP


def load_json(p: Path):
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def pearson(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (dx * dy) if dx and dy else None


def _ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return None
    return pearson(_ranks(x), _ranks(y))


def collect(selection):
    """Return list of (tier, direction, store_dir_dict, src_list)."""
    summ = load_json(OUT_DIR / f"gen_metrics_{selection}.json")
    if summ is None:
        return [], None
    items = []
    tag = "indomain_" if selection == "indomain" else ""
    for rec in summ:
        tier, step = rec["tier"], rec["step"]
        store = load_json(OUT_DIR / "gen" / f"{tag}{tier}_step{step}.json")
        if store is None:
            continue
        for direction in ("en2es", "es2en"):
            src = store["src_en"] if direction == "en2es" else store["src_es"]
            items.append((tier, direction, store["directions"][direction], src, rec))
    return items, summ


def main():
    out = {"corpus": [], "agreement": {}, "single_ref": {}, "domain_gap": [],
           "asymmetry": []}

    best_items, best_summ = collect("best")
    if not best_summ:
        raise SystemExit("Run generation (--selection best) first.")
    indom_items, _ = collect("indomain")

    # 1. corpus table -------------------------------------------------------
    for rec in best_summ:
        for direction in ("en2es", "es2en"):
            m = rec[direction]
            out["corpus"].append({
                "tier": rec["tier"], "direction": direction, "domain": "out",
                "bleu": round(m["bleu"], 2), "chrf": round(m["chrf"], 2),
                "chrf_pp": round(m["chrf_pp"], 2),
                "word_recall": round(100 * m["word_recall"], 1),
            })

    # 2/3. pool sentence-level signals across all best-ckpt out-domain items
    allb, allc, allr = [], [], []
    sr = {"en2es": {"adequate_low_bleu": 0, "total": 0, "examples": []},
          "es2en": {"adequate_low_bleu": 0, "total": 0, "examples": []}}
    for tier, direction, d, src, rec in best_items:
        sb, sc = d["sent_bleu"], d["sent_chrf"]
        wr = [100 * x for x in d["word_recall"]]
        allb += sb; allc += sc; allr += wr
        for i in range(len(sb)):
            sr[direction]["total"] += 1
            if wr[i] >= 70 and sb[i] < 15:
                sr[direction]["adequate_low_bleu"] += 1
                if len(sr[direction]["examples"]) < 8:
                    sr[direction]["examples"].append({
                        "tier": tier, "src": src[i], "ref": d["ref"][i],
                        "hyp": d["hyps"][i], "sent_bleu": round(sb[i], 1),
                        "sent_chrf": round(sc[i], 1), "word_recall": round(wr[i], 1),
                    })
    out["agreement"] = {
        "bleu_vs_chrf": {"pearson": pearson(allb, allc), "spearman": spearman(allb, allc)},
        "bleu_vs_wordrecall": {"pearson": pearson(allb, allr), "spearman": spearman(allb, allr)},
        "chrf_vs_wordrecall": {"pearson": pearson(allc, allr), "spearman": spearman(allc, allr)},
        "n_sentences": len(allb),
    }
    for direction in ("en2es", "es2en"):
        t = sr[direction]["total"]
        sr[direction]["pct_adequate_low_bleu"] = round(100 * sr[direction]["adequate_low_bleu"] / max(t, 1), 1)
    out["single_ref"] = sr

    # 4. domain gap (in vs out) --------------------------------------------
    if indom_items:
        indom_by = {(t, d): rec for (t, d, _, _, rec) in indom_items}
        for rec in best_summ:
            for direction in ("en2es", "es2en"):
                key = (rec["tier"], direction)
                if key in indom_by:
                    inm = indom_by[key][direction]
                    outm = rec[direction]
                    out["domain_gap"].append({
                        "tier": rec["tier"], "direction": direction,
                        "in_bleu": round(inm["bleu"], 2), "out_bleu": round(outm["bleu"], 2),
                        "bleu_gap": round(inm["bleu"] - outm["bleu"], 2),
                        "in_chrf": round(inm["chrf"], 2), "out_chrf": round(outm["chrf"], 2),
                        "in_word_recall": round(100 * inm["word_recall"], 1),
                        "out_word_recall": round(100 * outm["word_recall"], 1),
                    })

    # 5. direction asymmetry ------------------------------------------------
    for rec in best_summ:
        e, s = rec["en2es"], rec["es2en"]
        out["asymmetry"].append({
            "tier": rec["tier"],
            "en2es_bleu": round(e["bleu"], 2), "es2en_bleu": round(s["bleu"], 2),
            "bleu_ratio_es2en_over_en2es": round(s["bleu"] / e["bleu"], 2) if e["bleu"] else None,
            "en2es_word_recall": round(100 * e["word_recall"], 1),
            "es2en_word_recall": round(100 * s["word_recall"], 1),
            "en2es_chrf": round(e["chrf"], 2), "es2en_chrf": round(s["chrf"], 2),
        })

    json.dump(out, open(OUT_DIR / "metric_quality.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("wrote metric_quality.json")
    print("agreement:", json.dumps(out["agreement"], indent=2))
    for direction in ("en2es", "es2en"):
        print(f"single-ref {direction}: "
              f"{sr[direction]['pct_adequate_low_bleu']}% adequate-but-low-BLEU "
              f"(n={sr[direction]['total']})")


if __name__ == "__main__":
    main()
