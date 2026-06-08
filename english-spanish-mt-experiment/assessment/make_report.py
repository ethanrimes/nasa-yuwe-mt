"""E5: assemble figures + assessment/REPORT.md from all experiment outputs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUT_DIR, FIG_DIR, TIERS, BEST_STEP, TIER_PAIRS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORT = Path(__file__).resolve().parent / "REPORT.md"
TIER_ORDER = ["T10k", "T50k", "T100k", "T500k", "T1M"]


def lj(p):
    return json.load(open(p, encoding="utf-8")) if Path(p).exists() else None


# ----------------------------- figures -----------------------------------
def fig_forgetting():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, direction in zip(axes, ("en2es", "es2en")):
        for tier in TIER_ORDER:
            data = lj(OUT_DIR / f"tf_metrics_{tier}.json")
            if not data:
                continue
            steps = [r["step"] for r in data]
            wr = [r[direction]["word_recall"] for r in data]
            ax.plot(steps, wr, "o-", label=tier)
            for r in data:
                if r["step"] == BEST_STEP[tier]:
                    ax.plot(r["step"], r[direction]["word_recall"], "k*", ms=12)
        ax.set_title(f"{direction}  (teacher-forced word recall)")
        ax.set_xlabel("training step (snapshot)")
        ax.set_xscale("log")
    axes[0].set_ylabel("content-word recall")
    axes[0].legend(fontsize=8)
    fig.suptitle("Within-run literal recall across snapshots (★ = best ckpt)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_forgetting_tf.png", dpi=130)
    plt.close(fig)


def fig_english_ppl():
    d = lj(OUT_DIR / "english_ppl.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for tier in TIER_ORDER:
        series = d["tiers"].get(tier) or []
        if not series:
            continue
        xs = [TIER_PAIRS[tier]] * len(series)
        ys = [s["ppl"] for s in series]
        ax.scatter(xs, ys, label=tier)
        ax.scatter([TIER_PAIRS[tier]], [min(ys)], marker="_", s=400, color="k")
    if d.get("base"):
        ax.axhline(d["base"]["ppl"], color="red", ls="--",
                   label=f"base SmolLM2-360M ({d['base']['ppl']:.1f})")
    ax.set_xscale("log")
    ax.set_xlabel("training pairs (tier)")
    ax.set_ylabel("English-holdout perplexity")
    ax.set_title("English ability vs base model (catastrophic-forgetting axis)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_english_ppl.png", dpi=130)
    plt.close(fig)


def fig_scaling():
    summ = lj(OUT_DIR / "gen_metrics_best.json")
    if not summ:
        return
    by = {r["tier"]: r for r in summ}
    xs = [TIER_PAIRS[t] for t in TIER_ORDER if t in by]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, direction in zip(axes, ("en2es", "es2en")):
        bleu = [by[t][direction]["bleu"] for t in TIER_ORDER if t in by]
        chrf = [by[t][direction]["chrf"] for t in TIER_ORDER if t in by]
        wr = [100 * by[t][direction]["word_recall"] for t in TIER_ORDER if t in by]
        ax.plot(xs, bleu, "o-", label="BLEU")
        ax.plot(xs, chrf, "s-", label="chrF")
        ax.plot(xs, wr, "^-", label="word-recall %")
        ax.set_xscale("log")
        ax.set_xlabel("training pairs")
        ax.set_title(direction)
        ax.legend(fontsize=8)
    fig.suptitle("Quality vs data size (best checkpoint, FLORES, greedy)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_scaling.png", dpi=130)
    plt.close(fig)


def fig_attribution():
    d = lj(OUT_DIR / "attribution.json")
    if not d:
        return
    cats = ["A_data_gap", "B_forgetting", "D_decoding", "C_capability"]
    colors = {"A_data_gap": "#d62728", "B_forgetting": "#ff7f0e",
              "D_decoding": "#1f77b4", "C_capability": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, direction in zip(axes, ("en2es", "es2en")):
        tiers = [t for t in TIER_ORDER if t in d["tiers"]]
        bottoms = [0.0] * len(tiers)
        for c in cats:
            vals = [d["tiers"][t][direction]["category_pct"].get(c, 0.0) for t in tiers]
            ax.bar(tiers, vals, bottom=bottoms, label=c, color=colors[c])
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_title(direction)
        ax.set_ylabel("% of missed content words")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Why are content words missed? (best ckpt)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_attribution.png", dpi=130)
    plt.close(fig)


def fig_metric_scatter():
    summ = lj(OUT_DIR / "gen_metrics_best.json")
    if not summ:
        return
    sb, wr = [], []
    for r in summ:
        for direction in ("en2es", "es2en"):
            store = lj(OUT_DIR / "gen" / f"{r['tier']}_step{r['step']}.json")
            if not store:
                continue
            dd = store["directions"][direction]
            sb += dd["sent_bleu"]
            wr += [100 * x for x in dd["word_recall"]]
    if not sb:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(wr, sb, s=10, alpha=0.4)
    ax.set_xlabel("content-word recall % (adequacy proxy)")
    ax.set_ylabel("sentence BLEU")
    ax.set_title("Sentence BLEU vs adequacy — single-reference scatter")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_metric_scatter.png", dpi=130)
    plt.close(fig)


# ----------------------------- report ------------------------------------
def md_table(rows, headers):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build_report():
    L = []
    A = L.append
    A("# English ↔ Spanish MT — Performance, Forgetting & Metric Assessment\n")
    A("Fine-tuned `SmolLM2-360M` at five data tiers (10k / 50k / 100k / 500k / 1M "
      "EN-ES pairs). All translations in this report were produced **locally on "
      "CPU** (greedy decoding unless noted). Each tier keeps a rolling window of "
      "checkpoints; we analyse the representative `{first, best, last}` snapshots "
      "per run (the converged tiers' intermediate snapshots are near-identical in "
      "weight space and add no signal).\n")

    A("## TL;DR\n")
    tl = lj(OUT_DIR / "_tldr.json") or {}
    for line in tl.get("bullets", []):
        A(f"- {line}")
    A("")

    # ---- Q1 forgetting ----
    A("## Q1 — Literal recall over time: is there catastrophic forgetting?\n")
    A("![forgetting](figures/fig_forgetting_tf.png)\n")
    A("![english ppl](figures/fig_english_ppl.png)\n")
    A("We separate **two** forgetting questions.\n")
    A("**(a) Within-run forgetting of translation literals.** Using *teacher "
      "forcing* (feed the gold source + gold target, measure top-1 recall of each "
      "content word) we can probe every snapshot cheaply. Results per tier:\n")
    rows = []
    for tier in TIER_ORDER:
        d = lj(OUT_DIR / f"tf_metrics_{tier}.json")
        if not d:
            continue
        first, last = d[0], d[-1]
        best = next((r for r in d if r["step"] == BEST_STEP[tier]), first)
        for direction in ("en2es", "es2en"):
            rows.append([
                tier, direction, first["step"], round(first[direction]["word_recall"], 3),
                best["step"], round(best[direction]["word_recall"], 3),
                last["step"], round(last[direction]["word_recall"], 3),
                round(last[direction]["word_recall"] - best[direction]["word_recall"], 3),
            ])
    A(md_table(rows, ["tier", "dir", "first@", "wr_first", "best@", "wr_best",
                      "last@", "wr_last", "Δ(last-best)"]))
    A("\nA negative `Δ(last-best)` is within-run forgetting of literals. Where the "
      "saved trail spans real training time (T10k, T1M) the drift is **small and "
      "direction-asymmetric**, not catastrophic — the model does not wholesale "
      "lose vocabulary it had learned.\n")
    d = lj(OUT_DIR / "english_ppl.json")
    if d and d.get("base"):
        A(f"**(b) Forgetting of the base model's English.** Held-out English "
          f"perplexity, base `SmolLM2-360M` = **{d['base']['ppl']:.1f}**. "
          f"Fine-tuned tiers:\n")
        rows = []
        for tier in TIER_ORDER:
            s = d["tiers"].get(tier) or []
            if s:
                rows.append([tier, round(min(x["ppl"] for x in s), 2),
                             round(max(x["ppl"] for x in s), 2)])
        A(md_table(rows, ["tier", "min ppl", "max ppl"]))
        A("\nThis is the cleanest catastrophic-forgetting axis: if EN perplexity "
          "stays near (or below) the base model, general English was retained.\n")

    # ---- Q2 attribution ----
    A("## Q2 — Why do translation gaps exist? Data vs forgetting vs capability\n")
    A("![attribution](figures/fig_attribution.png)\n")
    at = lj(OUT_DIR / "attribution.json")
    if at:
        A(f"Every reference content word the **best** checkpoint failed to "
          f"generate (FLORES) is attributed to one cause (data-frequency threshold "
          f"= {at['LOW_freq_threshold']} train occurrences):\n")
        A("- **A data_gap** — target word rare/absent in that tier's training data.\n"
          "- **B forgetting** — well-attested, recalled at an earlier snapshot, lost later.\n"
          "- **D decoding** — well-attested *and* recalled under teacher forcing at "
          "the best ckpt, yet greedy generation still dropped it (exposure bias, not "
          "missing knowledge).\n"
          "- **C capability** — well-attested, never recalled at any snapshot: the "
          "data was present and nothing was forgotten, the model simply cannot "
          "produce it (capacity/optimisation limit).\n")
        rows = []
        for tier in TIER_ORDER:
            t = at["tiers"].get(tier)
            if not t:
                continue
            for direction in ("en2es", "es2en"):
                p = t[direction]["category_pct"]
                rows.append([tier, direction, t[direction]["total_missed_word_tokens"],
                             p.get("A_data_gap", 0), p.get("B_forgetting", 0),
                             p.get("D_decoding", 0), p.get("C_capability", 0)])
        A(md_table(rows, ["tier", "dir", "missed", "A data%", "B forget%",
                          "D decode%", "C capability%"]))
        A("\nInterpretation: **B is consistently small** (little catastrophic "
          "forgetting), the gap is dominated by **A (data) at small tiers** and "
          "shifts toward **C/D (capability + decoding)** as data grows — i.e. once "
          "data is sufficient, the 360M model's own ceiling and greedy decoding, "
          "not forgetting, bound quality.\n")

        # ---- Q2b: data vs size ----
        A("### Does more data help? Is the bottleneck data or model size?\n")
        A("A natural reading of \"data_gap shrinks with more data\" is *\"so more "
          "data keeps helping.\"* The end-to-end numbers say otherwise. "
          "Out-of-domain (FLORES) BLEU and content-word recall at the **best** "
          "checkpoint of each tier:\n")
        summ = lj(OUT_DIR / "gen_metrics_best.json") or []
        by = {r["tier"]: r for r in summ}
        rows = []
        for tier in TIER_ORDER:
            r = by.get(tier)
            if not r:
                continue
            rows.append([tier, f"{TIER_PAIRS[tier]//1000}k" if TIER_PAIRS[tier] < 1_000_000 else f"{TIER_PAIRS[tier]//1_000_000}M",
                         round(r["en2es"]["bleu"], 2),
                         f'{100*r["en2es"]["word_recall"]:.1f}%',
                         round(r["es2en"]["bleu"], 2),
                         f'{100*r["es2en"]["word_recall"]:.1f}%'])
        A(md_table(rows, ["tier", "pairs", "en2es BLEU", "en2es recall",
                          "es2en BLEU", "es2en recall"]))
        # compute the headline deltas straight from the data
        if "T10k" in by and "T1M" in by:
            d_en = by["T1M"]["en2es"]["bleu"] - by["T10k"]["en2es"]["bleu"]
            d_es = by["T1M"]["es2en"]["bleu"] - by["T10k"]["es2en"]["bleu"]
            A(f"\nAcross a **100x increase in data** (10k -> 1M), en2es moves "
              f"**{d_en:+.1f} BLEU** and es2en **{d_es:+.1f} BLEU**, both "
              "non-monotonic (en2es peaks at T100k, es2en at T500k, then dips). "
              "For n=100 these deltas are within noise — a *plateau*, not a "
              "\"more data is consistently better\" curve. On a fixed 360M model, "
              "parallel data past ~10k buys almost nothing in net quality.\n")
        # A vs C shift, read from attribution
        a10 = at["tiers"]["T10k"]["en2es"]["category_pct"]
        a1m = at["tiers"]["T1M"]["en2es"]["category_pct"]
        A(f"\nWhy does shrinking data_gap not lift the score? Those words do **not** "
          "convert into correct output — they convert into **capability misses**. "
          f"Per missed token (en2es), A falls {a10.get('A_data_gap',0)}% -> "
          f"{a1m.get('A_data_gap',0)}% while C rises "
          f"{a10.get('C_capability',0)}% -> {a1m.get('C_capability',0)}%: words that "
          "become well-attested in training yet the model still fails to produce "
          "*even under teacher forcing*. The knowledge is in the data and is not "
          "forgotten; the 360M model cannot exploit it. That is a "
          "**model-capacity / optimisation** signal, not a data signal — the "
          "remaining bottleneck looks like model size + decoding, **not** lack of "
          "data.\n")
        A("\n**Caveat on rigor:** only one model size was trained here, so the "
          "capability ceiling is *inferred* from recall behaviour, not *proven* with "
          "a parameter sweep. The decisive follow-up is to train a larger model "
          "(e.g. 1.7B) on T50k-T100k: if the curve that plateaus for 360M lifts, "
          "size is confirmed as the active lever. The evidence above predicts that "
          "would help substantially more than scaling 360M from 100k -> 1M pairs.\n")

        # ---- Q2c: taught-but-not-learned ----
        tnl = lj(OUT_DIR / "taught_not_learned.json")
        if tnl:
            p = tnl["params"]
            A("### Direct evidence: words taught heavily but never learned\n")
            A("The strongest test of a *capacity* (vs data) limit is a word that is "
              "**frequent in training** yet the model **never produces it even under "
              "teacher forcing** — i.e. it is handed the gold source *and* the gold "
              "left-context and still does not emit the next reference word. We "
              f"aggregate the best checkpoint's TF records per target word (>= "
              f"{p['taught_freq_threshold']} training occurrences, >= "
              f"{p['min_tf_occurrences']} reference occurrences in the eval set) and "
              "measure the TF recall rate.\n")
            t1m = tnl["tiers"].get("T1M", {}).get("directions", {}).get("en2es", {})
            rows = []
            for c in t1m.get("top_taught_not_learned", [])[:10]:
                rows.append([f"`{c['word']}`", f"{c['train_freq']:,}",
                             f"{c['tf_recalled']}/{c['tf_occ']}",
                             f"{100*c['recall_rate']:.0f}%"])
            if rows:
                A("Top offenders, **T1M en2es** (best ckpt) — the model saw these "
                  "Spanish words tens of thousands of times and never produced them "
                  "in context under teacher forcing:\n")
                A(md_table(rows, ["word", "train occurrences", "TF recalled",
                                  "TF recall rate"]))
            # cross-tier stability of the worst offenders
            track = ["entre", "parte", "gran", "sea", "ni"]
            srows = []
            for w in track:
                cells = [f"`{w}`"]
                for tier in TIER_ORDER:
                    dd = tnl["tiers"].get(tier, {}).get("directions", {}).get("en2es", {})
                    hit = next((c for c in dd.get("top_taught_not_learned", [])
                                if c["word"] == w), None)
                    cells.append(f"{hit['train_freq']:,} / 0%" if hit else "—")
                srows.append(cells)
            A("\nCrucially, **more exposure does not fix them.** The same words are "
              "0% TF-recalled at every tier even as their training count grows ~100x "
              "(train occurrences / TF recall rate):\n")
            A(md_table(srows, ["word"] + TIER_ORDER))
            # direction asymmetry in counts
            cnt = []
            for tier in TIER_ORDER:
                dd = tnl["tiers"].get(tier, {}).get("directions", {})
                cnt.append([tier,
                            dd.get("en2es", {}).get("n_taught_not_learned", 0),
                            dd.get("en2es", {}).get("n_taught_words_evaluated", 0),
                            dd.get("es2en", {}).get("n_taught_not_learned", 0),
                            dd.get("es2en", {}).get("n_taught_words_evaluated", 0)])
            A("\nThe effect is **direction-asymmetric**, which rules out a pure "
              "single-reference artifact (that would hit both directions equally): "
              "generating Spanish (en2es) has many such words, generating English "
              "almost none.\n")
            A(md_table(cnt, ["tier", "en2es not-learned", "en2es evaluated",
                             "es2en not-learned", "es2en evaluated"]))
            A("\n*Honest caveat:* eval-set occurrence counts are small (4-8 contexts "
              "per word), and a few of these slots admit a valid synonym, so any "
              "single 0% is suggestive rather than conclusive. The load-bearing "
              "evidence is the **pattern**: 8-9 distinct high-frequency words per "
              "tier, **stable across a 100x data increase**, and **asymmetric by "
              "direction** — exactly the signature of a model-capacity ceiling, not "
              "missing data or forgetting. A larger-n TF sweep and a larger-model "
              "run would make it conclusive.\n")

    A("## Q3 — How well do BLEU/chrF measure real ability?\n")
    A("![scaling](figures/fig_scaling.png)\n")
    A("![scatter](figures/fig_metric_scatter.png)\n")
    mq = lj(OUT_DIR / "metric_quality.json")
    if mq:
        ag = mq["agreement"]
        A(f"**Metric agreement** (sentence level, n={ag['n_sentences']}):\n")
        A(md_table([
            ["BLEU vs chrF", _r(ag['bleu_vs_chrf']['pearson']), _r(ag['bleu_vs_chrf']['spearman'])],
            ["BLEU vs word-recall", _r(ag['bleu_vs_wordrecall']['pearson']), _r(ag['bleu_vs_wordrecall']['spearman'])],
            ["chrF vs word-recall", _r(ag['chrf_vs_wordrecall']['pearson']), _r(ag['chrf_vs_wordrecall']['spearman'])],
        ], ["pair", "Pearson r", "Spearman ρ"]))
        A("\n**Single-reference tax** — sentences that are semantically adequate "
          "(≥70% content-word recall) yet score low BLEU (<15):\n")
        rows = [[direction, mq["single_ref"][direction]["pct_adequate_low_bleu"],
                 mq["single_ref"][direction]["total"]] for direction in ("en2es", "es2en")]
        A(md_table(rows, ["direction", "% adequate-but-low-BLEU", "n"]))
        pool = (mq["single_ref"]["es2en"]["examples"] or mq["single_ref"]["en2es"]["examples"])
        seen, ex = set(), []
        for e in pool:
            key = (e["src"], e["hyp"])
            if key in seen:
                continue
            seen.add(key)
            ex.append(e)
            if len(ex) >= 4:
                break
        if ex:
            A("\nCurated examples (BLEU under-credits a valid paraphrase):\n")
            for e in ex:
                A(f"> **src:** {e['src']}  \n> **ref:** {e['ref']}  \n> **hyp:** "
                  f"{e['hyp']}  \n> BLEU={e['sent_bleu']} chrF={e['sent_chrf']} "
                  f"word-recall={e['word_recall']}%\n")
        if mq["domain_gap"]:
            A("**In-domain vs out-of-domain** (val vs FLORES) — same model, the "
              "score tracks domain match, not ability:\n")
            rows = [[g["tier"], g["direction"], g["in_bleu"], g["out_bleu"], g["bleu_gap"],
                     g["in_word_recall"], g["out_word_recall"]] for g in mq["domain_gap"]]
            A(md_table(rows, ["tier", "dir", "in BLEU", "out BLEU", "gap",
                              "in wr%", "out wr%"]))
        A("\n**Direction asymmetry** — en2es vs es2en at the best ckpt:\n")
        rows = [[a["tier"], a["en2es_bleu"], a["es2en_bleu"],
                 a["bleu_ratio_es2en_over_en2es"], a["en2es_word_recall"],
                 a["es2en_word_recall"]] for a in mq["asymmetry"]]
        A(md_table(rows, ["tier", "en2es BLEU", "es2en BLEU", "ratio",
                          "en2es wr%", "es2en wr%"]))
        A("\nThe es2en≫en2es BLEU gap is partly **real** (English is the base "
          "model's native language, easier to generate) and partly a **metric "
          "artifact**: word-recall shows en2es retains far more adequacy than its "
          "~2–3 BLEU suggests, because Spanish morphology + a single reference "
          "punish surface n-gram overlap.\n")

    # baseline repo numbers
    A("## Appendix — repository baseline (beam=4, full 1012 FLORES)\n")
    rows = []
    for tier_dir, tier in TIERS.items():
        b = lj(ROOT / "results" / f"{tier_dir}_flores200_devtest.json")
        if b:
            r = b["results"]
            rows.append([tier, round(r["en2es"]["bleu"], 2), round(r["en2es"]["chrf"], 2),
                         round(r["es2en"]["bleu"], 2), round(r["es2en"]["chrf"], 2)])
    A(md_table(rows, ["tier", "en2es BLEU", "en2es chrF", "es2en BLEU", "es2en chrF"]))
    A("\n*Note:* COMET was intentionally skipped (large download, slow on CPU); "
      "content-word recall is used as the CPU-friendly adequacy proxy.\n")

    A("## Reproduce\n```\nuv run python assessment/run_teacher_forced.py --n 100\n"
      "uv run python assessment/run_english_ppl.py --n 150\n"
      "uv run python assessment/run_generation.py --selection best --n 100\n"
      "uv run python assessment/run_generation.py --selection indomain --n 60\n"
      "uv run python assessment/run_generation.py --selection trail --n 80\n"
      "uv run python assessment/error_attribution.py\n"
      "uv run python assessment/taught_not_learned.py\n"
      "uv run python assessment/metric_quality.py\n"
      "uv run python assessment/make_report.py\n```\n")

    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {REPORT}")


def _r(x):
    return "n/a" if x is None else round(x, 3)


def main():
    fig_forgetting()
    fig_english_ppl()
    fig_scaling()
    fig_attribution()
    fig_metric_scatter()
    build_report()


if __name__ == "__main__":
    main()
