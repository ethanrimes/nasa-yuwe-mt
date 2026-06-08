"""E1b: real beam/greedy generation + BLEU/chrF/word-recall (selective).

Generation is the expensive part, so we run it only where it adds something
the cheap teacher-forced pass cannot:

  --selection best   : the best checkpoint of every tier, full subset
                       (cross-tier scaling, mistranslation set for attribution,
                        and the data for the metric-quality experiment)
  --selection trail  : EVERY checkpoint of the two widest-span runs (T10k, T1M)
                       so we can see whether *generation* quality (not just
                       teacher-forced recall) degrades after the best snapshot.

Outputs assessment/data/gen/<tier>_step<step>.json with full hypotheses, plus
assessment/data/gen_metrics_<selection>.json summaries.
"""
from __future__ import annotations

import argparse
import gc
import time

from common import (
    BEST_STEP,
    OUT_DIR,
    PROC_DIR,
    TIERS,
    list_checkpoints,
    representative_checkpoints,
    load_jsonl,
    make_prompt,
    word_recall,
    write_json,
)

TRAIL_TIERS = {"T10k": "T10000", "T1M": "T1000000"}


def subset_from_disk():
    import json

    p = OUT_DIR / "eval_subset.json"
    if not p.exists():
        raise SystemExit("Run run_teacher_forced.py first (creates eval_subset.json).")
    d = json.load(open(p, encoding="utf-8"))
    return d["en"], d["es"]


def load_model(ckpt_path, n_threads):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(n_threads)
    tok = AutoTokenizer.from_pretrained(str(ckpt_path))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(str(ckpt_path), dtype=torch.float32)
    model.eval()
    return model, tok


def generate(model, tok, texts, direction, beams, batch_size, max_new_tokens):
    import torch

    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        prompts = [make_prompt(t, direction) for t in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=320)
        with torch.inference_mode():
            ids = model.generate(
                **enc, max_new_tokens=max_new_tokens, num_beams=beams, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        plen = enc["input_ids"].shape[1]
        for row in ids:
            txt = tok.decode(row[plen:], skip_special_tokens=True)
            out.append(txt.split("\n", 1)[0].strip())
    return out


def score(hyps, refs, tgt_lang):
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    chrfpp = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    sent_bleu = [sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(hyps, refs)]
    sent_chrf = [sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(hyps, refs)]
    recs = [word_recall(h, r, tgt_lang)[0] for h, r in zip(hyps, refs)]
    return {
        "bleu": bleu, "chrf": chrf, "chrf_pp": chrfpp,
        "sent_bleu_mean": sum(sent_bleu) / len(sent_bleu),
        "word_recall": sum(recs) / len(recs), "n": len(hyps),
    }, sent_bleu, sent_chrf, recs


def selected_checkpoints(selection):
    out = []
    if selection in ("best", "indomain"):
        for tier_dir, tier in TIERS.items():
            best = BEST_STEP[tier]
            for step, path in list_checkpoints(tier_dir):
                if step == best:
                    out.append((tier, step, path))
    elif selection == "trail":
        for tier, tier_dir in TRAIL_TIERS.items():
            for step, path in representative_checkpoints(tier_dir, tier):
                out.append((tier, step, path))
    else:
        raise ValueError(selection)
    return out


def indomain_subset(tier_dir, n):
    """Deterministic in-distribution eval pairs from the tier's held-out val set."""
    rows = load_jsonl(PROC_DIR / tier_dir / "val.jsonl")
    if n and n < len(rows):
        step = len(rows) / n
        rows = [rows[int(i * step)] for i in range(n)]
    return [r["en"] for r in rows], [r["es"] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", choices=["best", "trail", "indomain"], required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = use full eval_subset")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    if args.selection != "indomain":
        en, es = subset_from_disk()
        if args.n and args.n < len(en):
            en, es = en[: args.n], es[: args.n]

    targets = selected_checkpoints(args.selection)
    n_show = "per-tier val" if args.selection == "indomain" else len(en)
    print(f"selection={args.selection}  ckpts={len(targets)}  n={n_show}  beams={args.beams}", flush=True)

    summary = []
    g0 = time.time()
    for tier, step, path in targets:
        if args.selection == "indomain":
            tier_dir = next(d for d, s in TIERS.items() if s == tier)
            en, es = indomain_subset(tier_dir, args.n or 60)
        t0 = time.time()
        model, tok = load_model(path, args.threads)
        rec = {"tier": tier, "step": step, "is_best": step == BEST_STEP.get(tier),
               "domain": "in" if args.selection == "indomain" else "out"}
        store = {"tier": tier, "step": step, "src_en": en, "src_es": es,
                 "domain": rec["domain"], "directions": {}}
        for direction, src, ref, tgt_lang in (("en2es", en, es, "es"), ("es2en", es, en, "en")):
            hyps = generate(model, tok, src, direction, args.beams, args.batch_size, args.max_new_tokens)
            m, sb, sc, recs = score(hyps, ref, tgt_lang)
            rec[direction] = m
            store["directions"][direction] = {"hyps": hyps, "ref": ref, "sent_bleu": sb,
                                               "sent_chrf": sc, "word_recall": recs}
        rec["sec"] = round(time.time() - t0, 1)
        summary.append(rec)
        tag = "indomain_" if args.selection == "indomain" else ""
        write_json(store, OUT_DIR / "gen" / f"{tag}{tier}_step{step}.json")
        print(
            f"  {tier} step {step:>6}{' *BEST' if rec['is_best'] else ''}  "
            f"en2es BLEU {rec['en2es']['bleu']:5.2f} chrF {rec['en2es']['chrf']:5.2f} | "
            f"es2en BLEU {rec['es2en']['bleu']:5.2f} chrF {rec['es2en']['chrf']:5.2f}  ({rec['sec']}s)",
            flush=True,
        )
        del model, tok
        gc.collect()
    write_json(summary, OUT_DIR / f"gen_metrics_{args.selection}.json")
    print(f"\nGEN[{args.selection}] DONE in {time.time()-g0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
