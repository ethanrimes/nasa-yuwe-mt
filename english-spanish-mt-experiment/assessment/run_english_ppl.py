"""E2: English-holdout perplexity per checkpoint.

The catastrophic-forgetting safeguard from the README: if fine-tuning on
EN<->ES erodes the base model's English ability, perplexity on a held-out
English-only set rises across snapshots. This is cheap (forward passes only,
no generation) so we run it on EVERY checkpoint plus the BASE model for an
anchor.

Writes assessment/data/english_ppl.json
"""
from __future__ import annotations

import argparse
import math
import time

from common import EVAL_DIR, OUT_DIR, TIERS, list_checkpoints, representative_checkpoints, load_jsonl, write_json


def model_ppl(ckpt_path, texts, n_threads, max_len=320):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(n_threads)
    tok = AutoTokenizer.from_pretrained(str(ckpt_path))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(ckpt_path), dtype=torch.float32)
    model.eval()

    total_nll = 0.0
    total_tok = 0
    with torch.inference_mode():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len)
            ids = enc["input_ids"]
            if ids.shape[1] < 2:
                continue
            out = model(ids, labels=ids)
            # HF returns mean NLL over (n-1) target tokens.
            ntok = ids.shape[1] - 1
            total_nll += float(out.loss) * ntok
            total_tok += ntok
    del model, tok
    import gc

    gc.collect()
    return math.exp(total_nll / total_tok), total_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="english_holdout lines")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--base", default="HuggingFaceTB/SmolLM2-360M",
                    help="base model id for the pre-finetune anchor (set '' to skip)")
    ap.add_argument("--all-ckpts", action="store_true")
    args = ap.parse_args()

    rows = load_jsonl(EVAL_DIR / "english_holdout.jsonl")[: args.n]
    texts = [r["text"] for r in rows]
    print(f"english_holdout: {len(texts)} lines", flush=True)

    results = {"n": len(texts), "tiers": {}, "base": None}

    if args.base:
        try:
            t0 = time.time()
            ppl, ntok = model_ppl(args.base, texts, args.threads)
            results["base"] = {"model": args.base, "ppl": ppl, "tokens": ntok}
            print(f"BASE {args.base}: ppl={ppl:.3f} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"BASE skipped: {e}", flush=True)

    for tier_dir, tier in TIERS.items():
        ckpts = list_checkpoints(tier_dir) if args.all_ckpts else representative_checkpoints(tier_dir, tier)
        series = []
        print(f"\n== {tier}: {len(ckpts)} ckpts ==", flush=True)
        for step, path in ckpts:
            t0 = time.time()
            ppl, ntok = model_ppl(path, texts, args.threads)
            series.append({"step": step, "ppl": ppl, "tokens": ntok})
            print(f"  step {step:>6}: ppl={ppl:.3f} ({time.time()-t0:.0f}s)", flush=True)
        results["tiers"][tier] = series
        write_json(results, OUT_DIR / "english_ppl.json")  # incremental save

    write_json(results, OUT_DIR / "english_ppl.json")
    print("\nDONE english_ppl.json", flush=True)


if __name__ == "__main__":
    main()
