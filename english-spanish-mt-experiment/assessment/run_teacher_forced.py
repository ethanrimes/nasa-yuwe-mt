"""E1a: teacher-forced token & literal recall across ALL snapshots (cheap).

Generation on CPU is too slow to run on every checkpoint. But the question
"has the model forgotten literals it once knew?" can be answered WITHOUT
generation: feed the model the correct source prompt + the *reference*
translation, and measure, at each target position, whether the model's top-1
prediction equals the reference token (teacher forcing). This is one forward
pass per sentence and directly measures literal recall:

  - token_recall  = top-1 target-token accuracy (sub-word level)
  - word_recall   = fraction of reference CONTENT words whose every sub-token
                    is predicted top-1 correctly ("would it emit this exact word")
  - target_nll    = cross-entropy on target tokens (lower = more confident/correct)

We persist, per snapshot, the per-content-word recall booleans so the
attribution experiment can detect forgetting (word recalled at an earlier
snapshot, lost at a later one) at the level of individual literals.

Runs on a FIXED, deterministic FLORES subset shared by all experiments.
"""
from __future__ import annotations

import argparse
import gc
import time

from common import (
    BEST_STEP,
    EVAL_DIR,
    OUT_DIR,
    TIERS,
    content_tokens,  # noqa: F401  (kept for parity / debugging)
    list_checkpoints,
    representative_checkpoints,
    load_jsonl,
    make_prompt,
    tokenize_words,
    write_json,
    _WORD_RE,
    _STOP_EN,
    _STOP_ES,
)


def pick_subset(n: int) -> list[dict]:
    rows = load_jsonl(EVAL_DIR / "flores200_devtest.jsonl")
    if n >= len(rows):
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def load_model(ckpt_path, n_threads):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(n_threads)
    tok = AutoTokenizer.from_pretrained(str(ckpt_path))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(ckpt_path), dtype=torch.float32)
    model.eval()
    return model, tok


def content_words_with_spans(full: str, prefix_len: int, lang: str):
    """Yield (word, start, end) for content words located in the target region."""
    stop = _STOP_ES if lang == "es" else _STOP_EN
    for m in _WORD_RE.finditer(full.lower()):
        if m.start() < prefix_len:
            continue
        w = m.group(0)
        if len(w) > 1 and w not in stop:
            yield w, m.start(), m.end()


def tf_sentence(model, tok, src_text, ref_text, direction, tgt_lang):
    """One forward pass; return per-sentence TF stats + per-word recall list."""
    import torch

    prefix = make_prompt(src_text, direction)
    full = prefix + ref_text
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"]
    offs = enc["offset_mapping"][0].tolist()
    if ids.shape[1] < 2:
        return None
    with torch.inference_mode():
        logits = model(ids).logits[0]
    logp = torch.log_softmax(logits, dim=-1)

    plen = len(prefix)
    # token-position correctness + nll for target tokens
    tok_correct = {}  # token index -> bool
    tgt_positions = []
    nll_sum = 0.0
    n_tgt = 0
    for i in range(1, ids.shape[1]):
        if offs[i][0] >= plen:  # target-region token
            tid = int(ids[0, i])
            pred = int(logits[i - 1].argmax())
            ok = pred == tid
            tok_correct[i] = ok
            tgt_positions.append(i)
            nll_sum += float(-logp[i - 1, tid])
            n_tgt += 1
    if n_tgt == 0:
        return None

    token_recall = sum(tok_correct.values()) / n_tgt
    target_nll = nll_sum / n_tgt

    # word-level: a content word is recalled if all its target tokens are top-1 correct
    word_list = []
    for w, ws, we in content_words_with_spans(full, plen, tgt_lang):
        span_tokens = [i for i in tgt_positions if not (offs[i][1] <= ws or offs[i][0] >= we)]
        if not span_tokens:
            recalled = False
        else:
            recalled = all(tok_correct.get(i, False) for i in span_tokens)
        word_list.append([w, bool(recalled)])
    if word_list:
        word_recall = sum(1 for _, r in word_list if r) / len(word_list)
    else:
        word_recall = 1.0
    return {
        "token_recall": token_recall,
        "target_nll": target_nll,
        "word_recall": word_recall,
        "words": word_list,
    }


def run_direction(model, tok, srcs, refs, direction, tgt_lang):
    per = []
    tr = wr = nll = 0.0
    nseen = 0
    for s, r in zip(srcs, refs):
        out = tf_sentence(model, tok, s, r, direction, tgt_lang)
        if out is None:
            per.append(None)
            continue
        per.append({"token_recall": out["token_recall"], "word_recall": out["word_recall"],
                    "target_nll": out["target_nll"], "words": out["words"]})
        tr += out["token_recall"]; wr += out["word_recall"]; nll += out["target_nll"]; nseen += 1
    agg = {
        "token_recall": tr / max(nseen, 1),
        "word_recall": wr / max(nseen, 1),
        "target_nll": nll / max(nseen, 1),
        "n": nseen,
    }
    return agg, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--tiers", default="")
    ap.add_argument("--all-ckpts", action="store_true",
                    help="use every snapshot instead of representative {first,best,last}")
    args = ap.parse_args()

    subset = pick_subset(args.n)
    en = [r["en"] for r in subset]
    es = [r["es"] for r in subset]
    write_json({"n": len(subset), "en": en, "es": es}, OUT_DIR / "eval_subset.json")

    want = set(t.strip() for t in args.tiers.split(",") if t.strip())
    tiers = [(d, s) for d, s in TIERS.items() if (not want or s in want)]

    g0 = time.time()
    for tier_dir, tier in tiers:
        ckpts = list_checkpoints(tier_dir) if args.all_ckpts else representative_checkpoints(tier_dir, tier)
        print(f"\n==== {tier} : {len(ckpts)} checkpoints (TF) ====", flush=True)
        tier_summary = []
        for step, path in ckpts:
            t0 = time.time()
            model, tok = load_model(path, args.threads)
            a_e2s, p_e2s = run_direction(model, tok, en, es, "en2es", "es")
            a_s2e, p_s2e = run_direction(model, tok, es, en, "es2en", "en")
            row = {
                "tier": tier, "step": step, "is_best": step == BEST_STEP.get(tier),
                "en2es": a_e2s, "es2en": a_s2e, "sec": round(time.time() - t0, 1),
            }
            tier_summary.append(row)
            write_json(
                {"tier": tier, "step": step, "en2es": p_e2s, "es2en": p_s2e},
                OUT_DIR / "tf" / f"{tier}_step{step}.json",
            )
            print(
                f"  step {step:>6}{' *BEST' if row['is_best'] else '      '}  "
                f"en2es tok {a_e2s['token_recall']:.3f} word {a_e2s['word_recall']:.3f} "
                f"nll {a_e2s['target_nll']:.3f} | "
                f"es2en tok {a_s2e['token_recall']:.3f} word {a_s2e['word_recall']:.3f} "
                f"nll {a_s2e['target_nll']:.3f}  ({row['sec']}s)",
                flush=True,
            )
            del model, tok
            gc.collect()
        write_json(tier_summary, OUT_DIR / f"tf_metrics_{tier}.json")
    print(f"\nTF DONE in {time.time()-g0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
