"""Decisive root-cause repro for the NLLB es<->nasa divergence.

Hypothesis: the runs loaded the model with ``torch_dtype=torch.bfloat16`` (true
bf16 *master weights*, train.py:95) and then trained with adamw_torch. For
NLLB-200 this is numerically fragile: the first optimizer step(s) blow up, grads
go NaN (grad_norm=nan from step 25 in every run), AdamW poisons all weights, loss
flatlines at 0.0 and eval_loss=NaN.

This script loads the *base* nllb-200-distilled-600M and runs ONE forward+backward
on a real es<->nasa batch under:
  (A) bf16 master weights      (what training did)  -> expect NaN / huge loss
  (B) fp32 master weights      (the fix)            -> expect finite loss & grads
  (C) fp32 weights + bf16 autocast (best fix)       -> expect finite

It also checks the resize/warm-start leaves finite embeddings. CPU only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "spanish-nasa-mt-experiment" / "src"))

import torch  # noqa: E402

from snmt.lang import (  # noqa: E402
    NASA_LANG,
    SPANISH_LANG,
    ensure_nasa_lang_token,
)

HF_ID = "facebook/nllb-200-distilled-600M"
NASA_JSONL = REPO / "english-chinese-mt-experiment" / "data-en-zh" / "source" / "nasa_yuwe_parallel_dataset.jsonl"


def load_pairs(n=8):
    rows = []
    with NASA_JSONL.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("spanish") and r.get("nasa_yuwe"):
                rows.append(r)
            if len(rows) >= n:
                break
    return rows


def build_batch(tok, rows):
    from snmt.data import iter_bidirectional_examples
    from snmt.lang import encode_example
    from transformers import DataCollatorForSeq2Seq

    feats = []
    for ex in iter_bidirectional_examples(rows):
        feats.append(encode_example(
            tok, src_text=ex["src_text"], tgt_text=ex["tgt_text"],
            src_lang=ex["src_lang"], tgt_lang=ex["tgt_lang"],
            max_source_len=200, max_target_len=200,
        ))
    collator = DataCollatorForSeq2Seq(tok, padding="longest", label_pad_token_id=-100)
    return collator(feats)


def finite_report(model):
    emb = model.get_input_embeddings().weight
    nid = model.config.vocab_size  # not reliable; recompute below
    return {
        "emb_finite": bool(torch.isfinite(emb).all()),
        "emb_absmax": float(emb.abs().max()),
    }


def run_case(label, dtype, rows, tok_ref, autocast=False):
    from transformers import AutoModelForSeq2SeqLM

    model = AutoModelForSeq2SeqLM.from_pretrained(HF_ID, torch_dtype=dtype)
    nasa_id = ensure_nasa_lang_token(tok_ref, model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()

    emb = model.get_input_embeddings().weight
    new_row_finite = bool(torch.isfinite(emb[nasa_id]).all())
    new_row_absmax = float(emb[nasa_id].abs().max())
    donor_row = emb[tok_ref.convert_tokens_to_ids(SPANISH_LANG)]
    warmstart_match = bool(torch.equal(emb[nasa_id].float(), donor_row.float()))

    batch = build_batch(tok_ref, rows)
    batch = {k: v for k, v in batch.items()}

    if autocast:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = model(**batch)
            loss = out.loss
    else:
        out = model(**batch)
        loss = out.loss

    loss_val = float(loss.detach().float())
    loss.backward()

    g_sq = 0.0
    n_nan = 0
    n_params = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        n_params += 1
        g = p.grad.detach().float()
        if not torch.isfinite(g).all():
            n_nan += 1
        else:
            g_sq += float((g * g).sum())
    grad_norm = g_sq ** 0.5 if n_nan == 0 else float("nan")

    rep = {
        "case": label,
        "weight_dtype": str(dtype).replace("torch.", ""),
        "autocast_bf16": autocast,
        "loss": loss_val,
        "loss_finite": bool(loss_val == loss_val and abs(loss_val) != float("inf")),
        "new_token_row_finite": new_row_finite,
        "new_token_row_absmax": round(new_row_absmax, 4),
        "warmstart_eq_spanish": warmstart_match,
        "grad_tensors": n_params,
        "grad_nan_or_inf_tensors": n_nan,
        "global_grad_norm": grad_norm,
    }
    del model
    return rep


def main():
    from transformers import AutoTokenizer

    rows = load_pairs(8)
    print(f"[repro] {len(rows)} real es<->nasa pairs -> {2*len(rows)} bidirectional examples", flush=True)

    results = []
    # Fresh tokenizer per case (ensure_nasa_lang_token mutates it; idempotent but keep clean).
    for label, dtype, autocast in [
        ("A_bf16_master_weights (as trained)", torch.bfloat16, False),
        ("B_fp32_master_weights (fix)", torch.float32, False),
        ("C_fp32_weights+bf16_autocast (best fix)", torch.float32, True),
    ]:
        tok = AutoTokenizer.from_pretrained(HF_ID, src_lang=SPANISH_LANG)
        print(f"[repro] running case {label} ...", flush=True)
        try:
            rep = run_case(label, dtype, rows, tok, autocast=autocast)
        except Exception as e:  # noqa: BLE001
            rep = {"case": label, "error": repr(e)}
        results.append(rep)
        print("   ", json.dumps(rep, ensure_ascii=False), flush=True)

    out = REPO / "analysis" / "repro_divergence.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[repro] wrote", out, flush=True)


if __name__ == "__main__":
    main()
