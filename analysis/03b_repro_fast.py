"""Lean root-cause repro for the NLLB es<->nasa divergence (forward-loss-first).

The decisive signal is the *forward* loss on a real batch:
  (A) bf16 master weights  (what training did)  -> expect huge / NaN loss
  (C) fp32 + bf16 autocast (the fix)            -> expect finite ~ln(V) loss

A correctly warm-started NLLB should start near cross-entropy ceiling
ln(vocab) ~= 12.5 or below. The actual runs started at mean loss 51 / 3413 /
2.07e8 (600m/1.3b/3.3b) with grad_norm=nan at the first logged step -> the
forward pass itself blew up under bf16 master weights.

CPU only. 4 examples, no gradient checkpointing, forward loss printed before
the (slow) backward so the key number lands fast.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "spanish-nasa-mt-experiment" / "src"))

import torch  # noqa: E402

from snmt.lang import SPANISH_LANG, ensure_nasa_lang_token  # noqa: E402

HF_ID = "facebook/nllb-200-distilled-600M"
NASA_JSONL = (
    REPO / "english-chinese-mt-experiment" / "data-en-zh" / "source"
    / "nasa_yuwe_parallel_dataset.jsonl"
)


def load_pairs(n=2):
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
        feats.append(
            encode_example(
                tok,
                src_text=ex["src_text"],
                tgt_text=ex["tgt_text"],
                src_lang=ex["src_lang"],
                tgt_lang=ex["tgt_lang"],
                max_source_len=128,
                max_target_len=128,
            )
        )
    collator = DataCollatorForSeq2Seq(tok, padding="longest", label_pad_token_id=-100)
    return collator(feats)


def run_case(label, dtype, rows, autocast=False, do_backward=True):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_ID, src_lang=SPANISH_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_ID, dtype=dtype)
    nasa_id = ensure_nasa_lang_token(tok, model)
    model.config.use_cache = False
    model.train()

    emb = model.get_input_embeddings().weight
    donor = emb[tok.convert_tokens_to_ids(SPANISH_LANG)]
    warmstart_match = bool(torch.equal(emb[nasa_id].float(), donor.float()))
    new_row_finite = bool(torch.isfinite(emb[nasa_id]).all())

    batch = build_batch(tok, rows)

    if autocast:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = model(**batch)
            loss = out.loss
    else:
        out = model(**batch)
        loss = out.loss

    loss_val = float(loss.detach().float())
    loss_finite = loss_val == loss_val and abs(loss_val) != float("inf")
    print(
        f"   [{label}] weight_dtype={str(dtype).replace('torch.','')} "
        f"autocast={autocast} FORWARD loss={loss_val:.4g} finite={loss_finite} "
        f"warmstart_eq_spanish={warmstart_match} new_row_finite={new_row_finite}",
        flush=True,
    )

    grad_norm = None
    n_nan = None
    if do_backward and loss_finite:
        loss.backward()
        g_sq, n_nan, n_params = 0.0, 0, 0
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
        print(
            f"   [{label}] BACKWARD grad_nan_tensors={n_nan}/{n_params} "
            f"global_grad_norm={grad_norm}",
            flush=True,
        )
    elif not loss_finite:
        print(f"   [{label}] forward loss already non-finite -> divergence at step 0", flush=True)

    rep = {
        "case": label,
        "weight_dtype": str(dtype).replace("torch.", ""),
        "autocast_bf16": autocast,
        "forward_loss": loss_val,
        "forward_loss_finite": bool(loss_finite),
        "warmstart_eq_spanish": warmstart_match,
        "new_token_row_finite": new_row_finite,
        "grad_nan_tensors": n_nan,
        "global_grad_norm": grad_norm,
    }
    del model
    return rep


def main():
    rows = load_pairs(2)
    print(f"[repro] {len(rows)} pairs -> {2*len(rows)} bidirectional examples", flush=True)
    cases = [
        ("A_bf16_master_weights", torch.bfloat16, False),
        ("C_fp32+bf16_autocast", torch.float32, True),
        ("B_fp32_master_weights", torch.float32, False),
    ]
    results = []
    for label, dtype, autocast in cases:
        print(f"[repro] case {label} ...", flush=True)
        try:
            rep = run_case(label, dtype, rows, autocast=autocast)
        except Exception as e:  # noqa: BLE001
            rep = {"case": label, "error": repr(e)}
            print("   ERROR", repr(e), flush=True)
        results.append(rep)

    out = REPO / "analysis" / "repro_divergence.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[repro] wrote", out, flush=True)


if __name__ == "__main__":
    main()
