"""Forward-only A/B/C contrast (fast): bf16 vs fp32 vs fp32+autocast.

No backward (a single step won't NaN; divergence is a ~25-step accumulation).
Just the forward loss to confirm warm-start + base forward are sane in all
precisions, isolating the cause to the bf16 *optimizer trajectory*.
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
                tok, src_text=ex["src_text"], tgt_text=ex["tgt_text"],
                src_lang=ex["src_lang"], tgt_lang=ex["tgt_lang"],
                max_source_len=128, max_target_len=128,
            )
        )
    collator = DataCollatorForSeq2Seq(tok, padding="longest", label_pad_token_id=-100)
    return collator(feats)


def fwd_case(label, dtype, rows, autocast=False):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_ID, src_lang=SPANISH_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_ID, dtype=dtype)
    nasa_id = ensure_nasa_lang_token(tok, model)
    model.config.use_cache = False
    model.eval()
    emb = model.get_input_embeddings().weight
    donor = emb[tok.convert_tokens_to_ids(SPANISH_LANG)]
    warmstart_match = bool(torch.equal(emb[nasa_id].float(), donor.float()))
    batch = build_batch(tok, rows)
    with torch.no_grad():
        if autocast:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                loss = float(model(**batch).loss.float())
        else:
            loss = float(model(**batch).loss.float())
    finite = loss == loss and abs(loss) != float("inf")
    rep = {
        "case": label, "weight_dtype": str(dtype).replace("torch.", ""),
        "autocast_bf16": autocast, "forward_loss": round(loss, 4),
        "forward_loss_finite": bool(finite), "warmstart_eq_spanish": warmstart_match,
    }
    print("   ", json.dumps(rep), flush=True)
    del model
    return rep


def main():
    rows = load_pairs(2)
    print(f"[fwd] {len(rows)} pairs -> {2*len(rows)} examples", flush=True)
    results = [
        fwd_case("A_bf16_master_weights", torch.bfloat16, rows, False),
        fwd_case("B_fp32_master_weights", torch.float32, rows, False),
        fwd_case("C_fp32+bf16_autocast", torch.float32, rows, True),
    ]
    out = REPO / "analysis" / "repro_forward.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[fwd] wrote", out, flush=True)


if __name__ == "__main__":
    main()
