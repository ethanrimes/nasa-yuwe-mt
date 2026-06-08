"""Fine-tune one NLLB-200 model on the Spanish<->Nasa-Yuwe parallel corpus.

Public entry point: :func:`build_and_train`. All heavy imports (torch, transformers,
datasets) are lazy so the module imports cleanly for dry-runs / planning / tests on
a machine without CUDA wheels.

Conventions mirror the ``ecmt`` trainer so downstream tooling is uniform:
  - outputs land under ``<output_root>/<run_id>/`` with ``checkpoints`` (HF
    ``output_dir``), a ``final/`` save, and a ``summary.json``;
  - bf16 + gradient checkpointing by default to fit the packed H100 waves;
  - ``save_steps`` checkpoints so the background Blob mirror loses at most one
    interval if teardown fires early.

Training is **loss-only** (no generation/BLEU during training) to keep the GPU
busy with the packed En<->Zh waves; metric eval is a separate post-hoc pass
(:func:`evaluate_bleu`).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import data as snmt_data
from .lang import (
    SPANISH_LANG,
    ensure_nasa_lang_token,
)


def _resolve_run_dir(output_root: str | Path, run_id: str) -> Path:
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_examples(split_rows: list[dict], tokenizer, *, max_source_len: int, max_target_len: int):
    """Tokenize bidirectional examples into a HF Dataset of input_ids/labels."""
    from datasets import Dataset

    from .lang import encode_example

    directed = list(snmt_data.iter_bidirectional_examples(split_rows))

    def _gen():
        for ex in directed:
            yield encode_example(
                tokenizer,
                src_text=ex["src_text"],
                tgt_text=ex["tgt_text"],
                src_lang=ex["src_lang"],
                tgt_lang=ex["tgt_lang"],
                max_source_len=max_source_len,
                max_target_len=max_target_len,
            )

    return Dataset.from_list(list(_gen())), len(directed)


def build_and_train(
    *,
    run_id: str,
    model_cfg: dict,
    training_cfg: dict,
    splits_dir: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Run one NLLB fine-tune. Returns a summary dict (and writes ``summary.json``)."""
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    # CRITICAL stability fix: NLLB/M2M100 fine-tunes diverge under bf16 AUTOCAST on
    # this corpus — the bf16 forward yields a non-finite loss within ~100 steps (the
    # guard below then has to skip every step, so the model never learns). The proven
    # root cause was bf16 master weights; loading fp32 masters fixed the *optimizer*
    # NaN, but the bf16 autocast forward is still unstable here. We therefore train in
    # full fp32 (configs set bf16:false). To keep fp32 fast on the H100 we enable TF32
    # tensor-core matmuls — near-bf16 throughput with the FULL fp32 exponent range and
    # a 10-bit mantissa (vs bf16's 7), which is numerically stable for this fine-tune.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    splits_dir = Path(splits_dir)
    run_dir = _resolve_run_dir(output_root, run_id)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    hf_id = model_cfg["hf_id"]
    t = training_cfg

    print(f"[snmt] run_id={run_id} model={hf_id} -> {run_dir}", flush=True)

    # -- tokenizer + model -------------------------------------------------- #
    # src_lang seeds the tokenizer's default; per-example framing is manual (lang.py).
    tokenizer = AutoTokenizer.from_pretrained(hf_id, src_lang=SPANISH_LANG)
    # Load in fp32 master weights. We deliberately do NOT pass a bf16 torch_dtype:
    # the earlier run loaded true-bf16 master weights AND set bf16=True, so the
    # optimizer state itself was bf16 and accumulated rounding error until AdamW
    # produced NaNs (~step 25). With fp32 master weights + bf16=True the Trainer
    # still autocasts the forward/backward to bf16 (the speed/VRAM win) while the
    # optimizer keeps a stable fp32 copy — the standard mixed-precision recipe.
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_id)
    # Register Nasa-Yuwe as a real language token, resize + warm-start embeddings.
    nasa_id = ensure_nasa_lang_token(tokenizer, model)
    print(f"[snmt] pbb_Latn token id={nasa_id}; vocab={len(tokenizer)}", flush=True)

    if bool(t.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # -- data --------------------------------------------------------------- #
    train_rows = snmt_data.load_split_rows(splits_dir / "train.parquet")
    dev_rows = snmt_data.load_split_rows(splits_dir / "dev.parquet")
    max_src = int(t.get("max_source_len", 200))
    max_tgt = int(t.get("max_target_len", 200))
    train_ds, n_train = _build_examples(train_rows, tokenizer, max_source_len=max_src, max_target_len=max_tgt)
    dev_ds, n_dev = _build_examples(dev_rows, tokenizer, max_source_len=max_src, max_target_len=max_tgt)
    print(f"[snmt] train examples (bidir)={n_train:,} dev={n_dev:,}", flush=True)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest", label_pad_token_id=-100)

    args = Seq2SeqTrainingArguments(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=int(t.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(t.get("per_device_eval_batch_size", 16)),
        gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 4)),
        num_train_epochs=float(t.get("num_train_epochs", 3)),
        learning_rate=float(t.get("learning_rate", 3e-5)),
        weight_decay=float(t.get("weight_decay", 0.0)),
        warmup_ratio=float(t.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(t.get("lr_scheduler_type", "cosine")),
        bf16=bool(t.get("bf16", False)),
        fp16=bool(t.get("fp16", False)),
        gradient_checkpointing=bool(t.get("gradient_checkpointing", True)),
        optim=str(t.get("optim", "adamw_torch")),
        max_grad_norm=float(t.get("max_grad_norm", 1.0)),
        seed=int(t.get("seed", 17)),
        logging_steps=int(t.get("logging_steps", 25)),
        eval_strategy="steps",
        eval_steps=int(t.get("eval_steps", 500)),
        save_strategy="steps",
        save_steps=int(t.get("save_steps", 500)),
        save_total_limit=t.get("save_total_limit", 2),
        report_to=["none"],
        run_name=run_id,
        predict_with_generate=False,  # loss-only during training; BLEU is post-hoc
        dataloader_num_workers=int(t.get("dataloader_num_workers", 2)),
    )

    # A non-finite training loss (overflow on a bad batch) would otherwise let
    # NaN/Inf grads reach the optimizer and poison every weight permanently. This
    # subclass detects that, scrubs the grads to zero, and returns a zero loss so the
    # step is effectively skipped (lr scheduler still advances) instead of diverging.
    class _GuardedSeq2SeqTrainer(Seq2SeqTrainer):
        def training_step(self, *args, **kwargs):  # type: ignore[override]
            loss = super().training_step(*args, **kwargs)
            if not torch.isfinite(loss):
                for p in self.model.parameters():
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                print("[snmt][guard] non-finite training loss; grads scrubbed, step skipped",
                      flush=True)
                return torch.zeros_like(loss)
            return loss

    trainer = _GuardedSeq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    t0 = time.time()
    trainer.train()
    final_dir = run_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    elapsed = time.time() - t0

    summary = {
        "run_id": run_id,
        "kind": "nllb",
        "hf_id": hf_id,
        "model_key": model_cfg.get("model_key"),
        "n_train_examples_bidir": n_train,
        "n_dev_examples_bidir": n_dev,
        "epochs": float(t.get("num_train_epochs", 3)),
        "elapsed_s": round(elapsed, 1),
        "run_dir": str(run_dir),
        "nasa_lang_token_id": int(nasa_id),
        "vocab_size": int(len(tokenizer)),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[snmt] run finished: {summary}", flush=True)
    return summary


def evaluate_bleu(
    *,
    model_dir: str | Path,
    splits_dir: str | Path,
    split: str = "test",
    max_new_tokens: int = 256,
    batch_size: int = 16,
    limit: int | None = None,
) -> dict[str, Any]:
    """Post-hoc generation eval: chrF + BLEU for both directions on ``split``.

    Separate from training so generation (slow) never competes with the packed
    training waves on the H100.
    """
    import torch
    from sacrebleu.metrics import BLEU, CHRF
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from .lang import DIRECTIONS, NLLB_EOS_ID, build_input_ids, lang_token_id

    splits_dir = Path(splits_dir)
    rows = snmt_data.load_split_rows(splits_dir / f"{split}.parquet")
    if limit:
        rows = rows[:limit]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, src_lang=SPANISH_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else NLLB_EOS_ID

    def _frame_source(text: str, src_lang: str) -> list[int]:
        # Mirror training's manual framing: [src_lang] + content + [eos]. Manual
        # (not tokenizer.src_lang) because pbb_Latn is an added token the
        # NllbTokenizer's src-lang machinery doesn't natively understand.
        content = tokenizer(text, add_special_tokens=False, truncation=True, max_length=254)["input_ids"]
        return build_input_ids(content, lang_token_id(tokenizer, src_lang), NLLB_EOS_ID)

    bleu, chrf = BLEU(), CHRF()
    results: dict[str, Any] = {}
    for d in DIRECTIONS:
        srcs = [snmt_data._norm(r.get(d.src_field, "")) for r in rows]
        refs = [snmt_data._norm(r.get(d.tgt_field, "")) for r in rows]
        hyps: list[str] = []
        forced = lang_token_id(tokenizer, d.tgt_lang)
        for i in range(0, len(srcs), batch_size):
            chunk = srcs[i : i + batch_size]
            framed = [_frame_source(s, d.src_lang) for s in chunk]
            width = max(len(ids) for ids in framed)
            # Left-pad so generation isn't seeded from pad tokens (NLLB is right-to-left padded for gen).
            input_ids = [[pad_id] * (width - len(ids)) + ids for ids in framed]
            attn = [[0] * (width - len(ids)) + [1] * len(ids) for ids in framed]
            enc = {
                "input_ids": torch.tensor(input_ids, device=device),
                "attention_mask": torch.tensor(attn, device=device),
            }
            with torch.no_grad():
                out = model.generate(**enc, forced_bos_token_id=forced, max_new_tokens=max_new_tokens)
            hyps.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        tag = f"{d.src_lang}->{d.tgt_lang}"
        results[tag] = {
            "bleu": round(bleu.corpus_score(hyps, [refs]).score, 2),
            "chrf": round(chrf.corpus_score(hyps, [refs]).score, 2),
            "n": len(srcs),
        }
    return results
