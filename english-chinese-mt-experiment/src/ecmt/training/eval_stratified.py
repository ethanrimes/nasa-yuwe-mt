"""Stratified evaluation -> "where is the model deficient?" + checkpoint-over-time curves.

The user's framing: since the proxy data all fits in the sentence budget, train EN<->ZH
on *everything*, then diagnose where the model still fails and watch capabilities emerge
across checkpoints. This module provides the deficiency-diagnosis layer on top of eval.py:

  * per-segment scoring (sentence BLEU + chrF++) so we can GROUP, not just average;
  * stratification by **topic/domain** (free from FLORES metadata), **length bucket**,
    **direction** (en2zh / zh2en), and an optional **grammatical-feature** tag (from the
    grammar_coverage_spec held-out partition);
  * `find_deficiencies()` -> the strata that lag the overall mean (what to fix / what to
    over-sample in the real Nasa commission);
  * `evaluate_checkpoints_over_time()` -> one stratified report per retained checkpoint,
    yielding learning curves per capability.

The aggregation functions are pure (no model / no GPU) so they unit-test on CPU. The
model-generation path reuses eval.generate_translations().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Pure aggregation (CPU-testable, no torch / no model)
# ---------------------------------------------------------------------------

LENGTH_BUCKETS = (("short", 0, 10), ("medium", 10, 25), ("long", 25, 10**9))


def length_bucket(text: str, *, by: str = "words") -> str:
    n = len(text.split()) if by == "words" else len(text)
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= n < hi:
            return name
    return "long"


def segment_scores(refs: list[str], hyps: list[str], *, tgt_lang: str) -> list[dict[str, float]]:
    """Per-sentence BLEU + chrF++ (so results can be grouped by stratum)."""
    import sacrebleu

    bleu_tok = "zh" if tgt_lang == "zh" else "13a"
    out: list[dict[str, float]] = []
    for ref, hyp in zip(refs, hyps):
        b = sacrebleu.sentence_bleu(hyp, [ref], tokenize=bleu_tok).score
        c = sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score
        out.append({"bleu": b, "chrf": c})
    return out


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def stratify(
    rows: list[dict[str, Any]],
    seg: list[dict[str, float]],
    *,
    direction: str,
    keys: tuple[str, ...] = ("topic", "length", "feature"),
    tgt_lang: str,
) -> dict[str, Any]:
    """Group per-segment scores by each stratum key. `rows` carry metadata
    (topic/domain/feature); `length` is derived from the *target* reference.

    Returns {overall: {...}, by_key: {key: {value: {n,bleu,chrf}}}} for one direction.
    """
    assert len(rows) == len(seg), (len(rows), len(seg))
    tgt_field = "zh" if tgt_lang == "zh" else "en"

    def stratum_value(r: dict[str, Any], key: str) -> str:
        if key == "length":
            return length_bucket(r.get(tgt_field, ""))
        return str(r.get(key, "unknown") or "unknown")

    by_key: dict[str, dict[str, dict[str, float]]] = {}
    for key in keys:
        groups: dict[str, list[dict[str, float]]] = {}
        for r, s in zip(rows, seg):
            groups.setdefault(stratum_value(r, key), []).append(s)
        by_key[key] = {
            val: {"n": len(ss), "bleu": _mean(s["bleu"] for s in ss), "chrf": _mean(s["chrf"] for s in ss)}
            for val, ss in sorted(groups.items())
        }
    return {
        "direction": direction,
        "overall": {"n": len(seg), "bleu": _mean(s["bleu"] for s in seg), "chrf": _mean(s["chrf"] for s in seg)},
        "by": by_key,
    }


def find_deficiencies(report: dict[str, Any], *, metric: str = "chrf", rel_margin: float = 0.85,
                      min_n: int = 15) -> list[dict[str, Any]]:
    """Flag strata whose `metric` is < rel_margin * overall (i.e. clearly lagging).

    These are the capabilities the model is *weakest* at -> what to fix / over-sample in
    the real Nasa commission. Only strata with >= min_n segments are considered.
    """
    overall = report["overall"][metric]
    floor = rel_margin * overall
    out: list[dict[str, Any]] = []
    for key, groups in report["by"].items():
        for val, stats in groups.items():
            if stats["n"] >= min_n and stats[metric] < floor:
                out.append({
                    "direction": report["direction"], "key": key, "value": val,
                    "n": stats["n"], metric: stats[metric],
                    "gap_vs_overall": round(stats[metric] - overall, 3),
                })
    return sorted(out, key=lambda d: d["gap_vs_overall"])


# ---------------------------------------------------------------------------
# Model-facing wrappers (reuse eval.generate_translations)
# ---------------------------------------------------------------------------

def stratified_eval_rows(
    *, model, tokenizer, rows: list[dict[str, Any]], template: str,
    direction_tokens: dict[str, str], keys: tuple[str, ...] = ("topic", "length", "feature"),
    max_new_tokens: int = 256, num_beams: int = 4, batch_size: int = 8,
) -> dict[str, Any]:
    """Generate both directions over `rows`, return per-direction stratified reports
    plus the flagged deficiencies. `rows` must carry {en, zh, ...metadata}."""
    from .eval import generate_translations

    result: dict[str, Any] = {"n": len(rows), "directions": {}, "deficiencies": []}
    for direction, src_field, tgt_field, tgt_lang in (
        ("en2zh", "en", "zh", "zh"), ("zh2en", "zh", "en", "en"),
    ):
        hyps = generate_translations(
            model=model, tokenizer=tokenizer, rows=rows, direction=direction,
            template=template, direction_tokens=direction_tokens,
            max_new_tokens=max_new_tokens, num_beams=num_beams, batch_size=batch_size,
        )
        refs = [r[tgt_field] for r in rows]
        seg = segment_scores(refs, hyps, tgt_lang=tgt_lang)
        rep = stratify(rows, seg, direction=direction, keys=keys, tgt_lang=tgt_lang)
        result["directions"][direction] = rep
        result["deficiencies"].extend(find_deficiencies(rep))
    return result


def evaluate_checkpoints_over_time(
    *, run_dir: Path, eval_rows: list[dict[str, Any]], template: str,
    direction_tokens: dict[str, str], out_path: Path | None = None,
    keys: tuple[str, ...] = ("topic", "length", "feature"),
    max_new_tokens: int = 256, num_beams: int = 4, batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Run the stratified eval on EVERY retained checkpoint in `run_dir`, appending one
    record per checkpoint to `eval_curve.jsonl`. Lets you watch each capability's learning
    curve and pick the checkpoint to resume for the next data wave."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = Path(run_dir)
    out_path = out_path or (run_dir / "eval_curve.jsonl")
    ckpts = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if (run_dir / "config.json").exists():
        ckpts.append(run_dir)  # final model

    curve: list[dict[str, Any]] = []
    for ckpt in ckpts:
        step = int(ckpt.name.split("-")[-1]) if ckpt.name.startswith("checkpoint-") else -1
        tok = AutoTokenizer.from_pretrained(str(ckpt), use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt), torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        rep = stratified_eval_rows(
            model=model, tokenizer=tok, rows=eval_rows, template=template,
            direction_tokens=direction_tokens, keys=keys,
            max_new_tokens=max_new_tokens, num_beams=num_beams, batch_size=batch_size,
        )
        record = {"step": step, "checkpoint": ckpt.name, **rep}
        curve.append(record)
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return curve
