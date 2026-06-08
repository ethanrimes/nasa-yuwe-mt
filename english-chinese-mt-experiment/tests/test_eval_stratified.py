"""Tests for src/ecmt/training/eval_stratified.py pure-aggregation layer.

Only the model-free functions are exercised here (length_bucket / stratify /
find_deficiencies) so the deficiency-diagnosis logic is verified on CPU without
torch, transformers, or sacrebleu.
"""

from __future__ import annotations

from ecmt.training import eval_stratified as es


def test_length_bucket_boundaries():
    assert es.length_bucket("a b c") == "short"            # 3 words
    assert es.length_bucket(" ".join(["w"] * 10)) == "medium"  # 10 -> medium (lo inclusive)
    assert es.length_bucket(" ".join(["w"] * 24)) == "medium"
    assert es.length_bucket(" ".join(["w"] * 25)) == "long"    # 25 -> long
    assert es.length_bucket("") == "short"


def _rows_seg():
    rows = [
        {"en": "a", "zh": "x", "topic": "health", "feature": "past"},
        {"en": "b", "zh": "y", "topic": "health", "feature": "past"},
        {"en": "c", "zh": "z", "topic": "news", "feature": "plural"},
        {"en": "d", "zh": "w", "topic": "news", "feature": "plural"},
    ]
    seg = [
        {"bleu": 10.0, "chrf": 20.0},
        {"bleu": 20.0, "chrf": 30.0},
        {"bleu": 80.0, "chrf": 90.0},
        {"bleu": 90.0, "chrf": 95.0},
    ]
    return rows, seg


def test_stratify_groups_and_means():
    rows, seg = _rows_seg()
    rep = es.stratify(rows, seg, direction="en2zh", keys=("topic", "feature"), tgt_lang="zh")
    assert rep["direction"] == "en2zh"
    assert rep["overall"]["n"] == 4
    assert rep["overall"]["chrf"] == 58.75
    # health is the lagging topic; news is the strong one.
    assert rep["by"]["topic"]["health"] == {"n": 2, "bleu": 15.0, "chrf": 25.0}
    assert rep["by"]["topic"]["news"] == {"n": 2, "bleu": 85.0, "chrf": 92.5}


def test_stratify_length_uses_target_side():
    # 11 English words but the *target* (zh) is short -> bucket = short (en2zh).
    rows = [{"en": " ".join(["w"] * 11), "zh": "x", "topic": "t"}]
    seg = [{"bleu": 1.0, "chrf": 1.0}]
    rep = es.stratify(rows, seg, direction="en2zh", keys=("length",), tgt_lang="zh")
    assert list(rep["by"]["length"].keys()) == ["short"]


def test_find_deficiencies_flags_lagging_strata():
    rows, seg = _rows_seg()
    rep = es.stratify(rows, seg, direction="en2zh", keys=("topic", "feature"), tgt_lang="zh")
    # overall chrf = 58.75; floor = 0.85*58.75 = 49.9 -> health (25) flagged, news (92.5) not.
    defs = es.find_deficiencies(rep, metric="chrf", rel_margin=0.85, min_n=2)
    flagged = {(d["key"], d["value"]) for d in defs}
    assert ("topic", "health") in flagged
    assert ("feature", "past") in flagged
    assert ("topic", "news") not in flagged
    # negative gap == below overall, sorted worst-first.
    assert defs[0]["gap_vs_overall"] < 0


def test_find_deficiencies_respects_min_n():
    rows, seg = _rows_seg()
    rep = es.stratify(rows, seg, direction="en2zh", keys=("topic",), tgt_lang="zh")
    # min_n higher than any stratum size -> nothing flagged.
    assert es.find_deficiencies(rep, min_n=99) == []
