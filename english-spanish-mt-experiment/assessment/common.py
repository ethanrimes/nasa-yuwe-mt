"""Shared helpers for the assessment experiments.

Everything runs locally on CPU. We deliberately avoid importing the project
library so this folder is self-contained and reproducible from a fresh clone.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# UTF-8 stdout on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
CKPT_ROOT = ROOT / "checkpoints"
EVAL_DIR = ROOT / "data" / "eval"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parent / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# tier dir -> short label
TIERS = {
    "T10000": "T10k",
    "T50000": "T50k",
    "T100000": "T100k",
    "T500000": "T500k",
    "T1000000": "T1M",
}
TIER_PAIRS = {  # short -> nominal training pairs
    "T10k": 10000,
    "T50k": 50000,
    "T100k": 100000,
    "T500k": 500000,
    "T1M": 1000000,
}

# Best checkpoint step per tier (highest avg BLEU on the gen-eval subset; from
# the project's translate.py / summarize_metrics.py).
BEST_STEP = {
    "T10k": 600,
    "T50k": 7250,
    "T100k": 10500,
    "T500k": 30000,
    "T1M": 36000,
}

LANG = {"en": "English", "es": "Spanish"}


def list_checkpoints(tier_dir: str) -> list[tuple[int, Path]]:
    """Return [(step, path), ...] sorted by step for a tier directory name."""
    d = CKPT_ROOT / tier_dir
    out = []
    for p in d.glob("checkpoint-*"):
        m = re.match(r"checkpoint-(\d+)$", p.name)
        if m and (p / "config.json").exists():
            out.append((int(m.group(1)), p))
    return sorted(out, key=lambda x: x[0])


def representative_checkpoints(tier_dir: str, tier: str) -> list[tuple[int, Path]]:
    """Distinct {first, best, last} snapshots for a run.

    The saved trail is a rolling window: for the converged tiers all snapshots
    are post-convergence and nearly identical, while T10k and T1M keep an
    early *best* snapshot followed by a much later plateau. {first,best,last}
    captures the only states that actually differ, which is all we need to test
    for within-run forgetting (best vs. later).
    """
    ck = list_checkpoints(tier_dir)
    if not ck:
        return []
    steps = {ck[0][0], ck[-1][0]}
    best = BEST_STEP.get(tier)
    have = {s for s, _ in ck}
    if best in have:
        steps.add(best)
    return [(s, p) for s, p in ck if s in steps]


def make_prompt(text: str, direction: str) -> str:
    if direction == "en2es":
        return f"{LANG['en']}: {text}\n{LANG['es']}: "
    return f"{LANG['es']}: {text}\n{LANG['en']}: "


# ---------- word / literal tokenization (for recall metrics) ----------
# Keep letters incl. Spanish accents + digits; lowercase; split on the rest.
_WORD_RE = re.compile(r"[0-9a-zñáéíóúüïçàèìòù]+", re.IGNORECASE)

# Small stopword lists so "content literal" recall isn't dominated by function words.
_STOP_EN = set(
    "the a an and or but if then of to in on at for from by with as is are was were be been being "
    "this that these those it its he she they we you i him her them us me my your his their our "
    "do does did not no yes so than too very can could would should will shall may might must have "
    "has had what which who whom whose where when why how all any both each few more most other some "
    "such only own same s t".split()
)
_STOP_ES = set(
    "el la los las un una unos unas y o pero si de a en con por para como es son era eran ser sido "
    "este esta estos estas eso esa ese aquel lo le les su sus mi mis tu tus nos me te se que cual quien "
    "no si lo del al un una me te se nos os les ya muy mas más también porque cuando donde quien cómo "
    "qué cuál todo toda todos todas algun alguna alguno cada".split()
)


def tokenize_words(s: str) -> list[str]:
    return _WORD_RE.findall(s.lower())


def content_tokens(s: str, lang: str) -> set[str]:
    stop = _STOP_ES if lang == "es" else _STOP_EN
    return {w for w in tokenize_words(s) if w not in stop and len(w) > 1}


def word_recall(hyp: str, ref: str, lang: str) -> tuple[float, set, set]:
    """Recall of reference *content-word types* present in hypothesis.

    Returns (recall, hit_words, miss_words). Empty ref -> recall 1.0.
    """
    ref_set = content_tokens(ref, lang)
    if not ref_set:
        return 1.0, set(), set()
    hyp_set = set(tokenize_words(hyp))
    hits = {w for w in ref_set if w in hyp_set}
    miss = ref_set - hits
    return len(hits) / len(ref_set), hits, miss


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
