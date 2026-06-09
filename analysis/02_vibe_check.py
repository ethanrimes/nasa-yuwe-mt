"""Vibe-check translations for all trained models (CPU inference).

For each model present under analysis/models_cache/<run>/ :
  * en-zh causal-LM cells  -> en2zh + zh2en on an in-domain and an out-of-domain set
  * NLLB seq2seq runs      -> es->nasa + nasa->es on real pairs + OOD Spanish

Outputs:
  analysis/vibe_check/<run>.json         (structured)
  analysis/vibe_check/translations.md    (human-readable consolidated tables)
  analysis/vibe_check/_health.json       (per-model degeneracy flags)

Greedy decoding (num_beams=1) for speed; this is a qualitative sanity pass, not a
BLEU benchmark.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "english-chinese-mt-experiment" / "src"))

import torch  # noqa: E402

CACHE = REPO / "analysis" / "models_cache"
OUT = REPO / "analysis" / "vibe_check"
OUT.mkdir(parents=True, exist_ok=True)
NASA_JSONL = REPO / "english-chinese-mt-experiment" / "data-en-zh" / "source" / "nasa_yuwe_parallel_dataset.jsonl"
ENZH_JSONL = REPO / "english-chinese-mt-experiment" / "data-en-zh" / "processed" / "en_zh_parallel_dataset.jsonl"

TEMPLATE = "{direction}\n<|src|> {src} <|tgt|> {tgt}"
DIRTOKENS = {"en2zh": "<|en2zh|>", "zh2en": "<|zh2en|>"}

ENZH_RUNS = ["360m-sentence", "360m-sentvocab", "1p7b-sentence", "1p7b-sentvocab"]
NLLB_RUNS = [
    "nllb-600m-sent", "nllb-600m-sentvocab",
    "nllb-1.3b-sent", "nllb-1.3b-sentvocab",
    "nllb-3.3b-sent", "nllb-3.3b-sentvocab",
]

# ---- evaluation sentence sets ------------------------------------------------ #
# en-zh in-domain sets are loaded at runtime from the real parallel corpus so we
# can show generations against the ACTUAL ground-truth target (ref column).
# We sample from two registers present in training: religious (bible) and
# constitutional (legal/governance) to gauge how generation quality varies.
ENZH_DOMAINS = {"religious_bible": "religious", "literary_cultural": "literary"}


def _load_enzh_pairs(domain: str, n: int = 3) -> list[dict]:
    """Deterministically pick n real sentence-level {en, zh} pairs for a domain."""
    pairs: list[dict] = []
    if not ENZH_JSONL.exists():
        return pairs
    with ENZH_JSONL.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("domain") != domain or r.get("level") != "sentence":
                continue
            en, zh = (r.get("en") or "").strip(), (r.get("zh") or "").strip()
            # keep medium-length, single-sentence-ish examples for readability
            if not en or not zh or len(en) > 160 or len(en) < 25:
                continue
            pairs.append({"en": en, "zh": zh})
            if len(pairs) >= n:
                break
    return pairs


# Out-of-domain Spanish prompts for es->nasa (no reference; judging fluency only).
ES_OOD = [
    "El gobierno aprobó una nueva ley económica.",
    "Por favor reinicie el servidor desde la terminal.",
]


def _degenerate(text: str) -> bool:
    """Heuristic: empty, all-eos/pad, NaN-ish, or a single token repeated."""
    t = text.strip()
    if not t:
        return True
    toks = t.split()
    if len(toks) >= 4 and len(set(toks)) == 1:
        return True
    # long run of one repeated char
    if re.search(r"(.)\1{15,}", t):
        return True
    return False


# ---- en-zh (causal LM) ------------------------------------------------------- #
def vibe_enzh(run: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ecmt.training.eval import generate_translations

    mdir = CACHE / run
    print(f"[en-zh] loading {run} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(mdir))
    model = AutoModelForCausalLM.from_pretrained(str(mdir), torch_dtype=torch.float32)
    result = {"run": run, "kind": "en-zh-causal", "sets": {}}
    for setname, domain in ENZH_DOMAINS.items():
        pairs = _load_enzh_pairs(domain, 3)
        if not pairs:
            continue
        result["sets"][setname] = {}
        for direction in ("en2zh", "zh2en"):
            hyps = generate_translations(
                model=model, tokenizer=tok, rows=pairs, direction=direction,
                template=TEMPLATE, direction_tokens=DIRTOKENS,
                max_new_tokens=64, num_beams=1, batch_size=3, device="cpu",
            )
            # for en2zh the gold target is zh; for zh2en it is en
            src_key = "en" if direction == "en2zh" else "zh"
            ref_key = "zh" if direction == "en2zh" else "en"
            result["sets"][setname][direction] = [
                {"src": r[src_key], "hyp": h, "ref": r[ref_key]} for r, h in zip(pairs, hyps)
            ]
    del model
    return result


# ---- NLLB (seq2seq) ---------------------------------------------------------- #
def _load_nasa_pairs(n: int = 3) -> list[dict]:
    pairs = []
    if NASA_JSONL.exists():
        with NASA_JSONL.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("spanish") and r.get("nasa_yuwe"):
                    pairs.append({"spanish": r["spanish"], "nasa_yuwe": r["nasa_yuwe"]})
                if len(pairs) >= n:
                    break
    return pairs


def _nllb_generate(model, tok, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
    from snmt.lang import NLLB_EOS_ID, build_input_ids, lang_token_id

    src_id = lang_token_id(tok, src_lang)
    tgt_id = lang_token_id(tok, tgt_lang)
    outs = []
    for txt in texts:
        content = tok(txt, add_special_tokens=False, truncation=True, max_length=180)["input_ids"]
        input_ids = build_input_ids(content, src_id, NLLB_EOS_ID)
        enc = {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long),
        }
        with torch.no_grad():
            gen = model.generate(
                **enc, forced_bos_token_id=tgt_id, max_new_tokens=80, num_beams=1, do_sample=False,
            )
        outs.append(tok.batch_decode(gen, skip_special_tokens=True)[0])
    return outs


def vibe_nllb(run: str) -> dict:
    sys.path.insert(0, str(REPO / "spanish-nasa-mt-experiment" / "src"))
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from snmt.lang import NASA_LANG, SPANISH_LANG, ensure_nasa_lang_token

    mdir = CACHE / run
    print(f"[nllb] loading {run} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(mdir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(mdir), torch_dtype=torch.float32)
    ensure_nasa_lang_token(tok, model)  # idempotent; token already in saved tokenizer
    model.eval()

    # NaN weight check — a diverged run often serialises NaN/inf params.
    nan_params = 0
    total = 0
    for p in model.parameters():
        total += 1
        if torch.isnan(p).any() or torch.isinf(p).any():
            nan_params += 1

    pairs = _load_nasa_pairs(3)
    es_in = [p["spanish"] for p in pairs]
    nasa_in = [p["nasa_yuwe"] for p in pairs]

    es2nasa_in = _nllb_generate(model, tok, es_in, SPANISH_LANG, NASA_LANG)
    nasa2es_in = _nllb_generate(model, tok, nasa_in, NASA_LANG, SPANISH_LANG)
    es2nasa_ood = _nllb_generate(model, tok, ES_OOD, SPANISH_LANG, NASA_LANG)

    result = {
        "run": run, "kind": "nllb-seq2seq",
        "nan_or_inf_param_tensors": nan_params, "param_tensors": total,
        "sets": {
            "in_domain": {
                "es2nasa": [{"src": s, "ref": p["nasa_yuwe"], "hyp": h}
                            for s, p, h in zip(es_in, pairs, es2nasa_in)],
                "nasa2es": [{"src": s, "ref": p["spanish"], "hyp": h}
                            for s, p, h in zip(nasa_in, pairs, nasa2es_in)],
            },
            "out_of_domain": {
                "es2nasa": [{"src": s, "hyp": h} for s, h in zip(ES_OOD, es2nasa_ood)],
            },
        },
    }
    del model
    return result


# ---- driver ------------------------------------------------------------------ #
def _flag_health(res: dict) -> dict:
    flat = []
    for setname, dirs in res["sets"].items():
        for direction, items in dirs.items():
            for it in items:
                flat.append(_degenerate(it["hyp"]))
    deg = sum(flat)
    return {
        "run": res["run"], "kind": res["kind"],
        "n_outputs": len(flat), "n_degenerate": deg,
        "degenerate_frac": round(deg / max(1, len(flat)), 3),
        "nan_param_tensors": res.get("nan_or_inf_param_tensors"),
    }


def main(argv: list[str]) -> int:
    only = set(argv) if argv else None
    health = []
    results = []
    for run in ENZH_RUNS:
        if (only and run not in only) or not (CACHE / run / "config.json").exists():
            continue
        try:
            r = vibe_enzh(run)
            (OUT / f"{run}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(r)
            health.append(_flag_health(r))
            print(f"[en-zh] {run} done", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[en-zh] {run} FAILED: {e}", flush=True)
    for run in NLLB_RUNS:
        if (only and run not in only) or not (CACHE / run / "config.json").exists():
            continue
        try:
            r = vibe_nllb(run)
            (OUT / f"{run}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(r)
            health.append(_flag_health(r))
            print(f"[nllb] {run} done (nan_param_tensors={r['nan_or_inf_param_tensors']})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[nllb] {run} FAILED: {e}", flush=True)

    (OUT / "_health.json").write_text(json.dumps(health, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(results, health)
    print("[vibe] wrote", OUT / "translations.md", flush=True)
    return 0


def _write_markdown(results: list[dict], health: list[dict]) -> None:
    lines = ["# Translation vibe-check\n",
             "Greedy (num_beams=1) CPU decoding. Qualitative sanity pass.\n",
             "\n## Health summary\n",
             "| run | kind | outputs | degenerate | degenerate frac | NaN param tensors |",
             "|---|---|---|---|---|---|"]
    for h in health:
        lines.append(f"| {h['run']} | {h['kind']} | {h['n_outputs']} | {h['n_degenerate']} "
                     f"| {h['degenerate_frac']} | {h.get('nan_param_tensors')} |")
    for r in results:
        lines.append(f"\n## {r['run']}  ({r['kind']})\n")
        for setname, dirs in r["sets"].items():
            lines.append(f"\n### {setname}\n")
            for direction, items in dirs.items():
                lines.append(f"\n**{direction}**\n")
                lines.append("| source | hypothesis | reference |")
                lines.append("|---|---|---|")
                for it in items:
                    src = it["src"].replace("|", "\\|")
                    hyp = (it["hyp"] or "∅").replace("|", "\\|").replace("\n", " ")
                    ref = it.get("ref", "").replace("|", "\\|")
                    lines.append(f"| {src} | {hyp} | {ref} |")
    (OUT / "translations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
