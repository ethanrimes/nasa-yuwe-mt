"""LLM-as-judge driver (agent-fulfilled queue) for MT evaluation.

Single-reference n-gram metrics (BLEU) under-credit the many valid translations of one
input. This module adds a complementary **LLM-as-judge** track, run *off the GPU* on a
run's held-out hypotheses (FLORES+/XNLI) so it never holds the expensive H100.

Like `translate.py`, judging is modeled as a *queue the orchestrating agent fills* (we have
no Gemini/OpenAI API key, but the agent can spawn judge subagents):

    1. After a run, `enqueue_score(items, job=...)` or `enqueue_pairwise(items, job=...)`
       writes `queue/<job>/batch_XXXX.input.json` (chunked).
    2. The agent lists `pending(job)`, dispatches a judge subagent per batch with
       `render_prompt(input_file)`, and writes the returned JSON via `write_output(...)`.
    3. `collect(job)` merges outputs and `aggregate(job)` reduces them to per-system means
       (score mode) or win-rates (pairwise mode).

Two modes
---------
* **score**    — absolute rubric. Each item: {"id", "direction", "source", "reference",
                 "hypothesis"}. Judge returns {"id","adequacy","fluency","terminology",
                 "overall"} each 1-5 (+ optional "rationale").
* **pairwise** — A/B between two checkpoints on the same input. Each item: {"id",
                 "direction", "source", "reference", "hyp_a", "hyp_b"}. Judge returns
                 {"id","winner": "A"|"B"|"tie","confidence": 1-5} (+ optional rationale).
                 `hyp_a`/`hyp_b` are randomly labeled by the caller to avoid position bias.

For CI / dry-runs, `stub_fulfill(job)` writes deterministic placeholder verdicts so the
pipeline is testable without a real judge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config

QUEUE_ROOT = config.REPO_ROOT / "english-chinese-mt-experiment" / "eval" / "judge_queue"

SCORE_PROMPT = """\
You are an expert bilingual machine-translation evaluator. For each item, judge the
HYPOTHESIS translation of SOURCE (direction `{direction}`) against the REFERENCE, knowing
that MANY different translations can be fully correct — do not penalize valid paraphrases or
word-order differences that preserve meaning and register.

Score each item on three axes, integers 1-5 (5 = best):
- adequacy:   is all and only the source meaning conveyed? (no additions/omissions)
- fluency:    is the output natural, grammatical target-language text?
- terminology: are key terms / named entities / register rendered correctly?
Then give `overall` 1-5 as your holistic judgement.

Output ONLY a JSON array, one element per input item, same ids, same order:
{"id": <same id>, "adequacy": <1-5>, "fluency": <1-5>, "terminology": <1-5>,
 "overall": <1-5>, "rationale": <one short clause>}

Judge these {n} items:
{items_json}
"""

PAIRWISE_PROMPT = """\
You are an expert bilingual machine-translation evaluator. For each item you are given a
SOURCE (direction `{direction}`), a REFERENCE, and two candidate translations A (`hyp_a`)
and B (`hyp_b`) produced by two different model checkpoints. Decide which candidate is the
better translation, knowing MANY translations can be valid — prefer the one with better
combined adequacy + fluency + terminology; choose "tie" only if genuinely indistinguishable.
The A/B labels carry NO information (they were randomized); judge content only.

Output ONLY a JSON array, one element per input item, same ids, same order:
{"id": <same id>, "winner": "A" | "B" | "tie", "confidence": <1-5>,
 "rationale": <one short clause>}

Judge these {n} items:
{items_json}
"""

_PROMPTS = {"score": SCORE_PROMPT, "pairwise": PAIRWISE_PROMPT}


def _job_dir(job: str) -> Path:
    d = QUEUE_ROOT / job
    d.mkdir(parents=True, exist_ok=True)
    return d


def _enqueue(items: list[dict], job: str, mode: str, direction: str, batch_size: int) -> list[Path]:
    if mode not in _PROMPTS:
        raise ValueError(f"mode must be one of {sorted(_PROMPTS)}; got {mode!r}")
    d = _job_dir(job)
    paths: list[Path] = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        idx = i // batch_size
        p = d / f"batch_{idx:04d}.input.json"
        p.write_text(
            json.dumps(
                {"job": job, "batch": idx, "mode": mode, "direction": direction, "items": chunk},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        paths.append(p)
    return paths


def enqueue_score(items: list[dict], job: str, direction: str = "mixed", batch_size: int = 100) -> list[Path]:
    """Queue absolute-rubric scoring items. Each: {id, source, reference, hypothesis}."""
    return _enqueue(items, job, "score", direction, batch_size)


def enqueue_pairwise(items: list[dict], job: str, direction: str = "mixed", batch_size: int = 100) -> list[Path]:
    """Queue A/B items. Each: {id, source, reference, hyp_a, hyp_b}."""
    return _enqueue(items, job, "pairwise", direction, batch_size)


def input_files(job: str) -> list[Path]:
    return sorted(_job_dir(job).glob("batch_*.input.json"))


def output_path(input_file: Path) -> Path:
    return input_file.with_name(input_file.name.replace(".input.json", ".output.json"))


def pending(job: str) -> list[Path]:
    return [p for p in input_files(job) if not output_path(p).exists()]


def render_prompt(input_file: Path) -> str:
    """Build the judge-subagent prompt for one batch."""
    data = json.loads(input_file.read_text(encoding="utf-8"))
    items = data["items"]
    tmpl = _PROMPTS[data["mode"]]
    # Literal replacement (template embeds literal JSON braces in the schema example).
    return (
        tmpl.replace("{direction}", str(data.get("direction", "mixed")))
        .replace("{n}", str(len(items)))
        .replace("{items_json}", json.dumps(items, ensure_ascii=False, indent=2))
    )


def write_output(input_file: Path, verdicts: list[dict]) -> Path:
    out = output_path(input_file)
    out.write_text(json.dumps(verdicts, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def stub_fulfill(job: str) -> int:
    """Fill pending batches with deterministic placeholder verdicts (dry-run tests)."""
    n = 0
    for inp in pending(job):
        data = json.loads(inp.read_text(encoding="utf-8"))
        if data["mode"] == "score":
            verdicts = [
                {"id": it["id"], "adequacy": 3, "fluency": 3, "terminology": 3,
                 "overall": 3, "rationale": "[stub]"}
                for it in data["items"]
            ]
        else:
            verdicts = [
                {"id": it["id"], "winner": "tie", "confidence": 1, "rationale": "[stub]"}
                for it in data["items"]
            ]
        write_output(inp, verdicts)
        n += 1
    return n


def collect(job: str, strict: bool = True) -> list[dict]:
    """Merge all batch outputs into a flat list of verdicts."""
    merged: list[dict] = []
    for inp in input_files(job):
        out = output_path(inp)
        if not out.exists():
            if strict:
                raise RuntimeError(f"missing judge output for {inp.name}; run the agent fulfill step")
            continue
        merged.extend(json.loads(out.read_text(encoding="utf-8")))
    return merged


def aggregate(job: str, strict: bool = False) -> dict:
    """Reduce verdicts to summary stats (score means or pairwise win-rates)."""
    ins = input_files(job)
    if not ins:
        return {"job": job, "n": 0}
    mode = json.loads(ins[0].read_text(encoding="utf-8"))["mode"]
    rows = collect(job, strict=strict)
    if not rows:
        return {"job": job, "mode": mode, "n": 0}
    if mode == "score":
        keys = ["adequacy", "fluency", "terminology", "overall"]
        means = {k: round(sum(float(r.get(k, 0)) for r in rows) / len(rows), 3) for k in keys}
        return {"job": job, "mode": mode, "n": len(rows), "means": means}
    wins = {"A": 0, "B": 0, "tie": 0}
    for r in rows:
        wins[str(r.get("winner", "tie"))] = wins.get(str(r.get("winner", "tie")), 0) + 1
    n = len(rows)
    decided = wins["A"] + wins["B"]
    return {
        "job": job, "mode": mode, "n": n, "wins": wins,
        "a_win_rate": round(wins["A"] / n, 3),
        "b_win_rate": round(wins["B"] / n, 3),
        "a_vs_b_excl_ties": round(wins["A"] / decided, 3) if decided else None,
    }


def status(job: str) -> dict:
    ins = input_files(job)
    pend = pending(job)
    return {
        "job": job,
        "batches": len(ins),
        "done": len(ins) - len(pend),
        "pending": len(pend),
        "pending_files": [p.name for p in pend],
    }


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="nymt_shared.llm_judge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("status"); p.add_argument("job")
    p = sub.add_parser("pending"); p.add_argument("job")
    p = sub.add_parser("prompt"); p.add_argument("input_file")
    p = sub.add_parser("stub"); p.add_argument("job")
    p = sub.add_parser("aggregate"); p.add_argument("job"); p.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "status":
        print(json.dumps(status(args.job), ensure_ascii=False, indent=2))
    elif args.cmd == "pending":
        for p in pending(args.job):
            print(p)
    elif args.cmd == "prompt":
        print(render_prompt(Path(args.input_file)))
    elif args.cmd == "stub":
        print(f"stub-filled {stub_fulfill(args.job)} batch(es)")
    elif args.cmd == "aggregate":
        agg = aggregate(args.job, strict=False)
        if args.out:
            Path(args.out).write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(agg, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
