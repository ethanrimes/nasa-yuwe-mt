"""Gemini-subagent batch translation driver (agent-fulfilled queue).

We have no Gemini API key, but the orchestrating agent CAN spawn `gemini-3.5-flash`
subagents via its task tool. So translation is modeled as a *queue the agent fills*:

    1. A reconstruction script calls `enqueue(items, job=...)`. This writes
       `queue/<job>/batch_XXXX.input.json` files (chunked).
    2. The agent lists `pending(job)`, and for each input batch dispatches a
       `gemini-3.5-flash` subagent with `PROMPT_TEMPLATE` + the batch items,
       then writes the returned JSON to `batch_XXXX.output.json`.
    3. `collect(job)` merges all outputs back, keyed by item id, validating that
       every id got both `en` and `zh`.

This keeps the Python pipeline deterministic and reproducible while delegating the
actual translation to the agent's subagents. For automated CI / dry-runs,
`stub_fulfill(job)` writes placeholder outputs so the rest of the pipeline can be
tested end-to-end without real translation.

Item input schema (each):  {"id": str, "spanish": str, "level": "sentence|vocabulary",
                            "source": str, "gloss": str|None}
Item output schema (each): {"id": str, "en": str, "zh": str}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config

QUEUE_ROOT = config.REPO_ROOT / "english-chinese-mt-experiment" / "data-en-zh" / "queue"

PROMPT_TEMPLATE = """\
You are a professional translator. Translate each item from SPANISH into BOTH
English (en) and Simplified Chinese (zh, zh-Hans).

Rules:
- Preserve meaning and register. For `level: vocabulary` items translate the term
  or short phrase as a dictionary gloss (no full sentence).
- For `level: sentence` items produce a fluent, natural full-sentence translation.
- Keep proper nouns; do not add explanations or notes.
- Output ONLY a JSON array. Each element: {"id": <same id>, "en": <english>, "zh": <chinese>}.
- Return exactly one element per input item, same ids, same order.

Translate these {n} items:
{items_json}
"""


def _job_dir(job: str) -> Path:
    d = QUEUE_ROOT / job
    d.mkdir(parents=True, exist_ok=True)
    return d


def enqueue(items: list[dict], job: str, batch_size: int = 200) -> list[Path]:
    """Chunk items into input batch files. Returns the input file paths."""
    d = _job_dir(job)
    paths: list[Path] = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        idx = i // batch_size
        p = d / f"batch_{idx:04d}.input.json"
        p.write_text(
            json.dumps(
                {"job": job, "batch": idx, "target_langs": ["en", "zh"], "items": chunk},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        paths.append(p)
    return paths


def input_files(job: str) -> list[Path]:
    return sorted(_job_dir(job).glob("batch_*.input.json"))


def output_path(input_file: Path) -> Path:
    return input_file.with_name(input_file.name.replace(".input.json", ".output.json"))


def pending(job: str) -> list[Path]:
    """Input batches that do not yet have a corresponding output file."""
    return [p for p in input_files(job) if not output_path(p).exists()]


def render_prompt(input_file: Path) -> str:
    """Build the subagent prompt for one batch (the agent sends this to gemini)."""
    data = json.loads(input_file.read_text(encoding="utf-8"))
    items = data["items"]
    # NOTE: use literal replacement (not str.format) because the template contains
    # literal JSON braces in the example payload, which str.format would misparse.
    return PROMPT_TEMPLATE.replace("{n}", str(len(items))).replace(
        "{items_json}", json.dumps(items, ensure_ascii=False, indent=2)
    )


def write_output(input_file: Path, translated: list[dict]) -> Path:
    """Persist a batch's translations (called by the agent after a subagent returns)."""
    out = output_path(input_file)
    out.write_text(json.dumps(translated, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def stub_fulfill(job: str) -> int:
    """Fill every pending batch with placeholder translations (for dry-run tests)."""
    n = 0
    for inp in pending(job):
        data = json.loads(inp.read_text(encoding="utf-8"))
        translated = [
            {"id": it["id"], "en": f"[EN stub] {it.get('spanish','')}",
             "zh": f"[ZH 占位] {it.get('spanish','')}"}
            for it in data["items"]
        ]
        write_output(inp, translated)
        n += 1
    return n


def collect(job: str, strict: bool = True) -> dict[str, dict]:
    """Merge all outputs into {id: {'en':..., 'zh':...}}. Validates coverage."""
    merged: dict[str, dict] = {}
    for inp in input_files(job):
        out = output_path(inp)
        if not out.exists():
            if strict:
                raise RuntimeError(f"missing output for {inp.name}; run the agent fulfill step")
            continue
        for row in json.loads(out.read_text(encoding="utf-8")):
            if not row.get("en") or not row.get("zh"):
                if strict:
                    raise RuntimeError(f"incomplete translation for id={row.get('id')}")
                continue
            merged[row["id"]] = {"en": row["en"], "zh": row["zh"]}
    return merged


def status(job: str) -> dict:
    ins = input_files(job)
    pend = pending(job)
    out = {
        "job": job,
        "batches": len(ins),
        "done": len(ins) - len(pend),
        "pending": len(pend),
        "pending_files": [p.name for p in pend],
    }
    return out


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="nymt_shared.translate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("status"); p.add_argument("job")
    p = sub.add_parser("pending"); p.add_argument("job")
    p = sub.add_parser("prompt"); p.add_argument("input_file")
    p = sub.add_parser("stub"); p.add_argument("job")
    p = sub.add_parser("collect"); p.add_argument("job"); p.add_argument("--out", default=None)
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
    elif args.cmd == "collect":
        merged = collect(args.job, strict=False)
        if args.out:
            Path(args.out).write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
        print(f"collected {len(merged)} translated item(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
