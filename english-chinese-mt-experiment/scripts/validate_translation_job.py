#!/usr/bin/env python3
"""Validate a translation job's output batches and report exactly what's missing.

For each ``batch_*.input.json`` in a job, checks the matching ``.output.json``:
  * exists and parses as a JSON array
  * has exactly the same set of ids as the input (same count, no extras/missing)
  * every element has non-empty ``en`` and ``zh``

Prints a per-job summary plus, for any defective batch, a one-line reason. With
``--list-bad`` it prints ONLY the absolute input paths of batches needing (re)work,
one per line (handy for feeding a dispatch loop).

Usage:
    python scripts/validate_translation_job.py nonbible
    python scripts/validate_translation_job.py nonbible --list-bad
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nymt_shared import translate  # noqa: E402


def _check_one(inp: Path) -> tuple[bool, str]:
    out = translate.output_path(inp)
    payload = json.loads(inp.read_text(encoding="utf-8"))
    want_ids = [int(it["id"]) for it in payload["items"]]
    want = set(want_ids)
    if not out.exists():
        return False, "missing output"
    try:
        rows = json.loads(out.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return False, f"bad json: {e}"
    if not isinstance(rows, list):
        return False, "output not a list"
    got_ids = []
    for r in rows:
        if "id" not in r:
            return False, "row missing id"
        got_ids.append(int(r["id"]))
    got = set(got_ids)
    if got != want:
        miss = len(want - got)
        extra = len(got - want)
        return False, f"id mismatch (missing {miss}, extra {extra}, in={len(want)}, out={len(got)})"
    bad = sum(1 for r in rows if not str(r.get("en", "")).strip() or not str(r.get("zh", "")).strip())
    if bad:
        return False, f"{bad} rows with empty en/zh"
    return True, f"ok {len(rows)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--list-bad", action="store_true",
                    help="print only absolute input paths of batches needing work")
    args = ap.parse_args()

    inputs = translate.input_files(args.job)
    if not inputs:
        print(f"no input batches for job '{args.job}'", file=sys.stderr)
        return 2

    bad: list[Path] = []
    ok = 0
    reasons: list[str] = []
    for inp in inputs:
        good, reason = _check_one(inp)
        if good:
            ok += 1
        else:
            bad.append(inp)
            reasons.append(f"{inp.name}: {reason}")

    if args.list_bad:
        for p in bad:
            print(str(p))
        return 0

    print(f"job '{args.job}': {len(inputs)} batches | ok={ok} | bad/missing={len(bad)}")
    for r in reasons:
        print("  " + r)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
