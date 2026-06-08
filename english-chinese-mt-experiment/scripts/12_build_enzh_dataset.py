#!/usr/bin/env python3
"""Merge reconstructed Bible + machine-translated halves into the final En-Zh dataset.

Inputs (produced by ``10_reconstruct_from_nasa.py`` + Gemini subagent fulfilment):

  * ``data-en-zh/interim/bible_pairs.jsonl`` — real WEB/CUV pairs (recon=bible_real).
  * ``data-en-zh/queue/<job>/batch_*.input.json`` — per-id metadata for MT rows.
  * ``data-en-zh/queue/<job>/batch_*.output.json`` — Gemini en/zh per id.
    Jobs merged: ``nonbible`` (original reconstruction) + ``external_repos``
    (community-corpus ingestion from script 14). See ``MT_JOBS``.

Output:

  * ``data-en-zh/processed/en_zh_parallel_dataset.jsonl`` — the row-for-row En-Zh
    mirror of the Nasa-Spanish set (schema: en, zh, source, domain, level,
    nasa_id, recon, meta).
  * ``data-en-zh/subsets/sentence_only.jsonl``        (level == sentence)
  * ``data-en-zh/subsets/sentence_plus_vocab.jsonl``  (all rows)

These two subsets are the two *core* data ablations. The build is deterministic
(sorted by nasa_id) so re-runs are stable. Spends no GPU.

Usage:
    python scripts/12_build_enzh_dataset.py            # build from whatever is fulfilled
    python scripts/12_build_enzh_dataset.py --require-complete   # fail unless all 24,174 rows present
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data-en-zh"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
SUBSETS = DATA / "subsets"
BIBLE_PAIRS = INTERIM / "bible_pairs.jsonl"

from nymt_shared import translate  # noqa: E402

# Jobs whose Gemini outputs feed the final dataset. `nonbible` is the original
# 16,272-item reconstruction; `external_repos` is the community-corpus ingestion
# (script 14) that appended nasa_id >= 24174.
MT_JOBS = ["nonbible", "external_repos"]

EXPECTED_TOTAL = 24229
EXPECTED_SENTENCE = 16446
EXPECTED_VOCAB = 7783


def load_bible_rows() -> list[dict]:
    rows: list[dict] = []
    if not BIBLE_PAIRS.exists():
        return rows
    with BIBLE_PAIRS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_mt_metadata() -> dict[int, dict]:
    """nasa_id -> {spanish, gloss, level, source, domain} from queue input files,
    merged across every job in MT_JOBS."""
    meta: dict[int, dict] = {}
    for job in MT_JOBS:
        qdir = translate.QUEUE_ROOT / job
        if not qdir.exists():
            continue
        for inp in sorted(qdir.glob("batch_*.input.json")):
            payload = json.loads(inp.read_text(encoding="utf-8"))
            for it in payload["items"]:
                meta[int(it["id"])] = it
    return meta


def collect_translations() -> dict:
    """Merge {id: {en, zh}} across every job in MT_JOBS."""
    out: dict = {}
    for job in MT_JOBS:
        if not (translate.QUEUE_ROOT / job).exists():
            continue
        out.update(translate.collect(job, strict=False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--require-complete",
        action="store_true",
        help="exit non-zero unless every queued MT row has been fulfilled",
    )
    args = ap.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    SUBSETS.mkdir(parents=True, exist_ok=True)

    bible_rows = load_bible_rows()
    mt_meta = load_mt_metadata()
    translated = collect_translations()  # {id(str): {en, zh}} merged across MT_JOBS

    total_queued = len(mt_meta)
    fulfilled = len(translated)
    missing = total_queued - fulfilled
    print(f"bible_real rows : {len(bible_rows)}")
    print(f"mt queued ids   : {total_queued}")
    print(f"mt fulfilled    : {fulfilled}  (missing {missing})")

    if args.require_complete and missing:
        print(
            f"ERROR: --require-complete set but {missing} MT rows unfulfilled.",
            file=sys.stderr,
        )
        return 2

    mt_rows: list[dict] = []
    for nid, m in mt_meta.items():
        out = translated.get(str(nid)) or translated.get(nid)
        if not out:
            continue
        en = (out.get("en") or "").strip()
        zh = (out.get("zh") or "").strip()
        if not en or not zh:
            continue
        mt_rows.append(
            {
                "en": en,
                "zh": zh,
                "source": m.get("source"),
                "domain": m.get("domain"),
                "level": m.get("level"),
                "nasa_id": nid,
                "recon": "mt_gemini",
                "meta": {},
            }
        )

    all_rows = bible_rows + mt_rows
    all_rows.sort(key=lambda r: (r.get("nasa_id") is None, r.get("nasa_id", 0)))

    out_path = PROCESSED / "en_zh_parallel_dataset.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sentence_rows = [r for r in all_rows if r.get("level") == "sentence"]
    vocab_rows = [r for r in all_rows if r.get("level") == "vocabulary"]

    sent_path = SUBSETS / "sentence_only.jsonl"
    with sent_path.open("w", encoding="utf-8") as f:
        for r in sentence_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    full_path = SUBSETS / "sentence_plus_vocab.jsonl"
    with full_path.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {out_path}  ({len(all_rows)} rows)")
    print(f"  subset sentence_only        : {len(sentence_rows)}")
    print(f"  subset sentence_plus_vocab  : {len(all_rows)}")
    print(f"  (vocabulary rows            : {len(vocab_rows)})")

    if missing == 0:
        ok = (
            len(all_rows) == EXPECTED_TOTAL
            and len(sentence_rows) == EXPECTED_SENTENCE
            and len(vocab_rows) == EXPECTED_VOCAB
        )
        print(
            f"\nfull-build check: total={len(all_rows)}/{EXPECTED_TOTAL} "
            f"sentence={len(sentence_rows)}/{EXPECTED_SENTENCE} "
            f"vocab={len(vocab_rows)}/{EXPECTED_VOCAB} -> {'OK' if ok else 'MISMATCH'}"
        )
        if not ok:
            return 3

    manifest = {
        "bible_real": len(bible_rows),
        "mt_queued": total_queued,
        "mt_fulfilled": fulfilled,
        "mt_missing": missing,
        "total_rows": len(all_rows),
        "sentence_only": len(sentence_rows),
        "sentence_plus_vocab": len(all_rows),
        "vocabulary": len(vocab_rows),
        "complete": missing == 0,
    }
    (PROCESSED / "build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
