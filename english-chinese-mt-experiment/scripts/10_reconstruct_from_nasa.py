#!/usr/bin/env python3
"""Reconstruct an English-Chinese proxy of the Nasa-Spanish parallel dataset.

The goal is a row-for-row En-Zh mirror of ``nasa_yuwe_parallel_dataset.jsonl`` so
that every data-ablation we run on En-Zh is 1:1 comparable to what the real
Spanish->Nasa model would see. We preserve each row's ``source``, ``domain`` and
the derived ``level`` (sentence|vocabulary), and attach an ``nasa_id`` so the
machine-translated half can be merged back deterministically.

Two reconstruction tracks:

  * **Bible (``bible_nt``)** — we already have exact ``book/chapter/verse`` refs,
    so we fetch the REAL published WEB (English) + CUV (Chinese, ->Simplified)
    text and inner-join. No MT, highest quality. Any verse missing on either
    side falls through to the MT queue so row counts still mirror.

  * **Everything else** — enqueued as batches for ``gemini-3.5-flash`` subagents
    (see ``nymt_shared.translate``). The agent fulfils the queue out-of-band;
    ``12_build_enzh_dataset.py`` then merges outputs back by ``nasa_id``.

This script performs NO training and spends NO GPU. Bible fetches are cached on
disk, so re-runs are free/offline.

Usage:
    python scripts/10_reconstruct_from_nasa.py            # classify + fetch bible + enqueue
    python scripts/10_reconstruct_from_nasa.py --no-fetch # classify + enqueue only (offline)
    python scripts/10_reconstruct_from_nasa.py --limit-nonbible 5  # tiny enqueue for dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data-en-zh"
SRC_JSONL = DATA / "source" / "nasa_yuwe_parallel_dataset.jsonl"
INTERIM = DATA / "interim"

from nymt_shared import bible, translate  # noqa: E402

# --------------------------------------------------------------------------- #
# Level classification — ported verbatim from the upstream build script so the
# derived `level` exactly matches the dataset's published summary counts
# (sentence=16,391 / vocabulary=7,783).
# --------------------------------------------------------------------------- #


def clean(text: str | None) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize("NFC", text.strip())
    return re.sub(r"\s+", " ", text)


def constitution_article_id(spanish: str) -> str | None:
    match = re.match(r"^Artículo\s+([^\.]+)\.", spanish or "", flags=re.IGNORECASE)
    if not match:
        return None
    return clean(match.group(1)).lower()


def vocabulary_dedup_key(record: dict) -> tuple[str, str] | None:
    source = record["source"]
    spanish = record["spanish"]
    nasa = record["nasa_yuwe"]
    if source not in {"living_dict", "swarthmore", "broomva_translation", "kwesxyuwe_thematic"}:
        if source != "broomva_instruct" or constitution_article_id(spanish):
            return None
        if len(spanish) > 80 or len(nasa) > 80:
            return None

    def normalize(text: str) -> str:
        text = unicodedata.normalize("NFKC", text).casefold()
        text = text.replace("\u2019", "'").replace("\u00b4", "'").replace("`", "'")
        return re.sub(r"\s+", " ", text).strip()

    nasa_key = normalize(nasa)
    nasa_key = re.sub(r"\s*\([^)]*\)\s*", " ", nasa_key)
    nasa_key = re.sub(r"\s+", " ", nasa_key).strip()

    spanish_key = normalize(spanish)
    spanish_key = re.sub(
        r"\s*\((?:adj|adv|m|f|v\.?[itr]?|pron(?:\.\w+)?|prep|conj|"
        r"interj|num|pl|sg|árbol|ave|animal|planta|insecto|fruta|"
        r"especie|hecha|hecho|de [^)]+|[^)]*\bv\.[^)]+)\)",
        "",
        spanish_key,
        flags=re.IGNORECASE,
    )
    spanish_key = re.sub(r"\b(el|la|los|las|un|una|unos|unas)\s+", "", spanish_key)
    spanish_key = re.sub(r"\s*,\s*(el|la|los|las|un|una|unos|unas)\s+", ", ", spanish_key)
    spanish_key = re.sub(r"\s+", " ", spanish_key).strip(" .;:,")
    return nasa_key, spanish_key


def data_level(record: dict) -> str:
    return "vocabulary" if vocabulary_dedup_key(record) else "sentence"


# --------------------------------------------------------------------------- #


def load_source() -> list[dict]:
    if not SRC_JSONL.exists():
        sys.exit(f"[reconstruct] source not found: {SRC_JSONL}\n"
                 f"  download it first: python -m nymt_shared.blob download "
                 f"data/nasa_yuwe_parallel_dataset.jsonl {SRC_JSONL}")
    rows = []
    for i, line in enumerate(SRC_JSONL.open(encoding="utf-8")):
        r = json.loads(line)
        r["nasa_id"] = i
        r["level"] = data_level(r)
        rows.append(r)
    return rows


def reconstruct_bible(rows: list[dict], do_fetch: bool) -> tuple[list[dict], list[dict]]:
    """Return (real_pairs, misses). Misses are rows to send to the MT queue."""
    bible_rows = [r for r in rows if r["source"] == "bible_nt" and (r.get("meta") or {}).get("verse")]
    # Group by (book, chapter) so we fetch each chapter once per translation.
    chapters: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in bible_rows:
        m = r["meta"]
        chapters[(m["book"], int(m["chapter"]))].append(r)

    real, misses = [], []
    total_ch = len(chapters)
    for n, ((book, chapter), group) in enumerate(sorted(chapters.items()), 1):
        if do_fetch:
            en_ch = bible.fetch_chapter(book, chapter, "web")
            zh_ch = bible.fetch_chapter(book, chapter, "cuv")
        else:
            en_ch, zh_ch = {}, {}
        for r in group:
            v = int(r["meta"]["verse"])
            en, zh = en_ch.get(v), zh_ch.get(v)
            if en and zh:
                real.append({
                    "en": en, "zh": zh,
                    "source": r["source"], "domain": r["domain"], "level": r["level"],
                    "nasa_id": r["nasa_id"], "recon": "bible_real",
                    "meta": r["meta"],
                })
            else:
                misses.append(r)
        if do_fetch and (n % 25 == 0 or n == total_ch):
            print(f"[bible] fetched {n}/{total_ch} chapters "
                  f"(real={len(real)} miss={len(misses)})", flush=True)
    return real, misses


def enqueue_nonbible(rows: list[dict], bible_misses: list[dict], limit: int | None) -> dict:
    """Queue all non-Bible rows (+ bible misses) for gemini-3.5-flash subagents."""
    to_mt = [r for r in rows if not (r["source"] == "bible_nt" and (r.get("meta") or {}).get("verse"))]
    to_mt = to_mt + bible_misses
    # Deterministic order by nasa_id.
    to_mt = sorted({r["nasa_id"]: r for r in to_mt}.values(), key=lambda r: r["nasa_id"])
    if limit is not None:
        to_mt = to_mt[:limit]

    items = []
    for r in to_mt:
        meta = r.get("meta") or {}
        items.append({
            "id": r["nasa_id"],
            "spanish": r["spanish"],
            "gloss": meta.get("gloss_en"),
            "level": r["level"],
            "source": r["source"],
            "domain": r["domain"],
        })
    paths = translate.enqueue(items, job="nonbible", batch_size=200)
    by_level = Counter(i["level"] for i in items)
    return {"queued": len(items), "batches": len(paths), "by_level": dict(by_level)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true", help="skip Bible HTTP fetch (offline)")
    ap.add_argument("--limit-nonbible", type=int, default=None, help="cap MT-queue size (dry-run)")
    args = ap.parse_args(argv)

    INTERIM.mkdir(parents=True, exist_ok=True)
    rows = load_source()
    lvl = Counter(r["level"] for r in rows)
    print(f"[reconstruct] loaded {len(rows)} rows  level={dict(lvl)}")

    real, misses = reconstruct_bible(rows, do_fetch=not args.no_fetch)
    out = INTERIM / "bible_pairs.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in real:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[reconstruct] bible real pairs={len(real)} misses={len(misses)} -> {out}")

    q = enqueue_nonbible(rows, misses, args.limit_nonbible)
    print(f"[reconstruct] MT queue: {q}")

    manifest = {
        "source_rows": len(rows),
        "level_counts": dict(lvl),
        "bible_real": len(real),
        "bible_misses": len(misses),
        "mt_queued": q["queued"],
        "mt_batches": q["batches"],
        "mt_by_level": q["by_level"],
        "fetched": not args.no_fetch,
    }
    (INTERIM / "reconstruct_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[reconstruct] manifest -> {INTERIM / 'reconstruct_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
