#!/usr/bin/env python3
"""Ingest external community Nasa-Spanish corpora, dedup, append, and enqueue MT.

Three community repos contribute additional Nasa Yuwe <-> Spanish parallel data:

  * ``darcygith/nasaYuwe``  — RASA NLU bot. Parallel pairs live in ``corpus.txt``
    (governance/autonomy) and ``corpus_cultural`` (cultural practices), each as
    ``- <nasa with [span](Entity) annotations>`` followed by ``# <spanish>``.
    ``data/nlu.yml`` is Nasa-monolingual augmentation (sub-phrase variants with no
    Spanish), so it is NOT parallel data and is intentionally skipped.
  * ``darcysem/nasaYuwe``  — a sibling/fork with the same ``corpus.txt`` /
    ``corpus_cultural`` layout (cultural pairs). Dedup folds away anything shared
    with darcygith.
  * ``juanks235/MT-Colombian-Indigenous-Languages`` — ``datasets/nasa_full_dataset.csv``
    (columns ``esp,nas``): the 1991 Colombian Constitution (Nasa translation) plus a
    bilingual dictionary. ``nasa_sin_cartas.csv`` and the ``*_train/dev/test`` files
    are strict subsets of ``nasa_full_dataset.csv`` so only the full file is read.

Our existing comprehensive set already contains the Constitution (``broomva_instruct``)
and a dictionary (``living_dict``), so juanks235 is expected to overlap heavily. This
script dedups every candidate against the existing source rows (and against each other)
using a punctuation-insensitive normalized ``(nasa, spanish)`` key, then:

  1. (``--write``) appends genuinely-new rows to
     ``data-en-zh/source/nasa_yuwe_parallel_dataset.jsonl`` with ``nasa_id`` continuing
     after the current max, ``source`` tagging the origin repo, and a heuristic
     ``level`` (vocabulary|sentence).
  2. (``--write``) enqueues the new rows for ``gemini-3.5-flash`` translation under the
     job ``external_repos`` (separate dir, so the existing ``nonbible`` queue is
     untouched). ``12_build_enzh_dataset.py`` merges this job alongside ``nonbible``.

Default (no flags) is a DRY RUN: it parses, dedups, prints a full overlap report and
writes a preview of the new rows to ``data-en-zh/interim/external_new_preview.jsonl``
without mutating the source set or the queue. Spends no GPU.

Usage:
    python scripts/14_ingest_external_repos.py            # dry-run report + preview
    python scripts/14_ingest_external_repos.py --write    # append + enqueue (idempotent)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MONOREPO = REPO.parent
DATA = REPO / "data-en-zh"
SRC_JSONL = DATA / "source" / "nasa_yuwe_parallel_dataset.jsonl"
INTERIM = DATA / "interim"
EXT_ROOT = MONOREPO / "_external_repos"

from nymt_shared import translate  # noqa: E402

JOB = "external_repos"

# --------------------------------------------------------------------------- #
# Normalization / parsing helpers
# --------------------------------------------------------------------------- #

_ANNOT = re.compile(r"\[([^\]]*)\]\([^)]*\)")  # [inner](Entity_Label) -> inner
# Bare RASA entity labels that leak into the Nasa text in some corpus_cultural rows,
# e.g. "... (Sanación_Espiritual) ..." or "(Defensa_Familia)" / "(Tulpa)". They start
# with a capital and contain only letters/underscore (no spaces), so legitimate
# Spanish-side parentheticals like "(CNRS)" are unaffected (Spanish is never stripped).
_BARE_LABEL = re.compile(r"\(\s*[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ_ ]*\)")


def strip_annotations(text: str) -> str:
    """Remove RASA ``[span](Entity)`` markup, keeping the inner span text."""
    prev = None
    while prev != text:
        prev = text
        text = _ANNOT.sub(r"\1", text)
    text = _BARE_LABEL.sub(" ", text)
    # drop any stray unmatched brackets / leftover parens at token level
    text = text.replace("[", " ").replace("]", " ")
    return re.sub(r"\s+", " ", text).strip()


def norm_key(text: str) -> str:
    """Punctuation-insensitive comparison key (NFKC casefold, alnum+space only)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).casefold()
    for a in ("\u2019", "\u00b4", "`", "\u2018"):
        text = text.replace(a, "'")
    # keep letters/digits/space, drop the rest (quotes, brackets, punctuation)
    text = re.sub(r"[^0-9a-z\u00c0-\u024f\u00f1' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def pair_key(nasa: str, spanish: str) -> str:
    return norm_key(strip_annotations(nasa)) + " ||| " + norm_key(spanish)


def token_overlap(nasa: str, spanish: str) -> float:
    """Jaccard token overlap between the two sides. Genuine cross-language pairs
    share ~0 tokens; copied credit/header/name lines share many. Used to drop noise."""
    a = set(norm_key(strip_annotations(nasa)).split())
    b = set(norm_key(spanish).split())
    if not a or not b:
        return 1.0
    return len(a & b) / len(a | b)


def clean_text(text: str) -> str:
    """Display-clean a field: NFC normalize + collapse whitespace (keeps punctuation)."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def heuristic_level(nasa: str, spanish: str) -> str:
    """A row is 'vocabulary' iff both sides are short term-like glosses."""
    n, s = strip_annotations(nasa), spanish
    if len(n) <= 80 and len(s) <= 80 and len(s.split()) <= 5 and len(n.split()) <= 6:
        # sentence punctuation strongly implies a full sentence
        if not re.search(r"[.!?;:]", s.strip().rstrip(".")):
            return "vocabulary"
    return "sentence"


# --------------------------------------------------------------------------- #
# Source parsers -> list of {nasa_yuwe, spanish, source, domain}
# --------------------------------------------------------------------------- #


def parse_corpus(path: Path, source: str, domain: str) -> list[dict]:
    """Parse a RASA ``corpus``-style file: ``- nasa`` line then ``# spanish`` line."""
    rows: list[dict] = []
    if not path.exists():
        return rows
    lines = path.read_text(encoding="utf-8").splitlines()
    pending_nasa: str | None = None
    for raw in lines:
        line = raw.strip()
        if not line:
            pending_nasa = None
            continue
        if line.startswith("#"):
            spanish = clean_text(line.lstrip("#").strip())
            if pending_nasa and spanish:
                nasa = clean_text(strip_annotations(pending_nasa))
                if nasa:
                    rows.append({"nasa_yuwe": nasa, "spanish": spanish,
                                 "source": source, "domain": domain})
            pending_nasa = None
        elif line.startswith("-"):
            pending_nasa = line[1:].strip()
        else:
            # continuation of a wrapped spanish/nasa line; ignore for safety
            pending_nasa = None
    return rows


def parse_csv(path: Path, source: str, domain: str) -> list[dict]:
    """Parse ``nasa_full_dataset.csv`` (columns esp,nas)."""
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            esp = clean_text(r.get("esp") or "")
            nas = clean_text(r.get("nas") or "")
            if esp and nas:
                rows.append({"nasa_yuwe": nas, "spanish": esp,
                             "source": source, "domain": domain})
    return rows


def gather_candidates() -> list[dict]:
    cands: list[dict] = []
    cands += parse_corpus(EXT_ROOT / "darcygith_nasaYuwe" / "corpus.txt",
                          "darcygith_rasa", "governance")
    cands += parse_corpus(EXT_ROOT / "darcygith_nasaYuwe" / "corpus_cultural",
                          "darcygith_cultural", "cultural")
    cands += parse_corpus(EXT_ROOT / "darcysem_nasaYuwe" / "corpus.txt",
                          "darcysem_rasa", "governance")
    cands += parse_corpus(EXT_ROOT / "darcysem_nasaYuwe" / "corpus_cultural",
                          "darcysem_cultural", "cultural")
    cands += parse_csv(EXT_ROOT / "juanks235_MT" / "datasets" / "nasa_full_dataset.csv",
                       "juanks235_nasa_full", "mixed")
    return cands


# --------------------------------------------------------------------------- #


def load_existing() -> tuple[list[dict], set[str], set[str], set[str], int]:
    rows = [json.loads(l) for l in SRC_JSONL.open(encoding="utf-8")]
    pair_keys = {pair_key(r["nasa_yuwe"], r["spanish"]) for r in rows}
    nasa_keys = {norm_key(strip_annotations(r["nasa_yuwe"])) for r in rows}
    es_keys = {norm_key(r["spanish"]) for r in rows}
    return rows, pair_keys, nasa_keys, es_keys, len(rows)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="append new rows to the source set and enqueue them for MT")
    args = ap.parse_args(argv)

    if not SRC_JSONL.exists():
        sys.exit(f"[ingest] source not found: {SRC_JSONL}")

    _, pair_keys, nasa_keys, es_keys, n_existing = load_existing()
    cands = gather_candidates()

    print(f"[ingest] existing source rows : {n_existing}")
    print(f"[ingest] parsed candidates    : {len(cands)}")
    print("           by source: " + json.dumps(dict(Counter(c["source"] for c in cands))))

    # Dedup: against existing AND against earlier candidates in this run.
    new_rows: list[dict] = []
    seen_run: set[str] = set()
    dup_existing = 0
    dup_intra = 0
    identical_sides = 0
    high_overlap = 0
    nasa_only_overlap = 0
    es_only_overlap = 0
    for c in cands:
        nk = norm_key(strip_annotations(c["nasa_yuwe"]))
        ek = norm_key(c["spanish"])
        # Drop untranslated noise rows where both sides are identical (credits,
        # proper-noun headers, dates) — these teach the model to copy, not translate.
        if nk and nk == ek:
            identical_sides += 1
            continue
        # Drop near-copied credit/name lines: genuine Nasa↔Spanish pairs share ~0
        # tokens across languages; high overlap means the "translation" is a copy.
        if token_overlap(c["nasa_yuwe"], c["spanish"]) >= 0.5:
            high_overlap += 1
            continue
        k = pair_key(c["nasa_yuwe"], c["spanish"])
        if k in pair_keys:
            dup_existing += 1
            continue
        if k in seen_run:
            dup_intra += 1
            continue
        seen_run.add(k)
        if nk in nasa_keys:
            nasa_only_overlap += 1
        if ek in es_keys:
            es_only_overlap += 1
        new_rows.append(c)

    by_source_new = Counter(r["source"] for r in new_rows)
    levels = Counter(heuristic_level(r["nasa_yuwe"], r["spanish"]) for r in new_rows)

    print(f"\n[ingest] duplicates vs existing : {dup_existing}")
    print(f"[ingest] duplicates intra-run   : {dup_intra}")
    print(f"[ingest] dropped identical-side : {identical_sides}")
    print(f"[ingest] dropped high-overlap   : {high_overlap}")
    print(f"[ingest] NEW unique pairs       : {len(new_rows)}")
    print("           by source: " + json.dumps(dict(by_source_new)))
    print("           by level : " + json.dumps(dict(levels)))
    print(f"[ingest] (of new) nasa-side already seen elsewhere : {nasa_only_overlap}")
    print(f"[ingest] (of new) spanish-side already seen elsewhere: {es_only_overlap}")

    INTERIM.mkdir(parents=True, exist_ok=True)
    preview = INTERIM / "external_new_preview.jsonl"
    with preview.open("w", encoding="utf-8") as f:
        for i, r in enumerate(new_rows):
            f.write(json.dumps({
                "nasa_id": n_existing + i,
                "nasa_yuwe": r["nasa_yuwe"], "spanish": r["spanish"],
                "source": r["source"], "domain": r["domain"],
                "level": heuristic_level(r["nasa_yuwe"], r["spanish"]),
            }, ensure_ascii=False) + "\n")
    print(f"\n[ingest] preview written -> {preview}")

    if not args.write:
        print("\n[ingest] DRY RUN (no --write): source set and MT queue untouched.")
        return 0

    if not new_rows:
        print("\n[ingest] nothing new to add; source set and queue unchanged.")
        return 0

    # 1) Append to the source JSONL (idempotent: dedup already excluded existing).
    appended = []
    with SRC_JSONL.open("a", encoding="utf-8") as f:
        for i, r in enumerate(new_rows):
            nid = n_existing + i
            level = heuristic_level(r["nasa_yuwe"], r["spanish"])
            row = {"nasa_yuwe": r["nasa_yuwe"], "spanish": r["spanish"],
                   "source": r["source"], "domain": r["domain"],
                   "meta": {"ingested": "external_repos", "level": level}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            appended.append((nid, r, level))
    print(f"[ingest] appended {len(appended)} rows -> {SRC_JSONL} "
          f"(nasa_id {appended[0][0]}..{appended[-1][0]})")

    # 2) Enqueue for Gemini under a dedicated job dir.
    items = [{
        "id": nid, "spanish": r["spanish"], "gloss": None,
        "level": level, "source": r["source"], "domain": r["domain"],
    } for (nid, r, level) in appended]
    paths = translate.enqueue(items, job=JOB, batch_size=200)
    print(f"[ingest] enqueued {len(items)} items -> {len(paths)} batch(es) in job '{JOB}'")
    print(f"[ingest] status: {json.dumps(translate.status(JOB))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
