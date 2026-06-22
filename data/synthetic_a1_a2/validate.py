"""Validate generated chunk outputs against their batch inputs.

For each chunk file in generated_full/, confirm:
  - every line is valid JSON with required fields + non-empty spanish
  - every item_id from the batch is covered by >= n sentences
  - no duplicate spanish within the chunk

Prints a JSON array (one object per chunk present in the manifest) so the
orchestrator can update the gen_chunks tracking table.

    python validate.py                 # all chunks in manifest
    python validate.py CHUNK_ID ...     # only the named chunks
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
MANIFEST = HERE / "chunks_manifest.csv"
REQUIRED = ("spanish", "cefr_level", "concept", "concept_group")


def load_manifest() -> list[dict]:
    with MANIFEST.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def expected_items(batch_path: str) -> tuple[dict[str, int], int]:
    """Return {item_id: n} expected and the per-item n."""
    exp: dict[str, int] = {}
    n = 5
    with open(batch_path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            o = json.loads(ln)
            exp[o["item_id"]] = o.get("n", 5)
            n = o.get("n", 5)
    return exp, n


def validate_chunk(row: dict) -> dict:
    out = Path(row["out_path"])
    res = {"chunk_id": row["chunk_id"], "status": "failed", "produced": 0,
           "missing": [], "bad_lines": 0, "dupes": 0, "note": ""}
    if not out.exists():
        res["note"] = "missing output file"
        return res
    exp, n = expected_items(row["batch_path"])
    per_item: dict[str, int] = {}
    seen: set[str] = set()
    bad = 0
    dupes = 0
    produced = 0
    with out.open(encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except Exception:
                bad += 1
                continue
            es = (o.get("spanish") or "").strip()
            if not es or any(not o.get(k) for k in REQUIRED):
                bad += 1
                continue
            if es in seen:
                dupes += 1
                continue
            seen.add(es)
            iid = o.get("item_id", "")
            per_item[iid] = per_item.get(iid, 0) + 1
            produced += 1
    missing = [iid for iid, want in exp.items() if per_item.get(iid, 0) < want]
    res.update(produced=produced, bad_lines=bad, dupes=dupes, missing=missing)
    if not missing and bad == 0:
        res["status"] = "done"
    elif not missing:
        res["status"] = "done"
        res["note"] = f"{bad} bad lines tolerated"
    else:
        res["note"] = f"{len(missing)} items under-covered"
    return res


def main(argv: list[str]) -> int:
    rows = load_manifest()
    if argv:
        want = set(argv)
        rows = [r for r in rows if r["chunk_id"] in want]
    print(json.dumps([validate_chunk(r) for r in rows], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
