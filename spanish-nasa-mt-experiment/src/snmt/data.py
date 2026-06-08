"""Build train / dev / test splits from the Spanish<->Nasa-Yuwe parallel corpus.

The source is ``nasa_yuwe_parallel_dataset.jsonl`` (24k rows of
``{"nasa_yuwe", "spanish", "source", "domain", "meta"}``). On the H100 this file
is **not** rsynced (it lives under ``data-en-zh/source`` which is excluded), so
``scripts/prepare_data.py`` pulls it from Blob; locally it is read straight off
disk. Either way the rows are handed to :func:`build_splits` here.

Design choices:
  - **Deterministic, content-hashed split** so train/dev/test are stable across
    machines and reruns (no reliance on dict/file ordering or a shuffled seed that
    could drift between numpy versions).
  - **Dedup on the (spanish, nasa_yuwe) pair** to avoid leaking identical pairs
    across splits.
  - Splits are stored as parquet (one row per *pair*); bidirectional expansion to
    two directed training examples happens at train time via
    :func:`iter_bidirectional_examples` so the parquet stays small and direction-
    agnostic.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .lang import DIRECTIONS

# Canonical field names in the parallel jsonl.
SPANISH_FIELD = "spanish"
NASA_FIELD = "nasa_yuwe"


@dataclass(frozen=True)
class SplitConfig:
    dev_fraction: float = 0.03
    test_fraction: float = 0.03
    # Bucketed deterministic split uses this many buckets; dev/test fractions are
    # rounded to whole buckets.
    n_buckets: int = 1000
    seed: str = "snmt-v1"


def _norm(s: str) -> str:
    return " ".join((s or "").split())


def _pair_key(row: dict) -> str:
    return _norm(row.get(SPANISH_FIELD, "")) + "\u241f" + _norm(row.get(NASA_FIELD, ""))


def _bucket(key: str, *, seed: str, n_buckets: int) -> int:
    h = hashlib.sha1(f"{seed}\u241f{key}".encode()).hexdigest()
    return int(h[:8], 16) % n_buckets


# Length (hex chars) of the content hash used to key the sentence/vocab level map.
# Content-keyed (NOT row-index keyed) so it is robust to any row reordering between
# the local source copy and the Blob copy downloaded on the VM.
LEVEL_HASH_LEN = 16


def level_hash(row: dict) -> str:
    """Stable short content hash of a row's (spanish, nasa_yuwe) pair.

    Used to look a row up in the committed ``sentence_keys.json`` level map so we can
    split the corpus into the ``sentence_only`` vs ``sentence_plus_vocab`` subsets
    exactly the way the En<->Zh ablation did — keyed on pair content, not file order.
    """
    return hashlib.sha1(_pair_key(row).encode()).hexdigest()[:LEVEL_HASH_LEN]


def filter_by_keys(rows: Iterable[dict], keep_keys: set[str]) -> list[dict]:
    """Keep only rows whose :func:`level_hash` is in ``keep_keys`` (order-stable)."""
    keep = set(keep_keys)
    return [r for r in rows if level_hash(r) in keep]


def _valid(row: dict) -> bool:
    return bool(_norm(row.get(SPANISH_FIELD, ""))) and bool(_norm(row.get(NASA_FIELD, "")))


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dedup(rows: Iterable[dict]) -> list[dict]:
    """Drop invalid rows and exact (spanish, nasa_yuwe) duplicates, order-stable."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        if not _valid(r):
            continue
        k = _pair_key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def assign_splits(rows: list[dict], cfg: SplitConfig | None = None) -> dict[str, list[dict]]:
    """Deterministically bucket deduped rows into train/dev/test.

    Pure (no I/O): given the same rows + config, always yields the same partition.
    """
    cfg = cfg or SplitConfig()
    deduped = dedup(rows)
    dev_cut = int(round(cfg.dev_fraction * cfg.n_buckets))
    test_cut = dev_cut + int(round(cfg.test_fraction * cfg.n_buckets))

    out: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for r in deduped:
        b = _bucket(_pair_key(r), seed=cfg.seed, n_buckets=cfg.n_buckets)
        if b < dev_cut:
            out["dev"].append(r)
        elif b < test_cut:
            out["test"].append(r)
        else:
            out["train"].append(r)
    return out


def iter_bidirectional_examples(rows: Iterable[dict]) -> Iterator[dict]:
    """Expand each pair into the two directed examples (spa->pbb and pbb->spa).

    Yields ``{src_text, tgt_text, src_lang, tgt_lang}`` dicts — direction-tagged but
    not yet tokenized. Pure; used both by the trainer and by tests.
    """
    for r in rows:
        for d in DIRECTIONS:
            src = _norm(r.get(d.src_field, ""))
            tgt = _norm(r.get(d.tgt_field, ""))
            if not src or not tgt:
                continue
            yield {
                "src_text": src,
                "tgt_text": tgt,
                "src_lang": d.src_lang,
                "tgt_lang": d.tgt_lang,
            }


def _write_parquet(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Keep only the columns the trainer needs; meta is dropped to keep splits lean.
    cols = {
        SPANISH_FIELD: [_norm(r.get(SPANISH_FIELD, "")) for r in rows],
        NASA_FIELD: [_norm(r.get(NASA_FIELD, "")) for r in rows],
        "source": [str(r.get("source", "")) for r in rows],
        "domain": [str(r.get("domain", "")) for r in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), path)


def build_splits(
    rows: list[dict],
    out_dir: str | Path,
    cfg: SplitConfig | None = None,
) -> dict:
    """Build parquet splits + a manifest under ``out_dir``. Returns the manifest dict."""
    cfg = cfg or SplitConfig()
    out_dir = Path(out_dir)
    splits = assign_splits(rows, cfg)

    files = {}
    for name, split_rows in splits.items():
        p = out_dir / f"{name}.parquet"
        _write_parquet(split_rows, p)
        files[name] = p.name

    manifest = {
        "n_input_rows": len(rows),
        "n_deduped": sum(len(v) for v in splits.values()),
        "counts": {k: len(v) for k, v in splits.items()},
        # bidirectional => training examples = 2 * train pairs
        "train_pairs": len(splits["train"]),
        "train_examples_bidir": 2 * len(splits["train"]),
        "files": files,
        "split_config": {
            "dev_fraction": cfg.dev_fraction,
            "test_fraction": cfg.test_fraction,
            "n_buckets": cfg.n_buckets,
            "seed": cfg.seed,
        },
        "fields": {"spanish": SPANISH_FIELD, "nasa_yuwe": NASA_FIELD},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splits_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def load_split_rows(parquet_path: str | Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(parquet_path).to_pylist()
