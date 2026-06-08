"""Pure-logic tests for snmt.data — no torch, no GPU, no network."""

from __future__ import annotations

from snmt import data as D


def _rows():
    return [
        {"spanish": "hola", "nasa_yuwe": "aa", "source": "x", "domain": "d"},
        {"spanish": "hola", "nasa_yuwe": "aa", "source": "x", "domain": "d"},  # dup
        {"spanish": "  adios ", "nasa_yuwe": "bb", "source": "x", "domain": "d"},
        {"spanish": "", "nasa_yuwe": "cc"},          # invalid (empty src)
        {"spanish": "buenos dias", "nasa_yuwe": ""},  # invalid (empty tgt)
        {"spanish": "gracias", "nasa_yuwe": "dd"},
    ]


def test_dedup_drops_dupes_and_invalid():
    out = D.dedup(_rows())
    # dedup keeps original rows; normalize when comparing
    norm = {(" ".join(r["spanish"].split()), r["nasa_yuwe"]) for r in out}
    assert ("hola", "aa") in norm
    assert ("adios", "bb") in norm
    assert ("gracias", "dd") in norm
    assert len(out) == 3


def test_assign_splits_is_deterministic_and_partitions():
    rows = _rows()
    a = D.assign_splits(rows)
    b = D.assign_splits(rows)
    assert {k: [r["spanish"] for r in v] for k, v in a.items()} == {
        k: [r["spanish"] for r in v] for k, v in b.items()
    }
    total = sum(len(v) for v in a.values())
    assert total == len(D.dedup(rows))
    # no pair appears in two splits
    seen = set()
    for v in a.values():
        for r in v:
            key = (r["spanish"], r["nasa_yuwe"])
            assert key not in seen
            seen.add(key)


def test_bidirectional_expansion_yields_two_directions():
    rows = [{"spanish": "hola", "nasa_yuwe": "aa"}]
    ex = list(D.iter_bidirectional_examples(rows))
    assert len(ex) == 2
    dirs = {(e["src_lang"], e["tgt_lang"]) for e in ex}
    assert dirs == {("spa_Latn", "pbb_Latn"), ("pbb_Latn", "spa_Latn")}
    # texts are correctly routed per direction
    spa2pbb = next(e for e in ex if e["src_lang"] == "spa_Latn")
    assert spa2pbb["src_text"] == "hola" and spa2pbb["tgt_text"] == "aa"


def test_fractions_scale_dev_test_buckets():
    # 1000 buckets, dev=0.10 test=0.10 => roughly 20% held out across many rows
    rows = [{"spanish": f"s{i}", "nasa_yuwe": f"n{i}"} for i in range(2000)]
    cfg = D.SplitConfig(dev_fraction=0.10, test_fraction=0.10)
    sp = D.assign_splits(rows, cfg)
    held = len(sp["dev"]) + len(sp["test"])
    frac = held / len(rows)
    assert 0.15 < frac < 0.25
