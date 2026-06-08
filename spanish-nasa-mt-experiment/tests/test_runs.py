"""Tests for snmt.runs — scheduler glue. Requires nymt_shared (installed editable)."""

from __future__ import annotations

import pytest

from snmt import runs as R

ALL_RUN_IDS = {
    "nllb-600m-es-nasa-sent",
    "nllb-1.3b-es-nasa-sent",
    "nllb-3.3b-es-nasa-sent",
    "nllb-600m-es-nasa-sentvocab",
    "nllb-1.3b-es-nasa-sentvocab",
    "nllb-3.3b-es-nasa-sentvocab",
}


def test_run_specs_parses_yaml_with_per_run_epochs_and_subset():
    specs = R.run_specs()
    by_id = {s.run_id: s for s in specs}
    assert set(by_id) == ALL_RUN_IDS
    assert by_id["nllb-600m-es-nasa-sent"].model_key == "nllb-600m"
    assert by_id["nllb-600m-es-nasa-sent"].epochs == 3
    # 3.3B is capped at 2 epochs to stay in budget (both subsets)
    assert by_id["nllb-3.3b-es-nasa-sent"].epochs == 2
    assert by_id["nllb-3.3b-es-nasa-sentvocab"].epochs == 2
    # Subset axis wired through
    assert by_id["nllb-600m-es-nasa-sent"].subset == "sentence_only"
    assert by_id["nllb-600m-es-nasa-sentvocab"].subset == "sentence_plus_vocab"


def test_snmt_only_runs_filters_specs(monkeypatch):
    # Re-run just the small models: tokens match a substring of the run id.
    monkeypatch.setenv("SNMT_ONLY_RUNS", "600m,1.3b")
    ids = {s.run_id for s in R.run_specs()}
    assert ids == {
        "nllb-600m-es-nasa-sent",
        "nllb-1.3b-es-nasa-sent",
        "nllb-600m-es-nasa-sentvocab",
        "nllb-1.3b-es-nasa-sentvocab",
    }
    # The filter propagates to the schedule.Run list the packer consumes.
    run_ids = {r.run_id for r in R.load_nllb_runs(train_pairs=1)}
    assert run_ids == ids
    assert "nllb-3.3b-es-nasa-sent" not in run_ids


def test_snmt_only_runs_blank_or_unset_keeps_all(monkeypatch):
    monkeypatch.setenv("SNMT_ONLY_RUNS", "  ,  ")
    assert {s.run_id for s in R.run_specs()} == ALL_RUN_IDS
    monkeypatch.delenv("SNMT_ONLY_RUNS", raising=False)
    assert {s.run_id for s in R.run_specs()} == ALL_RUN_IDS


def test_load_nllb_runs_builds_schedule_runs_int_pairs():
    runs = R.load_nllb_runs(train_pairs=10000)
    assert len(runs) == 6
    keys = {r.model_key for r in runs}
    assert keys == {"nllb-600m", "nllb-1.3b", "nllb-3.3b"}
    for r in runs:
        r.compute()  # vram + duration resolve without raising
        assert r.vram_gb > 0
        assert r.duration_s > 0
        assert r.pairs == 10000


def test_load_nllb_runs_per_subset_pairs():
    runs = R.load_nllb_runs({"sentence_only": 15000, "sentence_plus_vocab": 22000})
    by_id = {r.run_id: r for r in runs}
    assert by_id["nllb-600m-es-nasa-sent"].pairs == 15000
    assert by_id["nllb-600m-es-nasa-sentvocab"].pairs == 22000
    # sentence_only is the smaller split for every size
    for size in ("600m", "1.3b", "3.3b"):
        assert (
            by_id[f"nllb-{size}-es-nasa-sent"].pairs
            < by_id[f"nllb-{size}-es-nasa-sentvocab"].pairs
        )


def test_load_nllb_runs_missing_subset_key_raises():
    with pytest.raises(KeyError):
        R.load_nllb_runs({"sentence_only": 100})  # sentence_plus_vocab absent


def test_load_nllb_runs_subset_carried_into_schedule_run():
    runs = R.load_nllb_runs(train_pairs=1)
    by_id = {r.run_id: r for r in runs}
    assert by_id["nllb-1.3b-es-nasa-sent"].subset == "sentence_only"
    assert by_id["nllb-1.3b-es-nasa-sentvocab"].subset == "sentence_plus_vocab"


def test_launch_argv_resolves_splits_dir_per_subset():
    spec = next(s for s in R.run_specs() if s.subset == "sentence_only")
    argv = R.launch_argv(spec, splits_root="/tmp/splits", output_root="/tmp/out", python="python")
    assert argv[0] == "python"
    assert str(R.TRAIN_SCRIPT) in argv[1]
    assert "--run-id" in argv and spec.run_id in argv
    i = argv.index("--splits-dir")
    assert "sentence_only" in argv[i + 1].replace("\\", "/")
    assert "--output-root" in argv and "/tmp/out" in argv
    assert "--yes" in argv


def test_launch_map_keys_by_run_id_and_routes_subset_dirs():
    specs = R.run_specs()
    m = R.launch_map(specs, splits_root="/s", output_root="/o")
    assert set(m) == {s.run_id for s in specs}
    by_id = {s.run_id: s for s in specs}
    for rid, argv in m.items():
        assert rid in argv
        i = argv.index("--splits-dir")
        assert by_id[rid].subset in argv[i + 1].replace("\\", "/")
