"""Tests for snmt.runs — scheduler glue. Requires nymt_shared (installed editable)."""

from __future__ import annotations

from snmt import runs as R


def test_run_specs_parses_yaml_with_per_run_epochs():
    specs = R.run_specs()
    by_id = {s.run_id: s for s in specs}
    assert set(by_id) == {
        "nllb-600m-es-nasa",
        "nllb-1.3b-es-nasa",
        "nllb-3.3b-es-nasa",
    }
    assert by_id["nllb-600m-es-nasa"].model_key == "nllb-600m"
    assert by_id["nllb-600m-es-nasa"].epochs == 3
    # 3.3B is capped at 2 epochs to stay in budget
    assert by_id["nllb-3.3b-es-nasa"].epochs == 2


def test_load_nllb_runs_builds_schedule_runs():
    runs = R.load_nllb_runs(train_pairs=10000)
    assert len(runs) == 3
    keys = {r.model_key for r in runs}
    assert keys == {"nllb-600m", "nllb-1.3b", "nllb-3.3b"}
    for r in runs:
        r.compute()  # vram + duration resolve without raising
        assert r.vram_gb > 0
        assert r.duration_s > 0
        assert r.pairs == 10000


def test_load_nllb_runs_empty_when_no_pairs_is_still_valid():
    runs = R.load_nllb_runs(train_pairs=1)
    assert all(r.pairs == 1 for r in runs)


def test_launch_argv_targets_train_script_with_flags():
    spec = R.run_specs()[0]
    argv = R.launch_argv(spec, splits_dir="/tmp/splits", output_root="/tmp/out", python="python")
    assert argv[0] == "python"
    assert str(R.TRAIN_SCRIPT) in argv[1]
    assert "--run-id" in argv and spec.run_id in argv
    assert "--splits-dir" in argv and "/tmp/splits" in argv
    assert "--output-root" in argv and "/tmp/out" in argv
    assert "--yes" in argv


def test_launch_map_keys_by_run_id():
    specs = R.run_specs()
    m = R.launch_map(specs, splits_dir="/s", output_root="/o")
    assert set(m) == {s.run_id for s in specs}
    for rid, argv in m.items():
        assert rid in argv
