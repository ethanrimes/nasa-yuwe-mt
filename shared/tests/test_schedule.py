"""Offline unit tests for the GPU-packing scheduler + cost model (no GPU/network)."""

from __future__ import annotations

from nymt_shared import config, schedule


def _core_runs(s: int = 15891, sv: int = 23674) -> list[schedule.Run]:
    return [
        schedule.Run("1p7b-sentence", "1.7b", "sentence_only", s),
        schedule.Run("1p7b-sentvocab", "1.7b", "sentence_plus_vocab", sv),
        schedule.Run("360m-sentence", "360m", "sentence_only", s),
        schedule.Run("360m-sentvocab", "360m", "sentence_plus_vocab", sv),
    ]


def test_run_compute_derives_vram_and_duration():
    r = schedule.Run("r", "1.7b", "sentence_only", 1000).compute()
    assert r.vram_gb == config.MODELS["1.7b"].approx_vram_gb_train
    # bidirectional => 2x examples; duration strictly positive and > fixed overhead
    assert r.duration_s > schedule.FIXED_OVERHEAD_S["1.7b"]


def test_pack_respects_vram_budget():
    waves = schedule.pack(_core_runs(), usable_vram_gb=72.0, max_concurrent=6)
    for w in waves:
        assert w.vram_gb <= 72.0
    # two 1.7B (34 each = 68) cannot co-seat with a 360M (10) -> would be 78 > 72
    assert len(waves) == 2


def test_pack_first_fit_decreasing_seats_big_jobs_first():
    waves = schedule.pack(_core_runs())
    # wave 0 holds the two 1.7B jobs (largest VRAM seats first)
    assert {r.model_key for r in waves[0].runs} == {"1.7b"}
    assert {r.model_key for r in waves[1].runs} == {"360m"}


def test_estimate_packed_cheaper_than_serial():
    est = schedule.estimate(_core_runs(), spot=True)
    assert est.gpu_hours < est.serial_gpu_hours
    assert est.cost_usd < est.serial_gpu_hours * est.rate_usd_h + 1e-6
    assert est.rate_usd_h == config.VM_COST_PER_HOUR_SPOT
    d = est.to_dict()
    assert d["n_runs"] == 4 and d["n_waves"] == 2


def test_estimate_spot_cheaper_than_ondemand():
    spot = schedule.estimate(_core_runs(), spot=True)
    od = schedule.estimate(_core_runs(), spot=False)
    assert spot.cost_usd < od.cost_usd
