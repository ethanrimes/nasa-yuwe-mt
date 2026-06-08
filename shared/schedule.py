"""GPU-packing scheduler + cost model for the ablation matrix.

Pure logic (no torch, no az, no GPU) so it is unit-testable and drives the
dry-run cost estimate. Given a set of training runs — each with a model size and
a dataset row count — it:

  1. estimates each run's VRAM footprint and wall-clock duration on one H100, and
  2. greedily packs runs into *waves* that fit concurrently in usable VRAM, then
  3. reports total GPU-hours (= sum of per-wave max durations) and $ cost.

The packing assumes a single H100 (80 GB, ~72 GB usable) running several training
processes at once via `CUDA_VISIBLE_DEVICES=0` + per-process memory caps. This is
what lets us finish the whole matrix in a couple of billed GPU-hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config

# Conservative bf16 + grad-checkpointing throughput on one H100, in *training
# examples per second* (one example = one direction of one pair). Deliberately
# pessimistic so the $ estimate is an upper bound rather than a surprise.
THROUGHPUT_EX_PER_S = {
    "360m": 38.0,
    "1.7b": 11.0,
    # NLLB seq2seq fine-tune throughput on one H100 (bf16 + grad-checkpointing),
    # training examples/s (one example = one direction of one pair). Conservative.
    "nllb-600m": 55.0,
    "nllb-1.3b": 28.0,
    "nllb-3.3b": 11.0,
}

# Fixed per-run overhead (model load, tokenize, eval passes, checkpoint writes).
FIXED_OVERHEAD_S = {
    "360m": 240.0,
    "1.7b": 480.0,
    "nllb-600m": 180.0,
    "nllb-1.3b": 300.0,
    "nllb-3.3b": 600.0,
}


@dataclass
class Run:
    run_id: str
    model_key: str          # "360m" | "1.7b"
    subset: str             # "sentence_only" | "sentence_plus_vocab" | ...
    pairs: int              # parallel rows; bidirectional => 2x training examples
    epochs: float = 3.0
    effective_batch: int = 64
    # derived:
    vram_gb: float = 0.0
    duration_s: float = 0.0

    def compute(self) -> "Run":
        spec = config.MODELS[self.model_key]
        self.vram_gb = spec.approx_vram_gb_train
        examples = 2 * self.pairs * self.epochs
        tput = THROUGHPUT_EX_PER_S[self.model_key]
        self.duration_s = FIXED_OVERHEAD_S[self.model_key] + examples / tput
        return self


@dataclass
class Wave:
    runs: list[Run] = field(default_factory=list)

    @property
    def vram_gb(self) -> float:
        return sum(r.vram_gb for r in self.runs)

    @property
    def duration_s(self) -> float:
        return max((r.duration_s for r in self.runs), default=0.0)


def pack(
    runs: list[Run],
    usable_vram_gb: float = config.H100_USABLE_VRAM_GB,
    max_concurrent: int = 6,
) -> list[Wave]:
    """Greedy first-fit-decreasing packing into VRAM-bounded waves.

    Runs are sorted by descending VRAM so the big 1.7B jobs seat first; smaller
    360M jobs then fill the remaining headroom in the same wave.
    """
    for r in runs:
        r.compute()
    ordered = sorted(runs, key=lambda r: r.vram_gb, reverse=True)
    waves: list[Wave] = []
    for r in ordered:
        placed = False
        for w in waves:
            if (
                w.vram_gb + r.vram_gb <= usable_vram_gb
                and len(w.runs) < max_concurrent
            ):
                w.runs.append(r)
                placed = True
                break
        if not placed:
            waves.append(Wave([r]))
    return waves


@dataclass
class Estimate:
    waves: list[Wave]
    gpu_hours: float
    serial_gpu_hours: float
    cost_usd: float
    rate_usd_h: float

    def to_dict(self) -> dict:
        return {
            "n_runs": sum(len(w.runs) for w in self.waves),
            "n_waves": len(self.waves),
            "gpu_hours_packed": round(self.gpu_hours, 3),
            "gpu_hours_serial": round(self.serial_gpu_hours, 3),
            "rate_usd_per_hour": self.rate_usd_h,
            "cost_usd_packed": round(self.cost_usd, 2),
            "cost_usd_serial": round(self.serial_gpu_hours * self.rate_usd_h, 2),
            "waves": [
                {
                    "vram_gb": round(w.vram_gb, 1),
                    "duration_min": round(w.duration_s / 60.0, 1),
                    "runs": [
                        {
                            "run_id": r.run_id,
                            "model": r.model_key,
                            "subset": r.subset,
                            "pairs": r.pairs,
                            "vram_gb": r.vram_gb,
                            "duration_min": round(r.duration_s / 60.0, 1),
                        }
                        for r in w.runs
                    ],
                }
                for w in self.waves
            ],
        }


def estimate(runs: list[Run], spot: bool | None = None) -> Estimate:
    spot = config.USE_SPOT if spot is None else spot
    rate = (
        config.VM_COST_PER_HOUR_SPOT if spot else config.VM_COST_PER_HOUR_ONDEMAND
    )
    waves = pack(runs)
    gpu_seconds = sum(w.duration_s for w in waves)
    serial_seconds = sum(r.duration_s for r in runs)
    gpu_hours = gpu_seconds / 3600.0
    return Estimate(
        waves=waves,
        gpu_hours=gpu_hours,
        serial_gpu_hours=serial_seconds / 3600.0,
        cost_usd=gpu_hours * rate,
        rate_usd_h=rate,
    )
