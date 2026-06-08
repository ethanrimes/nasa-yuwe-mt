"""Glue between the NLLB fine-tunes and the shared GPU-packing scheduler.

The En<->Zh on-VM runner (``run_matrix_on_vm.py``) calls into here to (a) turn the
``configs/nllb.yaml`` run list into ``schedule.Run`` objects it can pack alongside
the En<->Zh runs, and (b) get the exact CLI to launch each NLLB run. Keeping this
mapping in ``snmt`` (not the en-zh runner) means the runner stays agnostic and the
whole NLLB feature is opt-in / robust-if-absent.

Pure orchestration logic — no torch. The only third-party import is ``nymt_shared``
(for ``schedule.Run``), which is always installed wherever the runner runs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from nymt_shared import schedule

PKG_ROOT = Path(__file__).resolve().parents[2]  # spanish-nasa-mt-experiment/
DEFAULT_CONFIG = PKG_ROOT / "configs" / "nllb.yaml"
TRAIN_SCRIPT = PKG_ROOT / "scripts" / "train_nllb.py"


@dataclass(frozen=True)
class NllbRunSpec:
    run_id: str
    model_key: str          # "nllb-600m" | "nllb-1.3b" | "nllb-3.3b"
    model_config: str       # path to a configs/model_*.yaml (relative to pkg root ok)
    epochs: float


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def run_specs(cfg: dict | None = None) -> list[NllbRunSpec]:
    """Parse the nllb.yaml run list into structured specs."""
    cfg = cfg or load_config()
    default_epochs = float(cfg.get("epochs", 3))
    specs: list[NllbRunSpec] = []
    for r in cfg.get("runs", []):
        specs.append(
            NllbRunSpec(
                run_id=r["id"],
                model_key=r["model_key"],
                model_config=r["model_config"],
                epochs=float(r.get("epochs", default_epochs)),
            )
        )
    return specs


def load_nllb_runs(train_pairs: int, cfg: dict | None = None) -> list[schedule.Run]:
    """Build ``schedule.Run`` objects (one per NLLB run) for VRAM packing.

    ``train_pairs`` is the number of *parallel pairs* in the train split (the
    scheduler internally doubles it for the bidirectional example count). Each run
    can carry its own ``epochs`` so the big 3.3B can be capped lower to stay inside
    the GPU budget.
    """
    cfg = cfg or load_config()
    eff_batch = int(cfg.get("effective_batch", 64))
    runs: list[schedule.Run] = []
    for s in run_specs(cfg):
        runs.append(
            schedule.Run(
                run_id=s.run_id,
                model_key=s.model_key,
                subset="es_nasa_real",
                pairs=int(train_pairs),
                epochs=s.epochs,
                effective_batch=eff_batch,
            )
        )
    return runs


def launch_argv(
    spec: NllbRunSpec,
    *,
    splits_dir: str | Path,
    output_root: str | Path,
    python: str | None = None,
) -> list[str]:
    """Build the argv to launch one NLLB run via ``scripts/train_nllb.py``."""
    py = python or sys.executable
    return [
        py, str(TRAIN_SCRIPT),
        "--run-id", spec.run_id,
        "--model-config", spec.model_config,
        "--epochs", str(spec.epochs),
        "--splits-dir", str(splits_dir),
        "--output-root", str(output_root),
        "--yes",
    ]


def launch_map(
    specs: list[NllbRunSpec],
    *,
    splits_dir: str | Path,
    output_root: str | Path,
    python: str | None = None,
) -> dict[str, list[str]]:
    """run_id -> launch argv, for the runner's dispatch table."""
    return {
        s.run_id: launch_argv(s, splits_dir=splits_dir, output_root=output_root, python=python)
        for s in specs
    }
