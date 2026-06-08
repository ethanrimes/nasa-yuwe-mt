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

import os
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
    subset: str = "sentence_plus_vocab"   # which data subset dir this run trains on


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def run_specs(cfg: dict | None = None) -> list[NllbRunSpec]:
    """Parse the nllb.yaml run list into structured specs.

    The optional ``SNMT_ONLY_RUNS`` environment variable subsets which runs are
    returned (and therefore which the scheduler packs and the runner launches).
    It is a comma-separated list of case-insensitive substrings; a run is kept if
    ANY token is a substring of its ``run_id``. This lets us re-run just a subset
    (e.g. ``SNMT_ONLY_RUNS=600m,1.3b`` to redo the small models without touching
    the already-finished 3.3B runs) without editing the committed config. Unset or
    empty => no filtering.
    """
    cfg = cfg or load_config()
    default_epochs = float(cfg.get("epochs", 3))
    default_subset = str(cfg.get("subset", "sentence_plus_vocab"))
    specs: list[NllbRunSpec] = []
    for r in cfg.get("runs", []):
        specs.append(
            NllbRunSpec(
                run_id=r["id"],
                model_key=r["model_key"],
                model_config=r["model_config"],
                epochs=float(r.get("epochs", default_epochs)),
                subset=str(r.get("subset", default_subset)),
            )
        )
    tokens = [t.strip().lower() for t in os.environ.get("SNMT_ONLY_RUNS", "").split(",")]
    tokens = [t for t in tokens if t]
    if tokens:
        specs = [s for s in specs if any(tok in s.run_id.lower() for tok in tokens)]
    return specs


def load_nllb_runs(
    train_pairs: int | dict[str, int],
    cfg: dict | None = None,
) -> list[schedule.Run]:
    """Build ``schedule.Run`` objects (one per NLLB run) for VRAM packing.

    ``train_pairs`` is the number of *parallel pairs* in the train split (the
    scheduler internally doubles it for the bidirectional example count). It may be:
      * an ``int`` — the same pair count for every run (legacy / single-subset), or
      * a ``dict`` mapping subset name -> train-pair count, so the ``sentence_only``
        runs are costed on the smaller split and ``sentence_plus_vocab`` on the full
        corpus.
    Each run carries its own ``epochs`` so the big 3.3B can be capped lower to stay
    inside the GPU budget, and its own ``subset`` so the scheduler/labels track it.
    """
    cfg = cfg or load_config()
    eff_batch = int(cfg.get("effective_batch", 64))

    def pairs_for(subset: str) -> int:
        if isinstance(train_pairs, dict):
            if subset not in train_pairs:
                raise KeyError(
                    f"train_pairs dict missing subset '{subset}'; have {sorted(train_pairs)}"
                )
            return int(train_pairs[subset])
        return int(train_pairs)

    runs: list[schedule.Run] = []
    for s in run_specs(cfg):
        runs.append(
            schedule.Run(
                run_id=s.run_id,
                model_key=s.model_key,
                subset=s.subset,
                pairs=pairs_for(s.subset),
                epochs=s.epochs,
                effective_batch=eff_batch,
            )
        )
    return runs


def launch_argv(
    spec: NllbRunSpec,
    *,
    splits_root: str | Path,
    output_root: str | Path,
    python: str | None = None,
) -> list[str]:
    """Build the argv to launch one NLLB run via ``scripts/train_nllb.py``.

    ``splits_root`` is the PARENT dir holding per-subset split dirs; the run's
    ``--splits-dir`` is resolved to ``<splits_root>/<spec.subset>``.
    """
    py = python or sys.executable
    splits_dir = Path(splits_root) / spec.subset
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
    splits_root: str | Path,
    output_root: str | Path,
    python: str | None = None,
) -> dict[str, list[str]]:
    """run_id -> launch argv, for the runner's dispatch table."""
    return {
        s.run_id: launch_argv(
            s, splits_root=splits_root, output_root=output_root, python=python
        )
        for s in specs
    }
