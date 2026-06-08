#!/usr/bin/env python3
"""Launch one NLLB Spanish<->Nasa-Yuwe fine-tune (GPU).

Invoked per-run by the on-VM matrix runner (one process per packed run) or by hand
for a single model. Reads a ``configs/model_*.yaml`` for the model id + training
hyperparameters, optionally overrides the epoch count (the runner caps the 3.3B),
and calls ``snmt.train.build_and_train``.

Outputs land under ``<output-root>/<run-id>/`` (checkpoints / final / summary.json),
which the runner's background loop mirrors to Blob ``experiments/es-nasa/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[1]          # spanish-nasa-mt-experiment/
sys.path.insert(0, str(PKG / "src"))

from snmt import train as snmt_train  # noqa: E402


def _resolve_cfg_path(p: str) -> Path:
    cand = Path(p)
    if not cand.is_absolute():
        # Allow paths relative to the package root (e.g. "configs/model_600m.yaml").
        cand = PKG / p
    if not cand.exists():
        raise FileNotFoundError(cand)
    return cand


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model-config", required=True, help="path to a configs/model_*.yaml")
    ap.add_argument("--splits-dir", default=str(PKG / "data" / "splits"))
    ap.add_argument("--output-root", default=str(PKG / "models" / "runs"))
    ap.add_argument("--epochs", type=float, default=None, help="override num_train_epochs")
    ap.add_argument("--yes", action="store_true", help="run without interactive confirm")
    args = ap.parse_args()

    model_cfg = yaml.safe_load(_resolve_cfg_path(args.model_config).read_text(encoding="utf-8"))
    training_cfg = dict(model_cfg.get("training", {}))
    if args.epochs is not None:
        training_cfg["num_train_epochs"] = float(args.epochs)

    splits_dir = Path(args.splits_dir)
    if not (splits_dir / "train.parquet").exists():
        raise SystemExit(
            f"splits not found under {splits_dir}; run scripts/prepare_data.py first"
        )

    if not args.yes:
        try:
            if input(f"train {model_cfg['hf_id']} run={args.run_id}? type 'GO': ").strip() != "GO":
                print("aborted.")
                return 130
        except EOFError:
            print("non-interactive; pass --yes to proceed.")
            return 130

    summary = snmt_train.build_and_train(
        run_id=args.run_id,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        splits_dir=splits_dir,
        output_root=args.output_root,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
