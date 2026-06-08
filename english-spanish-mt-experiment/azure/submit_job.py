"""Submit a training job per tier to Azure ML.

Reads `.env` for subscription / RG / workspace / compute / data-asset names,
and submits one command job per requested tier. Each job is auto-named
T{tier}-{git_sha[:7]}-{timestamp}.

Prereqs (one-time, see docs/AZURE_SETUP.md):
  1. `az login` so DefaultAzureCredential picks up your identity.
  2. AML workspace exists; compute target exists; environment registered.
  3. Data assets registered: en-es-parallel-processed and flores200-en-es.

Usage:
  uv run python azure/submit_job.py --tiers 10000 50000
  uv run python azure/submit_job.py --tiers 500000 --wait     # block until finished

  # 1.7B size-sweep on the H100 cluster (AIERP subscription, westus2):
  uv run python azure/submit_job.py --tiers 10000 50000 100000 500000 1000000 \
      --model-config configs/model_1.7b.yaml --model-name SmolLM2-1.7B \
      --compute gpu-h100-1x
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", nargs="+", type=int, required=True)
    parser.add_argument("--data-asset", default=None,
                        help="Override AZURE data asset name (default from .env)")
    parser.add_argument("--compute", default=None,
                        help="Override AZURE_COMPUTE_NAME from .env")
    parser.add_argument("--model-config", default="configs/model.yaml",
                        help="Model config passed to scripts/04_train.py. Use "
                             "configs/model_1.7b.yaml for the H100 size-sweep.")
    parser.add_argument("--model-name", default=None,
                        help="Friendly model name for run tags/description "
                             "(default inferred from the model config filename).")
    parser.add_argument("--experiment", default="en-es-mt-scaling")
    parser.add_argument("--wait", action="store_true",
                        help="Block until each job completes before returning.")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    sub = os.environ.get("AZURE_SUBSCRIPTION_ID")
    rg = os.environ.get("AZURE_RESOURCE_GROUP")
    ws = os.environ.get("AZURE_ML_WORKSPACE")
    compute = args.compute or os.environ.get("AZURE_COMPUTE_NAME", "gpu-a100-1x")
    data_asset = args.data_asset or os.environ.get("AZURE_DATA_PROCESSED", "en-es-parallel-processed")

    # Friendly model name for tags: explicit flag wins, else infer from the
    # config filename (model_1.7b.yaml -> SmolLM2-1.7b), else default.
    model_name = args.model_name
    if not model_name:
        stem = Path(args.model_config).stem  # e.g. "model_1.7b" or "model"
        suffix = stem.replace("model", "").strip("_-")
        model_name = f"SmolLM2-{suffix}" if suffix else "SmolLM2-360M"

    missing = [k for k, v in {
        "AZURE_SUBSCRIPTION_ID": sub, "AZURE_RESOURCE_GROUP": rg, "AZURE_ML_WORKSPACE": ws,
    }.items() if not v]
    if missing:
        print(f"missing required env vars: {missing} (set in .env)", file=sys.stderr)
        sys.exit(2)

    from azure.ai.ml import Input, MLClient, Output, command
    from azure.ai.ml.constants import AssetTypes, InputOutputModes
    from azure.identity import DefaultAzureCredential

    ml = MLClient(DefaultAzureCredential(), sub, rg, ws)

    git_sha = _git_sha()[:7] or "nogit"
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M")

    for tier in args.tiers:
        display = f"T{tier}-{model_name}-{git_sha}-{stamp}"
        print(f"# submitting {display} on compute={compute} model={args.model_config}")
        job = command(
            display_name=display,
            experiment_name=args.experiment,
            description=f"{model_name} EN-ES fine-tune, tier T{tier}.",
            code=str(Path(__file__).resolve().parents[1]),
            command=(
                "pip install -e . && "
                "python scripts/04_train.py "
                f"--tiers {tier} "
                f"--model-config {args.model_config} "
                "--data-dir ${{inputs.processed_data}} "
                "--output-root ${{outputs.checkpoints}}"
            ),
            inputs={
                "processed_data": Input(
                    type=AssetTypes.URI_FOLDER,
                    path=f"azureml:{data_asset}@latest",
                    mode=InputOutputModes.RO_MOUNT,
                ),
            },
            outputs={
                "checkpoints": Output(type=AssetTypes.URI_FOLDER, mode=InputOutputModes.RW_MOUNT),
            },
            environment="en-es-mt-env@latest",
            compute=compute,
            environment_variables={
                "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "en-es-mt"),
                "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", ""),
                "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
                "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                "HF_HOME": "/tmp/hf_cache",
                "MLFLOW_EXPERIMENT_NAME": args.experiment,
            },
            tags={"tier": str(tier), "git_sha": git_sha, "model": model_name},
        )
        returned = ml.jobs.create_or_update(job)
        print(f"  → {returned.studio_url}")
        if args.wait:
            ml.jobs.stream(returned.name)


if __name__ == "__main__":
    main()
