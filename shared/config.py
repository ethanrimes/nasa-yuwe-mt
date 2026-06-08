"""Central configuration: Azure identifiers, storage layout, model registry.

Everything that another module might need to know about *where things live* is
here so there is a single source of truth. Values can be overridden via env vars
(loaded from a `.env` at the monorepo root if present).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # python-dotenv optional at import time
    load_dotenv = None  # type: ignore

# Monorepo root = parent of the shared/ package dir.
REPO_ROOT = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# --------------------------------------------------------------------------- #
# Azure
# --------------------------------------------------------------------------- #
SUBSCRIPTION_ID = _env("AZURE_SUBSCRIPTION_ID", "5423a68c-2f22-4512-a82c-eb6ed44d9aaf")
RESOURCE_GROUP = _env("AZURE_RESOURCE_GROUP", "nasa-yuwe-mt-rg")
LOCATION = _env("AZURE_LOCATION", "westus2")
STORAGE_ACCOUNT = _env("AZURE_STORAGE_ACCOUNT", "nasayuwemtdata")
STORAGE_CONNECTION_STRING = _env("AZURE_STORAGE_CONNECTION_STRING", "")

# --------------------------------------------------------------------------- #
# Blob containers / prefixes  (single source of truth for storage layout)
# --------------------------------------------------------------------------- #
CONTAINER = "training-data"  # existing container; we namespace with prefixes.

BLOB_LAYOUT = {
    # existing Nasa-Spanish source data (untouched)
    "nasa_source": "data/",
    # reconstructed English<->Chinese proxy dataset
    "enzh_raw": "reconstructed-en-zh/raw/",
    "enzh_processed": "reconstructed-en-zh/processed/",
    "enzh_subsets": "reconstructed-en-zh/subsets/",
    # experiment outputs
    "exp_en_es": "experiments/en-es/",
    "exp_en_zh": "experiments/en-zh/",  # + "<run-id>/{checkpoints,metrics,logs,samples}/"
    # real Spanish<->Nasa-Yuwe NLLB fine-tunes (co-run on the same H100 to fill spare
    # capacity during the En-Zh matrix); + "<run-id>/{checkpoints,metrics,logs}/"
    "exp_es_nasa": "experiments/es-nasa/",
}

# The single existing dataset object we mirror.
NASA_DATASET_BLOB = "data/nasa_yuwe_parallel_dataset.jsonl"
NASA_SUMMARY_BLOB = "data/nasa_yuwe_dataset_summary.json"

# --------------------------------------------------------------------------- #
# H100 VM
# --------------------------------------------------------------------------- #
VM_NAME = _env("H100_VM_NAME", "nymt-h100")
VM_SIZE = _env("H100_VM_SIZE", "Standard_NC40ads_H100_v5")
VM_IMAGE = _env(
    "H100_VM_IMAGE",
    # Ubuntu HPC image ships CUDA drivers preinstalled for NC/ND families.
    "microsoft-dsvm:ubuntu-hpc:2204:latest",
)
USE_SPOT = _env("H100_USE_SPOT", "1") == "1"
ADMIN_USER = _env("H100_ADMIN_USER", "azureuser")
SSH_PUBKEY = os.path.expanduser(_env("H100_SSH_PUBKEY", "~/.ssh/id_rsa.pub"))
SSH_KEY = os.path.expanduser(_env("H100_SSH_KEY", "~/.ssh/id_rsa"))
MAX_BUDGET_HOURS = float(_env("H100_MAX_BUDGET_HOURS", "6"))
OS_DISK_GB = int(_env("H100_OS_DISK_GB", "256"))

# Per-VM-uptime cost reference (USD/hr). Used for budget logging only.
VM_COST_PER_HOUR_SPOT = 6.28
VM_COST_PER_HOUR_ONDEMAND = 6.98

# --------------------------------------------------------------------------- #
# Models under test
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    approx_vram_gb_train: float  # bf16 + grad-checkpointing, per concurrent process


MODELS = {
    "360m": ModelSpec("360m", "HuggingFaceTB/SmolLM2-360M", approx_vram_gb_train=10.0),
    "1.7b": ModelSpec("1.7b", "HuggingFaceTB/SmolLM2-1.7B", approx_vram_gb_train=34.0),
    # ---- NLLB seq2seq models, fine-tuned on the REAL Spanish<->Nasa-Yuwe data ----
    # These are encoder-decoder (M2M100 arch); a *different* training path from the
    # SmolLM2 causal-LM proxy runs (see spanish-nasa-mt-experiment / pkg `snmt`).
    # They are packed onto the same H100 alongside the En-Zh matrix to fill spare
    # VRAM (max GPU utilization) and share the one auto-teardown. Trained in fp32 +
    # TF32 (bf16 autocast diverges on this corpus — see snmt/train.py); fp32 keeps
    # full master weights + fp32 AdamW state (~16 bytes/param) so the footprint is
    # ~2x the bf16 estimate. Deliberately conservative after a real OOM: with
    # usable=72 GB the 3.3B (75) always seats ALONE, while a 1.3B (32) can pair with
    # a 600M (18) or another 1.3B (64) — the 3.3B never shares a card.
    "nllb-600m": ModelSpec("nllb-600m", "facebook/nllb-200-distilled-600M", approx_vram_gb_train=18.0),
    "nllb-1.3b": ModelSpec("nllb-1.3b", "facebook/nllb-200-1.3B", approx_vram_gb_train=32.0),
    "nllb-3.3b": ModelSpec("nllb-3.3b", "facebook/nllb-200-3.3B", approx_vram_gb_train=75.0),
}

H100_VRAM_GB = 80.0
# Leave headroom for fragmentation / activation spikes.
H100_USABLE_VRAM_GB = 72.0

# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #
WANDB_PROJECT = _env("WANDB_PROJECT", "nasa-yuwe-mt")
WANDB_ENTITY = _env("WANDB_ENTITY", "")

# --------------------------------------------------------------------------- #
# Local paths
# --------------------------------------------------------------------------- #
EN_ZH_DIR = REPO_ROOT / "english-chinese-mt-experiment"
EN_ES_DIR = REPO_ROOT / "english-spanish-mt-experiment"
ES_NASA_DIR = REPO_ROOT / "spanish-nasa-mt-experiment"  # NLLB real-data fine-tunes
ENZH_DATA_DIR = EN_ZH_DIR / "data-en-zh"  # reconstructed dataset lands here


@dataclass
class StorageSettings:
    account: str = STORAGE_ACCOUNT
    container: str = CONTAINER
    connection_string: str = STORAGE_CONNECTION_STRING
    layout: dict = field(default_factory=lambda: dict(BLOB_LAYOUT))


def summary() -> str:
    return (
        f"subscription={SUBSCRIPTION_ID}\n"
        f"resource_group={RESOURCE_GROUP}  location={LOCATION}\n"
        f"storage_account={STORAGE_ACCOUNT}  container={CONTAINER}\n"
        f"vm={VM_NAME} ({VM_SIZE}) spot={USE_SPOT} max_budget_h={MAX_BUDGET_HOURS}\n"
        f"models={list(MODELS)}"
    )


if __name__ == "__main__":
    print(summary())
