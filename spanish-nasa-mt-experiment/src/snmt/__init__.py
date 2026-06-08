"""snmt — Spanish<->Nasa-Yuwe NLLB fine-tuning.

This package fine-tunes the NLLB-200 encoder-decoder models (600M / 1.3B / 3.3B)
on the REAL Spanish<->Nasa-Yuwe parallel corpus. It is intentionally separate from
the ``ecmt`` package (which trains SmolLM2 *causal LMs* on the synthetic En<->Zh
proxy): NLLB is a seq2seq architecture and needs a different training path.

The runs are designed to be *packed onto the same H100* as the En<->Zh ablation
matrix, filling spare VRAM so the rented GPU is fully utilised and a single
auto-teardown tears everything down the instant all runs finish.

Public surface:
    - ``snmt.runs.load_nllb_runs`` / ``nllb_run_specs`` — schedule.Run list for packing.
    - ``snmt.data`` — build train/dev/test splits from the parallel jsonl.
    - ``snmt.train.build_and_train`` — one NLLB fine-tune (GPU; lazy heavy imports).
"""

from __future__ import annotations

__all__ = ["lang", "data", "runs", "train"]
__version__ = "0.1.0"

# Language tag for Nasa-Yuwe (Páez, ISO 639-3 `pbb`). Not a native NLLB language —
# added as a new special token at fine-tune time. Spanish is native: `spa_Latn`.
NASA_LANG = "pbb_Latn"
SPANISH_LANG = "spa_Latn"
