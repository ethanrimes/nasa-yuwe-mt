"""Download the final/ model dir for selected runs from Blob into
analysis/models_cache/<run>/.  Idempotent (skips files already present
with matching size).

Usage:
  .venv\\Scripts\\python.exe analysis\\02a_download_models.py en-zh
  .venv\\Scripts\\python.exe analysis\\02a_download_models.py nllb-600m nllb-1.3b nllb-3.3b
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from nymt_shared import blob  # noqa: E402

CACHE = REPO / "analysis" / "models_cache"

RUN_FINAL = {
    "1p7b-sentence": "experiments/en-zh/1p7b-sentence/scale-15946-20260608T141021/final",
    "1p7b-sentvocab": "experiments/en-zh/1p7b-sentvocab/scale-23729-20260608T151709/final",
    "360m-sentence": "experiments/en-zh/360m-sentence/scale-15946-20260608T142917/final",
    "360m-sentvocab": "experiments/en-zh/360m-sentvocab/scale-23729-20260608T142917/final",
    "nllb-600m": "experiments/es-nasa/nllb-600m-es-nasa/final",
    "nllb-1.3b": "experiments/es-nasa/nllb-1.3b-es-nasa/final",
    "nllb-3.3b": "experiments/es-nasa/nllb-3.3b-es-nasa/final",
}

GROUPS = {
    "en-zh": ["360m-sentence", "360m-sentvocab", "1p7b-sentence", "1p7b-sentvocab"],
}


def download_run(run: str) -> None:
    prefix = RUN_FINAL[run]
    dest = CACHE / run
    dest.mkdir(parents=True, exist_ok=True)
    names = blob.list_blobs(prefix=prefix + "/")
    for n in names:
        leaf = n[len(prefix) + 1:]
        out = dest / leaf
        if out.exists():
            continue
        blob.download_file(n, out)
        print(f"  {run}: {leaf} ({out.stat().st_size/1e6:.1f} MB)")
    print(f"[done] {run} -> {dest}")


def main(argv: list[str]) -> int:
    targets: list[str] = []
    for a in argv:
        if a in GROUPS:
            targets += GROUPS[a]
        elif a in RUN_FINAL:
            targets.append(a)
        else:
            print(f"unknown target {a}")
    for t in targets:
        download_run(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
