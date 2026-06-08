"""Pull all training metrics (summary.json, forgetting.jsonl, trainer_state.json)
from Blob for the 7 finished runs into analysis/metrics/<run_id>/.

Run from repo root:  .venv\\Scripts\\python.exe analysis\\00_pull_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nymt_shared import blob  # noqa: E402

OUT = REPO / "analysis" / "metrics"

# (run_id, blob prefix of the run dir).  en-zh dirs carry their own scale-* run id.
EN_ZH = {
    "1p7b-sentence": "experiments/en-zh/1p7b-sentence/scale-15946-20260608T141021",
    "1p7b-sentvocab": "experiments/en-zh/1p7b-sentvocab/scale-23729-20260608T151709",
    "360m-sentence": "experiments/en-zh/360m-sentence/scale-15946-20260608T142917",
    "360m-sentvocab": "experiments/en-zh/360m-sentvocab/scale-23729-20260608T142917",
}
# fp32+TF32 healthy re-run: 6 cells = {600m,1.3b,3.3b} x {sent, sentvocab}
NLLB = {
    "nllb-600m-sent": "experiments/es-nasa/nllb-600m-es-nasa-sent",
    "nllb-600m-sentvocab": "experiments/es-nasa/nllb-600m-es-nasa-sentvocab",
    "nllb-1.3b-sent": "experiments/es-nasa/nllb-1.3b-es-nasa-sent",
    "nllb-1.3b-sentvocab": "experiments/es-nasa/nllb-1.3b-es-nasa-sentvocab",
    "nllb-3.3b-sent": "experiments/es-nasa/nllb-3.3b-es-nasa-sent",
    "nllb-3.3b-sentvocab": "experiments/es-nasa/nllb-3.3b-es-nasa-sentvocab",
}


def _ckpt_num(name: str) -> int:
    try:
        return int(name.rstrip("/").split("checkpoint-")[-1].split("/")[0])
    except Exception:
        return -1


def latest_trainer_state(all_names: list[str]) -> str | None:
    states = [n for n in all_names if n.endswith("trainer_state.json")]
    if not states:
        return None
    return max(states, key=lambda n: _ckpt_num(n))


def pull(run_id: str, prefix: str, wanted_leaf: list[str]) -> None:
    names = blob.list_blobs(prefix=prefix + "/")
    dest = OUT / run_id
    dest.mkdir(parents=True, exist_ok=True)
    # exact-leaf files
    for leaf in wanted_leaf:
        cand = [n for n in names if n.endswith("/" + leaf) and n.count("/checkpoint-") == 0]
        # prefer the top-level (run-dir) copy, not a checkpoint copy
        top = [c for c in cand if c == f"{prefix}/{leaf}"]
        pick = top[0] if top else (cand[0] if cand else None)
        if pick:
            blob.download_file(pick, dest / leaf)
            print(f"  {run_id}: {leaf}")
    # latest trainer_state.json (full log_history: loss + eval curves)
    ts = latest_trainer_state(names)
    if ts:
        blob.download_file(ts, dest / "trainer_state.json")
        print(f"  {run_id}: trainer_state.json  <- {ts.split('/')[-2]}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for rid, pfx in EN_ZH.items():
        pull(rid, pfx, ["summary.json", "forgetting.jsonl", "train.log"])
    for rid, pfx in NLLB.items():
        pull(rid, pfx, ["summary.json", "train.log"])
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
