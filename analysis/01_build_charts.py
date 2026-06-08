"""Consolidate all collected metrics into charts + a metrics table.

Reads analysis/metrics/<run>/{trainer_state.json,forgetting.jsonl,summary.json}
and the timings jsonl files, writes:
  analysis/charts/*.png
  analysis/metrics_table.json   (per-run final numbers, for the report)

Run:  .venv\\Scripts\\python.exe analysis\\01_build_charts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
M = REPO / "analysis" / "metrics"
C = REPO / "analysis" / "charts"
C.mkdir(parents=True, exist_ok=True)

EN_ZH = ["360m-sentence", "360m-sentvocab", "1p7b-sentence", "1p7b-sentvocab"]
NLLB = ["nllb-600m", "nllb-1.3b", "nllb-3.3b"]
COLORS = {
    "360m-sentence": "#4C72B0", "360m-sentvocab": "#55A868",
    "1p7b-sentence": "#C44E52", "1p7b-sentvocab": "#8172B3",
    "nllb-600m": "#4C72B0", "nllb-1.3b": "#DD8452", "nllb-3.3b": "#C44E52",
}


def load_json(p: Path):
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def log_history(run: str):
    ts = load_json(M / run / "trainer_state.json")
    return ts.get("log_history", []) if ts else []


def train_curve(run: str):
    lh = log_history(run)
    xs = [e["step"] for e in lh if "loss" in e and "eval_loss" not in e]
    ys = [e["loss"] for e in lh if "loss" in e and "eval_loss" not in e]
    return xs, ys


def eval_curve(run: str, key: str = "eval_loss"):
    lh = log_history(run)
    xs = [e["step"] for e in lh if key in e]
    ys = [e[key] for e in lh if key in e]
    return xs, ys


def forgetting(run: str):
    p = M / run / "forgetting.jsonl"
    if not p.exists():
        return [], []
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    return [r["step"] for r in rows], [r["delta_pct"] for r in rows]


def load_timings() -> dict[str, float]:
    """Map exact en-zh run_id -> training wall-clock seconds (run_end.elapsed_seconds)."""
    out: dict[str, float] = {}
    for f in ["enzh_run3_timings.jsonl", "enzh_rerun_timings.jsonl"]:
        p = M / f
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("event") != "run_end" or r.get("status") == "failed":
                continue
            rid = r.get("run_id", "")
            dur = r.get("elapsed_seconds")
            if rid and dur:
                out[rid] = float(dur)
    return out


# exact run_id per cell (from the blob run-dir each cell was downloaded from)
RUN_ID = {
    "1p7b-sentence": "scale-15946-20260608T141021",
    "360m-sentence": "scale-15946-20260608T142917",
    "1p7b-sentvocab": "scale-23729-20260608T151709",
    "360m-sentvocab": "scale-23729-20260608T142917",
}


# ---- Chart 1: en-zh train loss curves ----
def chart_train_loss():
    plt.figure(figsize=(8, 5))
    for run in EN_ZH:
        xs, ys = train_curve(run)
        if xs:
            plt.plot(xs, ys, label=run, color=COLORS[run], lw=1.8)
    plt.xlabel("training step"); plt.ylabel("train loss (cross-entropy)")
    plt.title("En↔Zh proxy: training loss")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(C / "01_enzh_train_loss.png", dpi=130); plt.close()


# ---- Chart 2: en-zh eval loss + eval token accuracy ----
def chart_eval():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for run in EN_ZH:
        xs, ys = eval_curve(run, "eval_loss")
        if xs:
            a1.plot(xs, ys, "-o", label=run, color=COLORS[run])
        xa, ya = eval_curve(run, "eval_mean_token_accuracy")
        if xa:
            a2.plot(xa, ya, "-o", label=run, color=COLORS[run])
    a1.set_title("En↔Zh: held-out eval loss"); a1.set_xlabel("step"); a1.set_ylabel("eval loss")
    a2.set_title("En↔Zh: held-out token accuracy"); a2.set_xlabel("step"); a2.set_ylabel("eval token acc")
    for a in (a1, a2):
        a.legend(); a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(C / "02_enzh_eval.png", dpi=130); plt.close()


# ---- Chart 3: catastrophic forgetting (English PPL delta) ----
def chart_forgetting():
    plt.figure(figsize=(8, 5))
    for run in EN_ZH:
        xs, ys = forgetting(run)
        if xs:
            plt.plot(xs, ys, "-o", label=run, color=COLORS[run], lw=1.8)
    plt.axhline(0, color="k", lw=1, ls="--", alpha=0.6, label="baseline (no forgetting)")
    plt.axhline(20, color="red", lw=1, ls=":", alpha=0.7, label="warn threshold (+20%)")
    plt.xlabel("training step"); plt.ylabel("English PPL change vs initial (%)")
    plt.title("Catastrophic-forgetting probe (lower/flatter = better retention)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(C / "03_forgetting.png", dpi=130); plt.close()


# ---- Chart 4: NLLB train loss curves ----
def chart_nllb_loss():
    plt.figure(figsize=(8, 5))
    for run in NLLB:
        xs, ys = train_curve(run)
        if xs:
            plt.plot(xs, ys, label=run, color=COLORS[run], lw=1.8)
    plt.xlabel("training step"); plt.ylabel("train loss")
    plt.title("NLLB es↔nasa (Yuwe) fine-tune: training loss")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(C / "04_nllb_train_loss.png", dpi=130); plt.close()


# ---- Chart 5: wall-clock timing ----
def chart_timing(table):
    runs = [r for r in table if table[r].get("wall_min")]
    mins = [table[r]["wall_min"] for r in runs]
    cols = [COLORS.get(r, "#888") for r in runs]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(runs, mins, color=cols)
    for b, m in zip(bars, mins):
        if m:
            plt.text(b.get_x() + b.get_width() / 2, m + 0.3, f"{m:.1f}", ha="center", fontsize=9)
    plt.ylabel("training wall-clock (min)"); plt.title("Per-run training wall-clock time")
    plt.xticks(rotation=30, ha="right"); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(C / "05_timing.png", dpi=130); plt.close()


def build_table():
    timings = load_timings()
    table: dict[str, dict] = {}
    for run in EN_ZH:
        lh = log_history(run)
        tx, ty = train_curve(run)
        ex, ey = eval_curve(run, "eval_loss")
        ea_x, ea_y = eval_curve(run, "eval_mean_token_accuracy")
        fx, fy = forgetting(run)
        summ = load_json(M / run / "summary.json") or {}
        wall = timings.get(RUN_ID.get(run, ""))
        table[run] = {
            "kind": "en-zh",
            "final_train_loss": round(ty[-1], 4) if ty else None,
            "final_eval_loss": round(ey[-1], 4) if ey else None,
            "final_eval_token_acc": round(ea_y[-1], 4) if ea_y else None,
            "forgetting_delta_pct_final": round(fy[-1], 3) if fy else None,
            "forgetting_delta_pct_max": round(max(fy), 3) if fy else None,
            "steps": summ.get("expected_steps"),
            "n_examples": summ.get("n_examples"),
            "wall_min": round(wall / 60.0, 1) if wall else None,
        }
    for run in NLLB:
        tx, ty = train_curve(run)
        ex, ey = eval_curve(run, "eval_loss")
        summ = load_json(M / run / "summary.json") or {}
        table[run] = {
            "kind": "nllb-es-nasa",
            "hf_id": summ.get("hf_id"),
            "final_train_loss": round(ty[-1], 4) if ty else None,
            "final_eval_loss": round(ey[-1], 4) if ey else None,
            "epochs": summ.get("epochs"),
            "n_train_bidir": summ.get("n_train_examples_bidir"),
            "wall_min": round(summ.get("elapsed_s", 0) / 60.0, 1) if summ.get("elapsed_s") else None,
        }
    return table


def main():
    table = build_table()
    chart_train_loss()
    chart_eval()
    chart_forgetting()
    chart_nllb_loss()
    chart_timing(table)
    json.dump(table, open(REPO / "analysis" / "metrics_table.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("charts written to analysis/charts/:")
    for p in sorted(C.glob("*.png")):
        print("  ", p.name)
    print("\nmetrics_table.json:")
    print(json.dumps(table, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
