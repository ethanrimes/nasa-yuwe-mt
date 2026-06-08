"""Plot training/eval loss and BLEU/chrF curves across all tiers."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
TIERS = [
    ("T10000",   "T10k",   "#1f77b4"),
    ("T50000",   "T50k",   "#ff7f0e"),
    ("T100000",  "T100k",  "#2ca02c"),
    ("T500000",  "T500k",  "#d62728"),
    ("T1000000", "T1M",    "#9467bd"),
]

def load(tier):
    p = ROOT / "metrics" / f"{tier}_trainer_state.json"
    return json.load(open(p))["log_history"]

data = {t[0]: load(t[0]) for t in TIERS}

# 1) Per-tier 4-up: train loss, eval loss, BLEU (avg/en2es/es2en), chrF (en2es)
fig, axes = plt.subplots(4, 5, figsize=(24, 14))
for col, (tier, lbl, color) in enumerate(TIERS):
    h = data[tier]
    train = [(e["step"], e["loss"]) for e in h if "loss" in e and "eval_loss" not in e]
    ev_l  = [(e["step"], e["eval_loss"]) for e in h if "eval_loss" in e]
    bleu  = [e for e in h if "eval_avg_bleu" in e]

    # row 0: train loss
    ax = axes[0, col]
    if train:
        xs, ys = zip(*train); ax.plot(xs, ys, color=color, lw=1)
    ax.set_title(f"{lbl} train_loss"); ax.set_xlabel("step"); ax.grid(alpha=.3)

    # row 1: eval loss
    ax = axes[1, col]
    if ev_l:
        xs, ys = zip(*ev_l); ax.plot(xs, ys, color=color, marker="o", ms=3)
    ax.set_title(f"{lbl} eval_loss"); ax.set_xlabel("step"); ax.grid(alpha=.3)

    # row 2: BLEU
    ax = axes[2, col]
    if bleu:
        xs = [e["step"] for e in bleu]
        ax.plot(xs, [e["eval_avg_bleu"] for e in bleu], "k-", label="avg", lw=2)
        ax.plot(xs, [e["eval_en2es_bleu"] for e in bleu], "--", color="#1f77b4", label="en→es")
        ax.plot(xs, [e["eval_es2en_bleu"] for e in bleu], "--", color="#ff7f0e", label="es→en")
        ax.legend(fontsize=8)
    ax.set_title(f"{lbl} BLEU (100-sample subset)"); ax.set_xlabel("step"); ax.grid(alpha=.3)

    # row 3: chrF en→es and es→en
    ax = axes[3, col]
    if bleu:
        xs = [e["step"] for e in bleu]
        ax.plot(xs, [e["eval_en2es_chrf"] for e in bleu], color="#1f77b4", label="en→es chrF")
        ax.plot(xs, [e["eval_es2en_chrf"] for e in bleu], color="#ff7f0e", label="es→en chrF")
        ax.legend(fontsize=8)
    ax.set_title(f"{lbl} chrF"); ax.set_xlabel("step"); ax.grid(alpha=.3)

fig.suptitle("Per-tier training curves (SmolLM2-360M en↔es)", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.98))
out1 = ROOT / "plot_per_tier.png"
fig.savefig(out1, dpi=110)
print(f"saved {out1}")

# 2) Cross-tier overlay: eval_loss vs cumulative step (each tier resets step, so plot vs epoch fraction)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for tier, lbl, color in TIERS:
    h = data[tier]
    ev = [e for e in h if "eval_loss" in e]
    bl = [e for e in h if "eval_avg_bleu" in e]
    if ev:
        xs = [e["epoch"] for e in ev]; ys = [e["eval_loss"] for e in ev]
        axes[0].plot(xs, ys, color=color, label=lbl, marker="o", ms=3)
    if bl:
        xs = [e["epoch"] if "epoch" in e else e["step"] for e in bl]
        axes[1].plot(xs, [e["eval_avg_bleu"]   for e in bl], color=color, label=lbl, lw=2)
        axes[2].plot(xs, [e["eval_en2es_bleu"] for e in bl], color=color, label=lbl+" en→es")
        axes[2].plot(xs, [e["eval_es2en_bleu"] for e in bl], color=color, label=lbl+" es→en", linestyle="--")

axes[0].set_title("eval_loss vs epoch"); axes[0].set_xlabel("epoch"); axes[0].grid(alpha=.3); axes[0].legend()
axes[1].set_title("avg BLEU vs epoch"); axes[1].set_xlabel("epoch"); axes[1].grid(alpha=.3); axes[1].legend()
axes[2].set_title("per-direction BLEU vs epoch"); axes[2].set_xlabel("epoch"); axes[2].grid(alpha=.3); axes[2].legend(fontsize=7)
fig.suptitle("Cross-tier comparison", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out2 = ROOT / "plot_cross_tier.png"
fig.savefig(out2, dpi=110)
print(f"saved {out2}")

# 3) Best-checkpoint summary table
print("\n=== Best checkpoint per tier ===")
print(f"{'tier':<8} {'best_step':>10} {'best_avg_bleu':>14} {'best_en2es':>12} {'best_es2en':>12} {'final_eval_loss':>16}")
for tier, lbl, _ in TIERS:
    bl = [e for e in data[tier] if "eval_avg_bleu" in e]
    el = [e for e in data[tier] if "eval_loss" in e]
    if not bl: continue
    best = max(bl, key=lambda e: e["eval_avg_bleu"])
    print(f"{lbl:<8} {best['step']:>10} {best['eval_avg_bleu']:>14.3f} "
          f"{best['eval_en2es_bleu']:>12.3f} {best['eval_es2en_bleu']:>12.3f} "
          f"{el[-1]['eval_loss'] if el else float('nan'):>16.4f}")
