#!/usr/bin/env python3
"""Generate report figures for the Sueca-WANN submission.

Outputs (into report/figures/):
  - training_curve.pdf : Phase-0 supervised val-accuracy + Phase-1 self-play fitness
  - tournament.pdf     : v6 champion win% vs each opponent (n=3000), with 95% CIs
  - complexity.pdf     : v5 vs v6 network/rule complexity

All numbers are the canonical, verified results (see problems.md / README).
Run: uv run python scripts/make_report_figures.py
"""
import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)
CSV = os.path.join(ROOT, "checkpoints", "production", "2026-06-14-2", "training_stats.csv")

plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.bbox": "tight"})
PHASE0_END = 150  # gens 0..149 supervised, 150..599 self-play


def training_curve():
    df = pd.read_csv(CSV)
    p0 = df[df.generation < PHASE0_END]
    p1 = df[df.generation >= PHASE0_END]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))

    # Phase 0: supervised validation accuracy of the two brains
    ax[0].plot(p0.generation, p0.lead_val_acc, label="lead brain", color="#3182bd")
    ax[0].plot(p0.generation, p0.follow_val_acc, label="follow brain", color="#de2d26")
    ax[0].axhline(1 / 3, ls=":", color="gray", lw=1, label="chance (3 intents)")
    ax[0].set_title("Phase 0: supervised bootstrap")
    ax[0].set_xlabel("generation")
    ax[0].set_ylabel("validation accuracy")
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].grid(alpha=0.3)

    # Phase 1: self-play best fitness (game-point delta vs HeuristicBot)
    ax[1].plot(p1.generation, p1.lead_best_fitness, label="lead best", color="#3182bd")
    ax[1].plot(p1.generation, p1.follow_best_fitness, label="follow best", color="#de2d26")
    ax[1].axhline(0, ls=":", color="gray", lw=1)
    ax[1].set_title("Phase 1: co-evolutionary self-play")
    ax[1].set_xlabel("generation")
    ax[1].set_ylabel("best fitness (delta vs HeuristicBot)")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIG, "training_curve.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def tournament():
    # Canonical n=3000 results: WANN champion win% vs each opponent + 95% CI half-width.
    opp = ["RandomBot", "OldHeuristicBot", "EliteHeuristicBot"]
    win = [95.0, 67.7, 52.1]
    err = [0.8, 1.7, 1.8]
    colors = ["#9ecae1", "#fdae6b", "#a1d99b"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bars = ax.bar(opp, win, yerr=err, capsize=5, color=colors, edgecolor="black")
    ax.axhline(50, ls="--", color="red", lw=1.2, label="50% (parity)")
    ax.set_ylabel("WANN champion win rate (%)")
    ax.set_title("v6 champion vs baselines (n=3000, seat-rotated)")
    ax.set_ylim(0, 100)
    for b, w in zip(bars, win):
        ax.text(b.get_x() + b.get_width() / 2, w + 2.5, f"{w:.1f}%", ha="center", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "tournament.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def complexity():
    # v5 vs v6: same strength, far simpler. Hidden gates + enabled connections.
    metrics = ["Hidden gates", "Enabled conns"]
    v5 = [132, 188]
    v6 = [29, 49]
    x = range(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.bar([i - w / 2 for i in x], v5, w, label="v5 (52.7%)", color="#c6dbef", edgecolor="black")
    ax.bar([i + w / 2 for i in x], v6, w, label="v6 (52.1%)", color="#a1d99b", edgecolor="black")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("count (lead + follow)")
    ax.set_title("Iso-strength, 4.5x simpler")
    for i, (a, b) in enumerate(zip(v5, v6)):
        ax.text(i - w / 2, a + 3, str(a), ha="center", fontsize=9)
        ax.text(i + w / 2, b + 3, str(b), ha="center", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "complexity.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    training_curve()
    tournament()
    complexity()
