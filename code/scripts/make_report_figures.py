#!/usr/bin/env python3
"""Generate report figures for the Sueca-WANN submission.

Outputs (into report/figures/):
  - training_curve.pdf : Phase-0 supervised val-accuracy + Phase-1 self-play fitness
  - tournament.pdf     : WANN champion win% vs each opponent (n=3000), with 95% CIs
  - complexity.pdf     : earlier vs final champion complexity

All numbers are the canonical, verified results (see problems.md / README).
Run: uv run python scripts/make_report_figures.py

Design principles (Bertin, Semiology of Graphics):
  - Colour hue only for qualitative (nominal) distinctions; never for ordered data.
  - One consistent colour family for WANN (blue), another for baselines (warm).
  - Direct label marks when the number of categories is small (≤ 4).
  - Phase-region shading encodes the training-phase nominal component.
  - Okabe-Ito colourblind-safe palette throughout.
"""

import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.normpath(os.path.join(ROOT, "..", "report", "figures"))
os.makedirs(FIG, exist_ok=True)
CSV = os.path.join(ROOT, "checkpoints", "production", "2026-06-14-2",
                   "training_stats.csv")

# ── Okabe-Ito colourblind-safe palette ──────────────────────────────────
C_WANN       = "#0072B2"   # blue          — WANN / champion (primary)
C_WANN_ALT   = "#56B4E9"   # sky blue      — WANN (secondary)
C_LEAD        = "#0072B2"   # blue          — LEAD brain
C_FOLLOW      = "#D55E00"   # vermilion     — FOLLOW brain (distinct from lead)
C_ELITE      = "#D55E00"   # vermilion     — Elite / strongest opponent
C_BASELINE   = "#E69F00"   # orange        — mid-tier baseline
C_GREEN      = "#009E73"   # bluish green  — weak baseline / accent
C_PURPLE     = "#CC79A7"   # reddish purple
C_REFERENCE  = "#404040"   # dark grey     — reference lines & labels
C_GRID       = "#d0d0d0"   # light grey    — gridlines

# ── Global style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "figure.dpi":        150,
    "savefig.bbox":      "tight",
    "font.family":       "sans-serif",
    "font.sans-serif":   ["DejaVu Sans", "Liberation Sans",
                          "Arial", "Helvetica"],
    # Remove chartjunk — no top/right spines
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "legend.frameon":     False,
    "legend.fontsize":    8,
})

PHASE0_END = 150   # gens 0..149 = supervised; 150..599 = self-play


# ══════════════════════════════════════════════════════════════════════════
#  Figure 1 — Training Curve
# ══════════════════════════════════════════════════════════════════════════

def training_curve():
    df = pd.read_csv(CSV)
    p0 = df[df.generation < PHASE0_END]
    p1 = df[df.generation >= PHASE0_END]

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))

    # ── Phase 0 panel ───────────────────────────────────────────────────
    ax0 = ax[0]

    # Phase-region shading (subtle, behind data)
    ax0.axvspan(p0.generation.min(), p0.generation.max(),
                facecolor="#E8F4FD", zorder=0)
    ax0.text(0.03, 0.95, "Phase 0", transform=ax0.transAxes,
             fontsize=8, fontstyle="italic", color=C_REFERENCE,
             ha="left", va="top")

    ax0.plot(p0.generation, p0.lead_val_acc,
             color=C_LEAD, lw=1.4)
    ax0.plot(p0.generation, p0.follow_val_acc,
             color=C_FOLLOW, lw=1.4)

    # Chance reference line with label
    ax0.axhline(1/3, ls="--", color=C_REFERENCE, lw=0.8, zorder=2)
    ax0.text(p0.generation.iloc[-1] + 2, 1/3 + 0.008,
             "chance (⅓)", fontsize=7.5, color=C_REFERENCE,
             ha="right", va="bottom")

    # Direct labels at the right edge of each curve
    for col, color, name in [
        ("lead_val_acc",   C_LEAD,     "lead"),
        ("follow_val_acc", C_FOLLOW,   "follow"),
    ]:
        y_end = p0[col].iloc[-1]
        ax0.annotate(name, xy=(p0.generation.iloc[-1], y_end),
                     xytext=(6, 0), textcoords="offset points",
                     fontsize=8, color=color, va="center", fontweight="bold",
                     clip_on=False)

    ax0.set_title("Phase 0: Supervised Bootstrap", fontweight="bold", pad=8)
    ax0.set_xlabel("Generation")
    ax0.set_ylabel("Validation Accuracy")
    ax0.set_xlim(p0.generation.min(), p0.generation.max() + 16)
    ax0.set_ylim(0.18, 0.65)
    ax0.yaxis.set_major_formatter(
        mticker.PercentFormatter(1.0, decimals=0))
    ax0.grid(axis="y", color=C_GRID, lw=0.4, alpha=0.6)

    # ── Phase 1 panel ───────────────────────────────────────────────────
    ax1 = ax[1]

    ax1.axvspan(p1.generation.min(), p1.generation.max(),
                facecolor="#FFF3E0", zorder=0)
    ax1.text(0.03, 0.95, "Phase 1", transform=ax1.transAxes,
             fontsize=8, fontstyle="italic", color=C_REFERENCE,
             ha="left", va="top")

    ax1.plot(p1.generation, p1.lead_best_fitness,
             color=C_LEAD, lw=1.4)
    ax1.plot(p1.generation, p1.follow_best_fitness,
             color=C_FOLLOW, lw=1.4)

    # Zero reference (HeuristicBot parity)
    ax1.axhline(0, ls="--", color=C_REFERENCE, lw=0.8, zorder=2)
    ax1.text(p1.generation.iloc[0] + 4, 0.12,
             "HeuristicBot parity", fontsize=7.5, color=C_REFERENCE,
             ha="left", va="bottom")

    # Direct labels — stagger vertically so the two end labels never collide
    ends = {"lead": p1.lead_best_fitness.iloc[-1],
            "follow": p1.follow_best_fitness.iloc[-1]}
    lo, hi = sorted(ends, key=ends.get)
    dy = {lo: -8, hi: 8}
    for col, color, name in [
        ("lead_best_fitness",   C_LEAD,     "lead"),
        ("follow_best_fitness", C_FOLLOW,   "follow"),
    ]:
        y_end = p1[col].iloc[-1]
        ax1.annotate(name, xy=(p1.generation.iloc[-1], y_end),
                     xytext=(6, dy[name]), textcoords="offset points",
                     fontsize=8, color=color, va="center", fontweight="bold",
                     clip_on=False)

    ax1.set_title("Phase 1: Co-evolutionary Self-play",
                  fontweight="bold", pad=8)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Fitness  (Δ game-points vs HeuristicBot)")
    ax1.set_xlim(p1.generation.min(), p1.generation.max() + 32)
    ax1.grid(axis="y", color=C_GRID, lw=0.4, alpha=0.6)

    # ── Save ────────────────────────────────────────────────────────────
    fig.tight_layout(pad=2.0)
    out = os.path.join(FIG, "training_curve.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# ══════════════════════════════════════════════════════════════════════════
#  Figure 2 — Tournament (v6 champion vs baselines)
# ══════════════════════════════════════════════════════════════════════════

def tournament():
    # Canonical n=3000 results — DO NOT EDIT VALUES.
    opp   = ["RandomBot",       "OldHeuristicBot",  "EliteHeuristicBot"]
    win   = [95.0,              67.7,               52.1]
    err   = [0.8,               1.7,                1.8]

    # Opponents ordered by difficulty → warm-to-cool sequential hue
    bar_colors = [C_GREEN, C_BASELINE, C_ELITE]

    fig, ax = plt.subplots(figsize=(5.8, 4.0))

    bars = ax.bar(opp, win, yerr=err, capsize=6,
                  color=bar_colors, edgecolor="white", lw=0.8, width=0.55,
                  error_kw={"lw": 1.0, "capthick": 1.0})

    # 50 % parity reference — the key "does it beat" threshold
    ax.axhline(50, ls="--", color=C_REFERENCE, lw=1.0, zorder=3)
    ax.text(0.01, 51.5, "parity (50%)", fontsize=8,
            color=C_REFERENCE, ha="left", va="bottom",
            transform=ax.get_yaxis_transform())

    ax.set_ylabel("WANN Champion Win Rate  (%)")
    ax.set_title("WANN Champion vs Baselines  (n = 3000, seat-rotated)",
                 fontweight="bold", pad=10)
    ax.set_ylim(0, 115)

    # Direct labels: win rate + CI on each bar
    for bar, w, e in zip(bars, win, err):
        cx = bar.get_x() + bar.get_width() / 2
        top = bar.get_height()
        ax.text(cx, top + e + 6.0, f"{w:.1f} %",
                ha="center", fontsize=11, fontweight="bold")
        ax.text(cx, top + e + 2.0, f"±{e:.1f} pp",
                ha="center", fontsize=7.5, color=C_REFERENCE)

    # X-axis labels are self-explanatory → no legend needed
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", color=C_GRID, lw=0.4, alpha=0.6)

    fig.tight_layout(pad=1.5)
    out = os.path.join(FIG, "tournament.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# ══════════════════════════════════════════════════════════════════════════
#  Figure 3 — Complexity (v5 vs v6)
# ══════════════════════════════════════════════════════════════════════════

def complexity():
    metrics = ["Hidden Gates",   "Enabled Connections"]
    v5      = [132,              188]
    v6      = [29,               49]
    x       = range(len(metrics))
    w       = 0.38

    fig, ax = plt.subplots(figsize=(5.4, 3.8))

    b5 = ax.bar([i - w/2 for i in x], v5, w,
                color=C_WANN_ALT, edgecolor="white", lw=0.8,
                label="Earlier champion (52.7%)")
    b6 = ax.bar([i + w/2 for i in x], v6, w,
                color=C_WANN, edgecolor="white", lw=0.8,
                label="Final champion (52.1%)")

    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Count  (lead + follow)")
    ax.set_title("Equal strength, 4.5× fewer gates", fontweight="bold", pad=10)

    # Direct labels on bars
    for bar in b5:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                str(int(bar.get_height())), ha="center",
                fontsize=10, fontweight="bold", color=C_WANN_ALT)
    for bar in b6:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                str(int(bar.get_height())), ha="center",
                fontsize=10, fontweight="bold", color=C_WANN)

    # Reduction annotations
    reductions = [
        (0, 132, 29,  "−78 %"),
        (1, 188, 49,  "−74 %"),
    ]
    for xi, v5_val, v6_val, label in reductions:
        y_mid = v6_val + (v5_val - v6_val) / 2
        ax.annotate(label, xy=(xi, y_mid),
                    fontsize=8.5, color=C_REFERENCE,
                    ha="center", va="center", fontstyle="italic")

    ax.legend(fontsize=9, loc="upper left",
              handlelength=1.0, handleheight=1.0, borderpad=0.6)
    ax.grid(axis="y", color=C_GRID, lw=0.4, alpha=0.6)
    ax.set_ylim(0, 215)

    fig.tight_layout(pad=1.5)
    out = os.path.join(FIG, "complexity.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# ══════════════════════════════════════════════════════════════════════════
#  Slide-only figures — single-panel, larger, distinct lead/follow colours
# ══════════════════════════════════════════════════════════════════════════

def training_phase0_slide():
    """Phase 0 only — single panel, larger, for presentation slide."""
    df = pd.read_csv(CSV)
    p0 = df[df.generation < PHASE0_END]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axvspan(p0.generation.min(), p0.generation.max(),
               facecolor="#E8F4FD", zorder=0)

    ax.plot(p0.generation, p0.lead_val_acc, color=C_LEAD, lw=1.2)
    ax.plot(p0.generation, p0.follow_val_acc, color=C_FOLLOW, lw=1.2)

    ax.axhline(1/3, ls="--", color=C_REFERENCE, lw=0.8, zorder=2)
    ax.text(p0.generation.iloc[-1] + 2, 1/3 + 0.008,
            "chance (⅓)", fontsize=8, color=C_REFERENCE,
            ha="right", va="bottom")

    for col, color, name in [
        ("lead_val_acc",   C_LEAD,   "lead"),
        ("follow_val_acc", C_FOLLOW, "follow"),
    ]:
        y_end = p0[col].iloc[-1]
        ax.annotate(name, xy=(p0.generation.iloc[-1], y_end),
                     xytext=(8, 0), textcoords="offset points",
                     fontsize=9, color=color, va="center", fontweight="bold",
                     clip_on=False)

    ax.set_title("Phase 0 — Supervised Bootstrap", fontweight="bold", pad=10,
                 fontsize=13)
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Validation Accuracy", fontsize=11)
    ax.set_xlim(p0.generation.min(), p0.generation.max() + 18)
    ax.set_ylim(0.18, 0.65)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=C_GRID, lw=0.4, alpha=0.6)

    fig.tight_layout(pad=2.0)
    out = os.path.join(FIG, "training_phase0.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def training_phase1_slide():
    """Phase 1 only — two lines drawn with distinct styles.

    Lead and follow share identical fitness values (co-evolved, evaluated
    on the same games), so the lines are superimposed. We draw both with
    different colours and line styles so the audience can see that both
    brains are present even when their fitness traces overlap.
    """
    df = pd.read_csv(CSV)
    p1 = df[df.generation >= PHASE0_END]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axvspan(p1.generation.min(), p1.generation.max(),
               facecolor="#FFF3E0", zorder=0)

    # Draw both even though values are identical — different styles
    # so the audience can see traces of both.
    ax.plot(p1.generation, p1.lead_best_fitness,
            color=C_LEAD, lw=1.8, linestyle='-',  zorder=5)
    ax.plot(p1.generation, p1.follow_best_fitness,
            color=C_FOLLOW, lw=1.8, linestyle='--', zorder=4)

    ax.axhline(0, ls=":", color=C_REFERENCE, lw=0.8, zorder=2)
    ax.text(p1.generation.iloc[0] + 4, 0.12,
            "HeuristicBot parity", fontsize=8, color=C_REFERENCE,
            ha="left", va="bottom")

    ax.set_title("Phase 1 — Co-evolutionary Self-play", fontweight="bold",
                 pad=10, fontsize=13)
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Best Fitness  (Δ game-points vs HeuristicBot)", fontsize=11)
    ax.set_xlim(p1.generation.min(), p1.generation.max() + 35)
    ax.grid(axis="y", color=C_GRID, lw=0.4, alpha=0.6)

    fig.tight_layout(pad=2.0)
    out = os.path.join(FIG, "training_phase1.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    training_curve()
    training_phase0_slide()
    training_phase1_slide()
    tournament()
    complexity()
