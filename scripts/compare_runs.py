#!/usr/bin/env python3
"""Compare training runs across dated checkpoint folders.

Handles both legacy single-WANN format and new lead/follow dual-WANN format.

Usage:
    uv run python scripts/compare_runs.py                  # plot all runs
    uv run python scripts/compare_runs.py --runs 2026-05-28-1 2026-06-01-1  # specific runs
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def discover_runs(checkpoints_dir: str = "checkpoints") -> list[str]:
    path = Path(checkpoints_dir)
    if not path.is_dir():
        return []
    return sorted([
        d.name for d in path.iterdir()
        if d.is_dir() and (d / "training_stats.csv").exists()
    ])


def detect_format(df: pd.DataFrame) -> str:
    """Detect whether CSV uses old single-WANN or new lead/follow format."""
    if "lead_best" in df.columns:
        return "dual"
    return "legacy"


def load_run(run_name: str, checkpoints_dir: str = "checkpoints") -> tuple[pd.DataFrame | None, str]:
    csv_path = Path(checkpoints_dir) / run_name / "training_stats.csv"
    if not csv_path.exists():
        return None, "unknown"
    df = pd.read_csv(csv_path)
    fmt = detect_format(df)
    df["run"] = run_name
    df["format"] = fmt
    return df, fmt


def plot_comparison(runs: list[str], checkpoints_dir: str = "checkpoints"):
    results = [load_run(r, checkpoints_dir) for r in runs]
    results = [(df, fmt) for df, fmt in results if df is not None]
    if not results:
        print("No run data found.")
        return

    combined = pd.concat([df for df, _ in results], ignore_index=True)
    runs_found = combined["run"].unique()
    print(f"Loaded {len(runs_found)} runs: {list(runs_found)}")

    # Separate legacy and dual-format runs
    legacy_runs = [r for r in runs_found if combined[combined["run"] == r]["format"].iloc[0] == "legacy"]
    dual_runs = [r for r in runs_found if combined[combined["run"] == r]["format"].iloc[0] == "dual"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Sueca WANN Training Progress — Cross-Run Comparison",
        fontweight="bold", fontsize=14,
    )

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs_found)))
    run_colors = dict(zip(runs_found, colors))

    # 1. Best fitness over generations
    ax = axes[0, 0]
    for run in runs_found:
        subset = combined[combined["run"] == run]
        fmt = subset["format"].iloc[0]
        if fmt == "dual":
            # Plot both lead and follow
            ax.plot(subset["generation"], subset["lead_best"],
                    label=f"{run} lead", color=run_colors[run],
                    linewidth=1.0, alpha=0.8, linestyle="-")
            ax.plot(subset["generation"], subset["follow_best"],
                    label=f"{run} follow", color=run_colors[run],
                    linewidth=1.0, alpha=0.8, linestyle="--")
        else:
            ax.plot(subset["generation"], subset["best_fitness"],
                    label=run, color=run_colors[run],
                    linewidth=1.0, alpha=0.8)
        # Mark phase transition
        phase1_start = subset[subset["phase"] == 1]["generation"].min()
        if pd.notna(phase1_start):
            ax.axvline(x=phase1_start, color=run_colors[run],
                       linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Best Fitness (Accuracy → Delta Game Points)")
    ax.legend(fontsize=6)

    # 2. Phase 1 delta
    ax = axes[0, 1]
    for run in runs_found:
        subset = combined[(combined["run"] == run) & (combined["phase"] == 1)]
        fmt = combined[combined["run"] == run]["format"].iloc[0]
        if fmt == "dual":
            ax.plot(subset["generation"], subset["best_delta_lead"],
                    label=f"{run} lead δ", color=run_colors[run],
                    linewidth=1.0, alpha=0.8, linestyle="-")
            ax.plot(subset["generation"], subset["best_delta_follow"],
                    label=f"{run} follow δ", color=run_colors[run],
                    linewidth=1.0, alpha=0.8, linestyle="--")
        else:
            ax.plot(subset["generation"], subset["best_delta"],
                    label=f"{run} δ", color=run_colors[run],
                    linewidth=1.0, alpha=0.8)
    ax.axhline(y=0, color="green", linestyle="-", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Delta (game pts)")
    ax.set_title("Phase 1: Raw Delta vs HeuristicBot (> 0 = winning)")
    ax.legend(fontsize=6)

    # 3. Species count
    ax = axes[1, 0]
    for run in runs_found:
        subset = combined[combined["run"] == run]
        fmt = subset["format"].iloc[0]
        if fmt == "dual":
            ax.plot(subset["generation"], subset["n_species_lead"],
                    label=f"{run} lead spp", color=run_colors[run],
                    linewidth=1.0, alpha=0.6, linestyle="-")
            ax.plot(subset["generation"], subset["n_species_follow"],
                    label=f"{run} follow spp", color=run_colors[run],
                    linewidth=1.0, alpha=0.6, linestyle="--")
        else:
            ax.plot(subset["generation"], subset["n_species"],
                    label=run, color=run_colors[run],
                    linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Number of Species")
    ax.set_title("Species Diversity")
    ax.legend(fontsize=6)

    # 4. Network complexity
    ax = axes[1, 1]
    for run in runs_found:
        subset = combined[combined["run"] == run]
        fmt = subset["format"].iloc[0]
        if fmt == "dual":
            ax.plot(subset["generation"], subset["n_conns_lead"],
                    label=f"{run} lead conns", color=run_colors[run],
                    linewidth=1.0, alpha=0.6, linestyle="-")
            ax.plot(subset["generation"], subset["n_conns_follow"],
                    label=f"{run} follow conns", color=run_colors[run],
                    linewidth=1.0, alpha=0.6, linestyle="--")
        else:
            ax.plot(subset["generation"], subset["n_connections_best"],
                    label=f"{run} conns", color=run_colors[run],
                    linewidth=1.0, alpha=0.6, linestyle="-")
            if "n_hidden_best" in subset.columns:
                ax.plot(subset["generation"], subset["n_hidden_best"],
                        label=f"{run} hidden", color=run_colors[run],
                        linewidth=1.0, alpha=0.6, linestyle=":")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Count")
    ax.set_title("Network Complexity (solid=connections, dotted=hidden)")
    ax.legend(fontsize=5)

    plt.tight_layout()
    out_path = Path(checkpoints_dir) / "run_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved comparison plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Sueca WANN training runs")
    parser.add_argument("--runs", nargs="*", default=None,
                        help="Specific run folders (default: all)")
    parser.add_argument("--checkpoints-dir", default="checkpoints",
                        help="Checkpoints root directory")
    args = parser.parse_args()

    runs = args.runs if args.runs else discover_runs(args.checkpoints_dir)
    if not runs:
        print("No runs found.")
        sys.exit(1)

    plot_comparison(runs, args.checkpoints_dir)


if __name__ == "__main__":
    main()
