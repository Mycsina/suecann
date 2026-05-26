#!/usr/bin/env python3
"""Compare training runs across dated checkpoint folders.

Usage:
    uv run python scripts/compare_runs.py                  # plot all runs
    uv run python scripts/compare_runs.py --runs 2026-05-26-1 2026-05-27-2  # specific runs
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
    return sorted(
        [
            d.name
            for d in path.iterdir()
            if d.is_dir() and (d / "training_stats.csv").exists()
        ]
    )


def load_run(
    run_name: str, checkpoints_dir: str = "checkpoints"
) -> pd.DataFrame | None:
    csv_path = Path(checkpoints_dir) / run_name / "training_stats.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df["run"] = run_name
    return df


def plot_comparison(runs: list[str], checkpoints_dir: str = "checkpoints"):
    dfs = [load_run(r, checkpoints_dir) for r in runs]
    dfs = [df for df in dfs if df is not None]
    if not dfs:
        print("No run data found.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    runs_found = combined["run"].unique()
    print(f"Loaded {len(runs_found)} runs: {list(runs_found)}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Sueca WANN Training Progress — Cross-Run Comparison",
        fontweight="bold",
        fontsize=14,
    )

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs_found)))
    run_colors = dict(zip(runs_found, colors))

    # 1. Best fitness over generations
    ax = axes[0, 0]
    for run in runs_found:
        subset = combined[combined["run"] == run]
        ax.plot(
            subset["generation"],
            subset["best_fitness"],
            label=run,
            color=run_colors[run],
            linewidth=1.0,
            alpha=0.8,
        )
        # Highlight phase transition by marking where phase=1 starts
        phase1_start = subset[subset["phase"] == 1]["generation"].min()
        if pd.notna(phase1_start):
            ax.axvline(
                x=phase1_start,
                color=run_colors[run],
                linestyle="--",
                alpha=0.3,
                linewidth=0.8,
            )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Best Fitness (Accuracy → Delta Game Points)")
    ax.legend(fontsize=7)

    # 2. Best raw delta vs HeuristicBot
    ax = axes[0, 1]
    for run in runs_found:
        subset = combined[(combined["run"] == run) & (combined["phase"] == 1)]
        ax.plot(
            subset["generation"],
            subset["best_delta"],
            label=run,
            color=run_colors[run],
            linewidth=1.0,
            alpha=0.8,
        )
    ax.axhline(
        y=0,
        color="green",
        linestyle="-",
        linewidth=1.0,
        alpha=0.5,
        label="Beat HeuristicBot",
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Delta (game pts)")
    ax.set_title("Phase 1: Raw Delta vs HeuristicBot (> 0 = winning)")
    ax.legend(fontsize=7)

    # 3. Species count
    ax = axes[1, 0]
    for run in runs_found:
        subset = combined[combined["run"] == run]
        ax.plot(
            subset["generation"],
            subset["n_species"],
            label=run,
            color=run_colors[run],
            linewidth=1.0,
            alpha=0.8,
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Number of Species")
    ax.set_title("Species Diversity")
    ax.legend(fontsize=7)

    # 4. Network complexity (connections + hidden nodes)
    ax = axes[1, 1]
    for run in runs_found:
        subset = combined[combined["run"] == run]
        ax.plot(
            subset["generation"],
            subset["n_connections_best"],
            label=f"{run} conns",
            color=run_colors[run],
            linewidth=1.0,
            alpha=0.6,
            linestyle="-",
        )
        ax.plot(
            subset["generation"],
            subset["n_hidden_best"],
            label=f"{run} hidden",
            color=run_colors[run],
            linewidth=1.0,
            alpha=0.6,
            linestyle=":",
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Count")
    ax.set_title("Network Complexity (solid=connections, dotted=hidden nodes)")
    ax.legend(fontsize=6)

    plt.tight_layout()
    out_path = Path(checkpoints_dir) / "run_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved comparison plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Sueca WANN training runs")
    parser.add_argument(
        "--runs", nargs="*", default=None, help="Specific run folders (default: all)"
    )
    parser.add_argument(
        "--checkpoints-dir", default="checkpoints", help="Checkpoints root directory"
    )
    args = parser.parse_args()

    runs = args.runs if args.runs else discover_runs(args.checkpoints_dir)
    if not runs:
        print("No runs found.")
        sys.exit(1)

    plot_comparison(runs, args.checkpoints_dir)


if __name__ == "__main__":
    main()
