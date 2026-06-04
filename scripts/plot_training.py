"""Generate comprehensive training analysis graphs for a checkpoint.

Handles both legacy single-WANN and new lead/follow dual-WANN formats.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.edgecolor": "#dee2e6",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
})


def detect_format(df: pd.DataFrame) -> str:
    if "lead_best" in df.columns or "lead_best_fitness" in df.columns:
        return "dual"
    return "legacy"


def load_stats(csv_path: str) -> tuple[pd.DataFrame, int, int, str]:
    # Read the first line of the file to see if it is a header
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()
    
    if not first_line:
        raise ValueError(f"CSV file {csv_path} is empty")
        
    parts = first_line.split(',')
    has_header = True
    try:
        float(parts[0])
        # If we successfully parsed it as a float, then it's a number, so it has no header
        has_header = False
    except ValueError:
        pass
        
    if has_header:
        df = pd.read_csv(csv_path)
    else:
        # No header! Read without header and assign column names based on column count
        df = pd.read_csv(csv_path, header=None)
        n_cols = len(df.columns)
        if n_cols == 11:
            df.columns = [
                'generation', 'phase', 'lead_best_fitness', 'lead_avg_fitness',
                'follow_best_fitness', 'follow_avg_fitness', 'lead_n_species',
                'follow_n_species', 'lead_n_connections_best', 'follow_n_connections_best',
                'elapsed_sec'
            ]
        elif n_cols == 13:
            # Older dual or legacy. Older dual starts with lead_best. Legacy starts with best_fitness.
            # All older runs had headers, but if headers are missing:
            # We assume it is older dual if we can, else default to older dual names
            df.columns = [
                'generation', 'phase', 'lead_best', 'lead_avg', 'follow_best', 'follow_avg',
                'best_delta_lead', 'best_delta_follow', 'n_species_lead', 'n_species_follow',
                'n_conns_lead', 'n_conns_follow', 'elapsed_sec'
            ]

    # Standardize dual format column names
    if "lead_best_fitness" in df.columns:
        df["lead_best"] = df["lead_best_fitness"]
        df["lead_avg"] = df["lead_avg_fitness"]
        df["follow_best"] = df["follow_best_fitness"]
        df["follow_avg"] = df["follow_avg_fitness"]
        df["best_delta_lead"] = df["lead_best_fitness"]
        df["best_delta_follow"] = df["follow_best_fitness"]
        df["n_species_lead"] = df["lead_n_species"]
        df["n_species_follow"] = df["follow_n_species"]
        df["n_conns_lead"] = df["lead_n_connections_best"]
        df["n_conns_follow"] = df["follow_n_connections_best"]

    fmt = detect_format(df)
    
    # Standardize legacy column names (just in case they are missing headers)
    if fmt == "legacy" and "global_best_fitness" not in df.columns:
        # If it was legacy and had 13 columns with no header:
        df.columns = [
            'generation', 'phase', 'best_fitness', 'avg_fitness', 'median_fitness',
            'best_delta', 'median_delta', 'global_best_fitness', 'n_species',
            'n_connections_best', 'n_hidden_best', 'oracle_tax', 'elapsed_sec'
        ]

    # Coerce columns to numeric (excluding run/format if present)
    for col in df.columns:
        if col not in ["run", "format"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where generation or phase is NaN
    df = df.dropna(subset=["generation", "phase"])
    df["generation"] = df["generation"].astype(int)
    df["phase"] = df["phase"].astype(int)

    # Filter out corrupted large generation indices (e.g. >= 10000)
    df = df[df["generation"] < 10000]

    # Drop duplicate generations keeping the last
    df = df.drop_duplicates(subset=["generation"], keep="last")

    # Sort by generation ascending
    df = df.sort_values("generation").reset_index(drop=True)

    phase0_end = df[df["phase"] == 0]["generation"].max()
    phase1_start = phase0_end + 1 if pd.notna(phase0_end) else 0
    return df, phase0_end, phase1_start, fmt


def plot_fitness(df: pd.DataFrame, phase1_start: int, fmt: str, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Training Dynamics — Fitness & Performance", fontsize=14, fontweight="bold")

    if fmt == "dual":
        # 1. Lead + Follow best fitness
        ax = axes[0, 0]
        ax.plot(df["generation"], df["lead_best"], color="#2ecc71", linewidth=1.0, alpha=0.8, label="Lead Best")
        ax.plot(df["generation"], df["follow_best"], color="#3498db", linewidth=1.0, alpha=0.8, label="Follow Best")
        if phase1_start > 0:
            ax.axvline(phase1_start, color="#e74c3c", linestyle="--", alpha=0.5, label="Phase 0→1")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")
        ax.set_title("Lead / Follow Best Fitness Over Training")
        ax.legend(fontsize=8)

        # 2. Per-generation avg fitness
        ax = axes[0, 1]
        ax.plot(df["generation"], df["lead_avg"], color="#2ecc71", linewidth=0.6, alpha=0.7, label="Lead Avg")
        ax.plot(df["generation"], df["follow_avg"], color="#3498db", linewidth=0.6, alpha=0.7, label="Follow Avg")
        if phase1_start > 0:
            ax.axvline(phase1_start, color="#e74c3c", linestyle="--", alpha=0.5)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Average Fitness")
        ax.set_title("Per-Generation Average Fitness (Lead/Follow)")
        ax.legend(fontsize=8)

        # 3. Phase 0 accuracy
        ax = axes[1, 0]
        phase0 = df[df["phase"] == 0]
        if len(phase0) > 0:
            ax.plot(phase0["generation"], phase0["lead_best"], color="#2ecc71", linewidth=1.2, label="Lead Acc")
            ax.plot(phase0["generation"], phase0["follow_best"], color="#3498db", linewidth=1.2, label="Follow Acc")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Phase 0 — PIMC Intent Accuracy (L:{phase0['lead_best'].iloc[-1]:.4f} F:{phase0['follow_best'].iloc[-1]:.4f})")
            ax.legend(fontsize=8)

        # 4. Phase 1 delta fitness
        ax = axes[1, 1]
        phase1 = df[df["phase"] == 1]
        if len(phase1) > 0:
            window = max(10, len(phase1) // 50)
            lead_smooth = phase1["best_delta_lead"].rolling(window, center=True, min_periods=1).mean()
            follow_smooth = phase1["best_delta_follow"].rolling(window, center=True, min_periods=1).mean()
            ax.plot(phase1["generation"], phase1["best_delta_lead"], color="#bdc3c7", linewidth=0.3, alpha=0.4)
            ax.plot(phase1["generation"], phase1["best_delta_follow"], color="#bdc3c7", linewidth=0.3, alpha=0.4)
            ax.plot(phase1["generation"], lead_smooth, color="#2ecc71", linewidth=1.5, label=f"Lead δ (w={window})")
            ax.plot(phase1["generation"], follow_smooth, color="#3498db", linewidth=1.5, label=f"Follow δ (w={window})")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Game-Point Delta vs HeuristicBot")
            ax.set_title("Phase 1 — Self-Play Delta Fitness (Lead/Follow)")
            ax.legend(fontsize=8)
    else:
        # Legacy format
        ax = axes[0, 0]
        ax.plot(df["generation"], df["global_best_fitness"], color="#2c3e50", linewidth=1.2)
        if phase1_start > 0:
            ax.axvline(phase1_start, color="#e74c3c", linestyle="--", alpha=0.5, label="Phase 0→1")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Global Best Fitness")
        ax.set_title("Global Best Fitness Over Training")
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        ax.plot(df["generation"], df["best_fitness"], color="#2ecc71", linewidth=0.8, alpha=0.7, label="Best (gen)")
        ax.plot(df["generation"], df["median_fitness"], color="#3498db", linewidth=0.8, alpha=0.7, label="Median")
        ax.plot(df["generation"], df["avg_fitness"], color="#e74c3c", linewidth=0.5, alpha=0.5, label="Mean")
        if phase1_start > 0:
            ax.axvline(phase1_start, color="#e74c3c", linestyle="--", alpha=0.5)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Per-Generation Fitness Distribution")
        ax.legend(fontsize=8)

        ax = axes[1, 0]
        phase0 = df[df["phase"] == 0]
        if len(phase0) > 0:
            ax.plot(phase0["generation"], phase0["global_best_fitness"], color="#8e44ad", linewidth=1.5)
            ax.scatter([phase0["generation"].iloc[-1]], [phase0["global_best_fitness"].iloc[-1]],
                       color="#8e44ad", s=80, zorder=5)
            ax.set_xlabel("Generation")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Phase 0 — PIMC Intent Accuracy (final: {phase0['global_best_fitness'].iloc[-1]:.4f})")

        ax = axes[1, 1]
        phase1 = df[df["phase"] == 1]
        if len(phase1) > 0:
            window = max(10, len(phase1) // 50)
            phase1_global = phase1["global_best_fitness"].rolling(window, center=True, min_periods=1).mean()
            ax.plot(phase1["generation"], phase1["global_best_fitness"], color="#bdc3c7", linewidth=0.4, alpha=0.5)
            ax.plot(phase1["generation"], phase1_global, color="#e67e22", linewidth=1.5, label=f"Smoothed (w={window})")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Game-Point Delta vs HeuristicBot")
            ax.set_title("Phase 1 — Self-Play Delta Fitness")
            ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fitness plots → {out_path}")


def plot_complexity(df: pd.DataFrame, phase1_start: int, fmt: str, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Training Dynamics — Network Complexity & Diversity", fontsize=14, fontweight="bold")

    if fmt == "dual":
        # 1. Connections
        ax = axes[0, 0]
        ax.plot(df["generation"], df["n_conns_lead"], color="#2ecc71", linewidth=1.0, label="Lead Conns")
        ax.plot(df["generation"], df["n_conns_follow"], color="#3498db", linewidth=1.0, label="Follow Conns")
        if phase1_start > 0:
            ax.axvline(phase1_start, color="#95a5a6", linestyle="--", alpha=0.4)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Connections")
        ax.set_title("Network Size Growth (Connections)")
        ax.legend(fontsize=8)

        # 2. Species
        ax = axes[0, 1]
        ax.fill_between(df["generation"], df["n_species_lead"], alpha=0.2, color="#2ecc71")
        ax.fill_between(df["generation"], df["n_species_follow"], alpha=0.2, color="#3498db")
        ax.plot(df["generation"], df["n_species_lead"], color="#27ae60", linewidth=1.0, label="Lead Species")
        ax.plot(df["generation"], df["n_species_follow"], color="#2980b9", linewidth=1.0, label="Follow Species")
        if phase1_start > 0:
            ax.axvline(phase1_start, color="#95a5a6", linestyle="--", alpha=0.4)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Species Count")
        ax.set_title("Species Diversity Over Time")
        ax.legend(fontsize=8)

        # 3. Delta boxplot
        ax = axes[1, 0]
        phase1 = df[df["phase"] == 1]
        if len(phase1) > 20:
            n_bins = min(20, len(phase1) // 50)
            phase1_copy = phase1.copy()
            phase1_copy["gen_bin"] = pd.cut(phase1_copy["generation"], n_bins, labels=False)
            box_data_lead = [
                phase1_copy[(phase1_copy["gen_bin"] == i)]["best_delta_lead"].dropna().values
                for i in range(n_bins)
            ]
            box_data_follow = [
                phase1_copy[(phase1_copy["gen_bin"] == i)]["best_delta_follow"].dropna().values
                for i in range(n_bins)
            ]
            positions_lead = np.arange(1, n_bins + 1) - 0.15
            positions_follow = np.arange(1, n_bins + 1) + 0.15
            bp1 = ax.boxplot(box_data_lead, positions=positions_lead, widths=0.25, patch_artist=True)
            bp2 = ax.boxplot(box_data_follow, positions=positions_follow, widths=0.25, patch_artist=True)
            for patch in bp1["boxes"]:
                patch.set_facecolor("#2ecc71"); patch.set_alpha(0.6)
            for patch in bp2["boxes"]:
                patch.set_facecolor("#3498db"); patch.set_alpha(0.6)
            ax.set_xlabel("Generation Bin")
            ax.set_ylabel("Best Delta per Generation")
            ax.set_title("Delta Fitness Distribution (green=lead, blue=follow)")

        # 4. Pareto: best_delta_lead vs n_conns_lead
        ax = axes[1, 1]
        sample = df.iloc[::max(1, len(df)//500)]
        sc = ax.scatter(sample["n_conns_lead"], sample["lead_best"],
                        c=sample["generation"], cmap="viridis", alpha=0.6, s=20, marker="o", label="Lead")
        sc2 = ax.scatter(sample["n_conns_follow"], sample["follow_best"],
                         c=sample["generation"], cmap="plasma", alpha=0.6, s=20, marker="^", label="Follow")
        ax.set_xlabel("Connections (complexity)")
        ax.set_ylabel("Best Fitness (performance)")
        ax.set_title("Pareto Front: Performance vs Complexity")
        ax.legend(fontsize=7)

    else:
        # Legacy format
        ax = axes[0, 0]
        ax2 = ax.twinx()
        line1, = ax.plot(df["generation"], df["n_connections_best"], color="#3498db", linewidth=1.0, label="Connections")
        line2, = ax2.plot(df["generation"], df["n_hidden_best"], color="#e74c3c", linewidth=1.0, label="Hidden Nodes")
        if phase1_start > 0:
            ax.axvline(phase1_start, color="#95a5a6", linestyle="--", alpha=0.4)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Connections", color="#3498db")
        ax2.set_ylabel("Hidden Nodes", color="#e74c3c")
        ax.set_title("Network Size Growth")
        ax.legend([line1, line2], ["Connections", "Hidden Nodes"], fontsize=8)

        ax = axes[0, 1]
        ax.fill_between(df["generation"], df["n_species"], alpha=0.3, color="#2ecc71")
        ax.plot(df["generation"], df["n_species"], color="#27ae60", linewidth=1.0)
        if phase1_start > 0:
            ax.axvline(phase1_start, color="#95a5a6", linestyle="--", alpha=0.4)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Species Count")
        ax.set_title("Species Diversity Over Time")

        ax = axes[1, 0]
        phase1 = df[df["phase"] == 1]
        if len(phase1) > 20:
            n_bins = min(20, len(phase1) // 50)
            phase1_copy = phase1.copy()
            phase1_copy["gen_bin"] = pd.cut(phase1_copy["generation"], n_bins, labels=False)
            box_data = [
                phase1_copy[phase1_copy["gen_bin"] == i]["best_delta"].dropna().values
                for i in range(n_bins) if len(phase1_copy[phase1_copy["gen_bin"] == i]) > 0
            ]
            bp = ax.boxplot(box_data, widths=0.6, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("#3498db"); patch.set_alpha(0.6)
            ax.set_xlabel("Generation Bin")
            ax.set_ylabel("Best Delta per Generation")
            ax.set_title("Delta Fitness Distribution (Phase 1)")

        ax = axes[1, 1]
        sample = df.iloc[::max(1, len(df)//500)]
        sc = ax.scatter(sample["n_connections_best"], sample["best_fitness"],
                        c=sample["generation"], cmap="viridis", alpha=0.6, s=20)
        ax.set_xlabel("Connections (complexity)")
        ax.set_ylabel("Best Fitness (performance)")
        ax.set_title("Pareto Front: Performance vs Complexity")
        plt.colorbar(sc, ax=ax, label="Generation")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Complexity plots → {out_path}")


def plot_phase_transition(df: pd.DataFrame, phase0_end: int, phase1_start: int, fmt: str, out_path: str):
    """Zoom in on the Phase 0 → 1 transition."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Phase 0 → 1 Transition Analysis", fontsize=14, fontweight="bold")

    if pd.isna(phase0_end):
        for ax in axes:
            ax.text(0.5, 0.5, "Phase 0 data not available\n(overwritten during resume)",
                    ha="center", va="center", color="gray", fontsize=12)
            ax.set_axis_off()
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Transition plots (placeholder) → {out_path}")
        return

    window = 30
    start = max(0, int(phase0_end) - window)
    end = min(len(df) - 1, int(phase1_start) + window)
    transition = df.iloc[start:end + 1]

    if fmt == "dual":
        ax = axes[0]
        phase0_t = transition[transition["phase"] == 0]
        phase1_t = transition[transition["phase"] == 1]
        if len(phase0_t) > 0:
            ax.plot(phase0_t["generation"], phase0_t["lead_best"], color="#2ecc71", linewidth=1.5, label="Lead P0")
            ax.plot(phase0_t["generation"], phase0_t["follow_best"], color="#3498db", linewidth=1.5, label="Follow P0")
        if len(phase1_t) > 0:
            ax.plot(phase1_t["generation"], phase1_t["best_delta_lead"], color="#2ecc71", linewidth=1.5, linestyle="--", label="Lead δ P1")
            ax.plot(phase1_t["generation"], phase1_t["best_delta_follow"], color="#3498db", linewidth=1.5, linestyle="--", label="Follow δ P1")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Fitness Across Phase Transition")
        ax.legend(fontsize=8)

        ax = axes[1]
        if len(phase0_t) > 0:
            ax.plot(phase0_t["generation"], phase0_t["n_species_lead"], color="#2ecc71", linewidth=1.5, label="Lead P0")
            ax.plot(phase0_t["generation"], phase0_t["n_species_follow"], color="#3498db", linewidth=1.5, label="Follow P0")
        if len(phase1_t) > 0:
            ax.plot(phase1_t["generation"], phase1_t["n_species_lead"], color="#2ecc71", linewidth=1.5, linestyle="--", label="Lead P1")
            ax.plot(phase1_t["generation"], phase1_t["n_species_follow"], color="#3498db", linewidth=1.5, linestyle="--", label="Follow P1")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Species Count")
        ax.set_title("Species Diversity Across Phase Transition")
        ax.legend(fontsize=8)
    else:
        ax = axes[0]
        phase0_t = transition[transition["phase"] == 0]
        phase1_t = transition[transition["phase"] == 1]
        if len(phase0_t) > 0:
            ax.plot(phase0_t["generation"], phase0_t["global_best_fitness"], color="#8e44ad", linewidth=2, label="Phase 0 (Accuracy)")
        if len(phase1_t) > 0:
            ax.plot(phase1_t["generation"], phase1_t["global_best_fitness"], color="#e67e22", linewidth=2, label="Phase 1 (Delta)")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Global Best Fitness")
        ax.set_title("Fitness Across Phase Transition")
        ax.legend(fontsize=9)

        ax = axes[1]
        if len(phase0_t) > 0:
            ax.plot(phase0_t["generation"], phase0_t["n_species"], color="#8e44ad", linewidth=2, label="Phase 0")
        if len(phase1_t) > 0:
            ax.plot(phase1_t["generation"], phase1_t["n_species"], color="#e67e22", linewidth=2, label="Phase 1")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Species Count")
        ax.set_title("Species Diversity Across Phase Transition")
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Transition plots → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate training analysis graphs")
    parser.add_argument("--stats", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, phase0_end, phase1_start, fmt = load_stats(args.stats)

    phase0 = df[df["phase"] == 0]
    phase1 = df[df["phase"] == 1]

    print(f"Training Summary ({fmt} format):")
    print(f"  Total generations: {len(df)}")
    print(f"  Phase 0: gen 0 → {phase0_end} ({len(phase0)} gens)")
    if fmt == "dual":
        if len(phase0) > 0:
            print(f"    Final lead accuracy:   {phase0['lead_best'].iloc[-1]:.4f}")
            print(f"    Final follow accuracy: {phase0['follow_best'].iloc[-1]:.4f}")
        print(f"  Phase 1: gen {phase1_start} → {df['generation'].max()} ({len(phase1)} gens)")
        if len(phase1) > 0:
            print(f"    Peak lead delta:   {phase1['best_delta_lead'].max():.4f} @ gen {phase1['generation'].iloc[phase1['best_delta_lead'].argmax()]}")
            print(f"    Peak follow delta: {phase1['best_delta_follow'].max():.4f} @ gen {phase1['generation'].iloc[phase1['best_delta_follow'].argmax()]}")
            print(f"    Max lead species:  {phase1['n_species_lead'].max()}")
            print(f"    Max follow species:{phase1['n_species_follow'].max()}")
            print(f"    Max lead conns:    {phase1['n_conns_lead'].max()}")
            print(f"    Max follow conns:  {phase1['n_conns_follow'].max()}")
    else:
        if len(phase0) > 0:
            print(f"    Final accuracy: {phase0['global_best_fitness'].iloc[-1]:.4f}")
        print(f"  Phase 1: gen {phase1_start} → {df['generation'].max()} ({len(phase1)} gens)")
        if len(phase1) > 0:
            print(f"    Peak delta:   {phase1['global_best_fitness'].max():.4f} @ gen {phase1['generation'].iloc[phase1['global_best_fitness'].argmax()]}")
            print(f"    Final delta:  {phase1['global_best_fitness'].iloc[-1]:.4f}")
            print(f"    Max species:  {phase1['n_species'].max()}")
            print(f"    Max conns:    {phase1['n_connections_best'].max()}")

    plot_fitness(df, phase1_start, fmt, str(out_dir / "training_fitness.png"))
    plot_complexity(df, phase1_start, fmt, str(out_dir / "training_complexity.png"))
    plot_phase_transition(df, phase0_end, phase1_start, fmt, str(out_dir / "training_transition.png"))


if __name__ == "__main__":
    main()
