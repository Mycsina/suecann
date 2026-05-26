# src/benchmark.py
"""Tournament Benchmarking Suite for Sueca Bots.

Runs a round-robin tournament between:
- RandomBot
- HeuristicBot
- PIMCBot
- WannBotSweep (Champion)

Uses duplicate deals (symmetric seat rotation) to eliminate deal luck and
computes 95% binomial confidence intervals. All game simulations run in Rust.
"""

import argparse
import csv
import math
import os
import sys
import time
from typing import Any

import numpy as np

# Set matplotlib backend to Agg before any imports to prevent headless issues
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


from src.compat import (
    DealRecord,
    generate_deals,
    RustDealCompat,
)


def compute_binomial_ci_margin(p: float, n: int) -> float:
    """Compute the 95% binomial confidence interval margin of error."""
    if n <= 0:
        return 0.0
    return 1.96 * math.sqrt(p * (1.0 - p) / n)


# ─── Matchup Runner delegating to Rust ──────────────────────────────────────


def run_matchup(
    bot_a: tuple[int, Any],
    bot_b: tuple[int, Any],
    deals: list[DealRecord],
    base_seed: int,
    use_multiprocessing: bool = True,
    pimc_worlds: int = 80,
    pimc_depth: int = 4,
) -> tuple[dict, dict]:
    """Run a matchup in Rust and return statistics for BOTH bots."""
    import sueca_solver

    # Convert Python deals to PyDeal-compatible structure for Rust
    rust_deals = [
        RustDealCompat(
            [[int(c.suit) * 10 + int(c.rank) for c in hand] for hand in deal_rec.hands],
            int(deal_rec.trump),
            deal_rec.seed,
        )
        for deal_rec in deals
    ]

    bot_a_type, bot_a_network = bot_a
    bot_b_type, bot_b_network = bot_b

    # Define weight sweep
    sweep_weights = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

    # Delegate play to Rust parallel evaluator
    score_a1, score_a2, gpts_a1, gpts_a2, gpts_b1, gpts_b2 = (
        sueca_solver.run_matchup_rust(
            rust_deals,
            bot_a_type,
            bot_a_network,
            bot_b_type,
            bot_b_network,
            sweep_weights,
            base_seed,
            pimc_worlds,
            pimc_depth,
        )
    )

    total_games = len(deals) * 2
    wins_a, wins_b = 0, 0
    losses_a, losses_b = 0, 0
    ties = 0
    pts_a, pts_b = 0.0, 0.0
    gpts_a, gpts_b = 0.0, 0.0

    for sa1, sa2, ga1, ga2, gb1, gb2 in zip(
        score_a1, score_a2, gpts_a1, gpts_a2, gpts_b1, gpts_b2
    ):
        sb1 = 120 - sa1
        sb2 = 120 - sa2

        # Game 1: A is seats 0/2 (team 02), B is seats 1/3 (team 13)
        pts_a += sa1
        pts_b += sb1
        gpts_a += ga1
        gpts_b += gb1
        if sa1 > 60:
            wins_a += 1
            losses_b += 1
        elif sa1 < 60:
            losses_a += 1
            wins_b += 1
        else:
            ties += 1

        # Game 2: B is seats 0/2 (team 02), A is seats 1/3 (team 13)
        pts_a += sa2
        pts_b += sb2
        gpts_a += ga2
        gpts_b += gb2
        if sa2 > 60:
            wins_a += 1
            losses_b += 1
        elif sa2 < 60:
            losses_a += 1
            wins_b += 1
        else:
            ties += 1

    win_rate_a = (wins_a + 0.5 * ties) / total_games
    win_rate_b = (wins_b + 0.5 * ties) / total_games

    ci_a = compute_binomial_ci_margin(win_rate_a, total_games)
    ci_b = compute_binomial_ci_margin(win_rate_b, total_games)

    stats_a = {
        "wins": wins_a,
        "losses": losses_a,
        "ties": ties,
        "win_rate": win_rate_a,
        "ci_margin": ci_a,
        "avg_pts": pts_a / total_games,
        "avg_gpts": gpts_a / total_games,
        "total_games": total_games,
    }

    stats_b = {
        "wins": wins_b,
        "losses": losses_b,
        "ties": ties,
        "win_rate": win_rate_b,
        "ci_margin": ci_b,
        "avg_pts": pts_b / total_games,
        "avg_gpts": gpts_b / total_games,
        "total_games": total_games,
    }

    return stats_a, stats_b


# ─── Tournament Runner ──────────────────────────────────────────────────────


def run_tournament(
    bots_dict: dict,
    n_deals: int,
    base_seed: int,
    use_multiprocessing: bool = True,
    pimc_worlds: int = 80,
    pimc_depth: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a complete round-robin tournament between registered bots."""
    bot_names = list(bots_dict.keys())
    n_bots = len(bot_names)

    win_rate_matrix = np.zeros((n_bots, n_bots))
    ci_matrix = np.zeros((n_bots, n_bots))
    pts_matrix = np.zeros((n_bots, n_bots))
    gpts_matrix = np.zeros((n_bots, n_bots))

    print(f"Generating {n_deals} duplicate deals for the tournament...")
    deals = generate_deals(gen=0, n_deals=n_deals, base_seed=base_seed * 1000)

    matchup_idx = 1
    total_matchups = int(n_bots * (n_bots - 1) / 2)

    for i in range(n_bots):
        for j in range(i + 1, n_bots):
            name_a = bot_names[i]
            name_b = bot_names[j]
            bot_a = bots_dict[name_a]
            bot_b = bots_dict[name_b]

            print(
                f"\n[{matchup_idx}/{total_matchups}] Running matchup: {name_a} vs {name_b}..."
            )
            t0 = time.time()
            stats_a, stats_b = run_matchup(
                bot_a,
                bot_b,
                deals,
                base_seed,
                use_multiprocessing,
                pimc_worlds,
                pimc_depth,
            )
            elapsed = time.time() - t0
            print(
                f"  -> Result: {name_a} win rate = {stats_a['win_rate'] * 100:.1f}% ± {stats_a['ci_margin'] * 100:.1f}% | "
                f"{name_b} win rate = {stats_b['win_rate'] * 100:.1f}% ± {stats_b['ci_margin'] * 100:.1f}%"
            )
            print(
                f"  -> Card Pts: {name_a} = {stats_a['avg_pts']:.1f} vs {name_b} = {stats_b['avg_pts']:.1f} | Time = {elapsed:.1f}s"
            )

            # Fill A vs B
            win_rate_matrix[i, j] = stats_a["win_rate"]
            ci_matrix[i, j] = stats_a["ci_margin"]
            pts_matrix[i, j] = stats_a["avg_pts"]
            gpts_matrix[i, j] = stats_a["avg_gpts"]

            # Fill B vs A
            win_rate_matrix[j, i] = stats_b["win_rate"]
            ci_matrix[j, i] = stats_b["ci_margin"]
            pts_matrix[j, i] = stats_b["avg_pts"]
            gpts_matrix[j, i] = stats_b["avg_gpts"]

            matchup_idx += 1

    # Fill diagonal references
    for i in range(n_bots):
        win_rate_matrix[i, i] = 0.5
        ci_matrix[i, i] = 0.0
        pts_matrix[i, i] = 60.0
        gpts_matrix[i, i] = 0.0

    return win_rate_matrix, ci_matrix, pts_matrix, gpts_matrix


def plot_tournament_heatmap(
    win_rates: np.ndarray,
    ci_margins: np.ndarray,
    bot_names: list[str],
    save_path: str,
) -> None:
    """Generate and save a beautiful heatmap of win rates with confidence intervals."""
    try:
        plt.figure(figsize=(10, 8))

        annots = np.empty_like(win_rates, dtype=object)
        for i in range(len(bot_names)):
            for j in range(len(bot_names)):
                if i == j:
                    annots[i, j] = "50.0%\n(Ref)"
                else:
                    annots[i, j] = (
                        f"{win_rates[i, j] * 100:.1f}%\n± {ci_margins[i, j] * 100:.1f}%"
                    )

        sns.set_theme(style="white")
        cmap = sns.diverging_palette(15, 130, as_cmap=True)

        ax = sns.heatmap(
            win_rates * 100,
            annot=annots,
            fmt="",
            cmap=cmap,
            vmin=10,
            vmax=90,
            cbar_kws={"label": "Win Rate (%)"},
            linewidths=1.5,
            linecolor="white",
            square=True,
            annot_kws={"size": 11, "weight": "bold", "fontname": "DejaVu Sans"},
        )

        ax.set_xticklabels(
            bot_names, rotation=45, ha="right", fontname="DejaVu Sans", weight="bold"
        )
        ax.set_yticklabels(bot_names, rotation=0, fontname="DejaVu Sans", weight="bold")

        plt.title(
            "Sueca Tournament Performance Matrix\n(Row Bot Win Rate vs Column Bot, showing 95% Binomial CI)",
            fontsize=14,
            fontname="DejaVu Sans",
            weight="bold",
            pad=20,
        )
        plt.tight_layout()

        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Generated performance heatmap at {save_path}")
    except Exception as e:
        print(f"Warning: Could not plot heatmap: {e}")


def write_csv_report(
    win_rates: np.ndarray,
    ci_margins: np.ndarray,
    pts: np.ndarray,
    gpts: np.ndarray,
    bot_names: list[str],
    save_path: str,
) -> None:
    """Write the complete tournament report to a CSV file."""
    n = len(bot_names)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Candidate Bot",
                "Opponent Bot",
                "Win Rate (%)",
                "CI Margin (%)",
                "Avg Card Pts",
                "Avg Game Pts",
            ]
        )
        for i in range(n):
            for j in range(n):
                writer.writerow(
                    [
                        bot_names[i],
                        bot_names[j],
                        f"{win_rates[i, j] * 100:.2f}",
                        f"{ci_margins[i, j] * 100:.2f}",
                        f"{pts[i, j]:.2f}",
                        f"{gpts[i, j]:.2f}",
                    ]
                )
    print(f"Saved CSV report to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Sueca bot tournament benchmark")
    parser.add_argument(
        "--deals",
        type=int,
        default=200,
        help="Number of duplicate deals (default: 200)",
    )
    parser.add_argument(
        "--genome",
        type=str,
        default=None,
        help="Path to champion WANN genome (searches checkpoints/ for latest if omitted)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed offset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save artifacts (default: derived from genome path)",
    )
    parser.add_argument(
        "--no-mp",
        action="store_true",
        help="Disable multiprocessing (ignored, as Rust parallelizes natively)",
    )
    parser.add_argument(
        "--pimc-worlds",
        type=int,
        default=80,
        help="Number of worlds for PIMC bot (default: 80)",
    )
    parser.add_argument(
        "--pimc-depth",
        type=int,
        default=4,
        help="Search depth for PIMC bot (default: 4)",
    )

    args = parser.parse_args()

    # Find latest genome if not specified
    if args.genome is None:
        checkpoints_dir = "checkpoints"
        if os.path.isdir(checkpoints_dir):
            run_dirs = sorted(
                [
                    d
                    for d in os.listdir(checkpoints_dir)
                    if os.path.isdir(os.path.join(checkpoints_dir, d))
                ],
                reverse=True,
            )
            for d in run_dirs:
                candidate = os.path.join(
                    checkpoints_dir, d, "genomes", "best_genome_final.json"
                )
                if os.path.exists(candidate):
                    args.genome = candidate
                    print(f"Auto-detected genome: {args.genome}")
                    break
            if args.genome is None:
                print(
                    "Error: No best_genome_final.json found in any checkpoint run folder."
                )
                sys.exit(1)
        else:
            print("Error: No checkpoints directory found.")
            sys.exit(1)

    # Derive output dir from genome path if not specified
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.genome)
        print(f"Output directory (derived): {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading WANN Champion from {args.genome}...")
    if not os.path.exists(args.genome):
        print(f"Error: WANN genome checkpoint not found at {args.genome}")
        sys.exit(1)

    import sueca_solver

    network = sueca_solver.load_genome(args.genome)

    # Bot representation: (bot_type_int, network_or_None)
    bots = {
        "RandomBot": (0, None),
        "HeuristicBot": (1, None),
        "PIMCBot": (-1, None),
        "WANN (Champion)": (2, network),
    }

    print(f"Starting Sueca Benchmarking Tournament (N={args.deals} deals)...")
    win_rates, ci_margins, pts, gpts = run_tournament(
        bots,
        n_deals=args.deals,
        base_seed=args.seed,
        pimc_worlds=args.pimc_worlds,
        pimc_depth=args.pimc_depth,
    )

    bot_names = list(bots.keys())
    print("\n" + "=" * 80)
    print(" SUECA BOT TOURNAMENT BENCHMARK RESULTS ".center(80, "="))
    print("=" * 80)
    print(
        f"{'Candidate 👇 / Opponent 👉':<25} | "
        + " | ".join(f"{name[:12]:^12}" for name in bot_names)
    )
    print("-" * 80)
    for i, name_a in enumerate(bot_names):
        cells = []
        for j, name_b in enumerate(bot_names):
            if i == j:
                cells.append(f"{'50.0% (Ref)':^12}")
            else:
                cells.append(
                    f"{win_rates[i, j] * 100:5.1f}% ±{ci_margins[i, j] * 100:4.1f}%"
                )
        print(f"{name_a:<25} | " + " | ".join(cells))
    print("=" * 80)

    heatmap_path = os.path.join(args.output_dir, "tournament_matrix.png")
    plot_tournament_heatmap(win_rates, ci_margins, bot_names, heatmap_path)

    csv_path = os.path.join(args.output_dir, "tournament_report.csv")
    write_csv_report(win_rates, ci_margins, pts, gpts, bot_names, csv_path)

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
