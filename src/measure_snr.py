"""Measure the signal-to-noise ratio of delta-fitness scoring.

Compares HeuristicBot vs RandomBot and seed strategies vs RandomBot on the
same deals, reporting mean delta, std, SE, and SNR in both game points and
raw card points.
"""
import numpy as np
from src.baselines.heuristic_bot import HeuristicBot
from src.baselines.random_bot import RandomBot
from src.engine.duplicate_loop import (
    DealRecord,
    generate_deals,
    play_game_with_bots,
    rotate_seats,
)
from src.engine.cards import Suit
from src.wann.population import SEED_STRATEGIES, _create_seed_genome
from src.oracle.fitness import WannBotSweep


def measure_delta(
    bot_a,  # "genome" bot
    bot_b,  # baseline bot
    n_deals: int = 64,
    seed: int = 42,
):
    """Play n_deals × 4 rotations, return per-game deltas in both metrics."""
    deals = generate_deals(gen=0, n_deals=n_deals, base_seed=seed)
    game_point_deltas = []
    card_point_deltas = []

    for deal in deals:
        for rot in range(4):
            rotated = rotate_seats(deal.hands, rot)
            game_seed = seed + deal.seed + rot * 10000

            # Bot A plays seat 0
            bots_a = [bot_a, RandomBot(), RandomBot(), RandomBot()]
            # Use a copy of rotated hands for each game
            result_a = play_game_with_bots(
                [list(h) for h in rotated], deal.trump, bots_a,
                first_player=rot % 4, seed=game_seed,
            )

            # Bot B plays seat 0 (same cards, same opponents, same seed)
            bots_b = [bot_b, RandomBot(), RandomBot(), RandomBot()]
            result_b = play_game_with_bots(
                [list(h) for h in rotated], deal.trump, bots_b,
                first_player=rot % 4, seed=game_seed,
            )

            game_point_deltas.append(result_a.game_points[0] - result_b.game_points[0])
            card_point_deltas.append(result_a.team_scores[0] - result_b.team_scores[0])

    return np.array(game_point_deltas), np.array(card_point_deltas)


def report(name: str, gp_deltas: np.ndarray, cp_deltas: np.ndarray):
    n = len(gp_deltas)
    print(f"\n{'='*60}")
    print(f"  {name}  ({n} games)")
    print(f"{'='*60}")

    for label, d in [("Game Points", gp_deltas), ("Card Points", cp_deltas)]:
        mean = np.mean(d)
        std = np.std(d, ddof=1)
        se = std / np.sqrt(n)
        snr = abs(mean) / se if se > 0 else float('inf')
        print(f"\n  {label}:")
        print(f"    Mean Δ = {mean:+.4f}")
        print(f"    Std    = {std:.4f}")
        print(f"    SE     = {se:.4f}")
        print(f"    SNR    = {snr:.2f}σ")
        print(f"    Range  = [{np.min(d)}, {np.max(d)}]")

        # Distribution of values
        unique, counts = np.unique(d, return_counts=True)
        if len(unique) <= 20:
            print(f"    Distribution:")
            for v, c in sorted(zip(unique, counts)):
                pct = 100 * c / n
                bar = '#' * int(pct / 2)
                print(f"      {v:+6.0f}: {c:4d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    N_DEALS = 64  # 64 deals × 4 rotations = 256 games

    print("Measuring delta-fitness signal-to-noise ratio...")
    print(f"Configuration: {N_DEALS} deals × 4 rotations = {N_DEALS * 4} games per comparison")

    # --- HeuristicBot vs RandomBot ---
    gp, cp = measure_delta(HeuristicBot(), RandomBot(), n_deals=N_DEALS)
    report("HeuristicBot vs RandomBot", gp, cp)

    # --- Seed strategies vs RandomBot ---
    for name, conns in SEED_STRATEGIES[:5]:  # top 5 seeds
        genome = _create_seed_genome(conns)
        bot = WannBotSweep(genome)
        gp, cp = measure_delta(bot, RandomBot(), n_deals=N_DEALS)
        report(f"Seed '{name}' vs RandomBot", gp, cp)

    # --- RandomBot vs RandomBot (noise floor) ---
    gp, cp = measure_delta(RandomBot(), RandomBot(), n_deals=N_DEALS)
    report("RandomBot vs RandomBot (noise floor)", gp, cp)
