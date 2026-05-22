# src/measure_snr.py
"""Measure the signal-to-noise ratio of delta-fitness scoring.

Compares HeuristicBot vs RandomBot and seed strategies vs RandomBot on the
same deals, reporting mean delta, std, SE, and SNR in both game points and
raw card points. All simulations run in Rust.
"""

from __future__ import annotations

import os
import sys
from enum import IntEnum
from typing import NamedTuple, Any
from dataclasses import dataclass

import numpy as np


from src.compat import (
    Suit,
    Rank,
    Card,
    DealRecord,
    generate_deals,
    RustDealCompat,
)


# ─── Seed Strategies Definition ─────────────────────────────────────────────

SEED_STRATEGIES = [
    ("aggressive", [(21, 24, 1)]),
    ("take_cheaply", [(21, 23, 1)]),
    ("partner_aware", [(7, 22, 1), (7, 24, -1)]),
    ("trump_cutter", [(0, 26, -1), (1, 26, 1)]),
    ("feeder", [(7, 25, 1)]),
]


def create_seed_network(conns: list[tuple[int, int, int]]) -> Any:
    """Construct a PyWannNetwork directly using PyO3 from a list of connections."""
    import sueca_solver

    node_ids = list(range(27))
    node_types = [0] * 27
    node_activations = [0] * 27
    node_aggregations = [0] * 27

    conn_srcs = [src for src, dst, sign in conns]
    conn_dsts = [dst for src, dst, sign in conns]
    conn_signs = [sign for src, dst, sign in conns]
    conn_enableds = [True] * len(conns)

    return sueca_solver.PyWannNetwork.from_genome(
        node_ids,
        node_types,
        node_activations,
        node_aggregations,
        conn_srcs,
        conn_dsts,
        conn_signs,
        conn_enableds,
    )


# ─── Measure Delta delegating to Rust ───────────────────────────────────────


def measure_delta(
    bot_a: tuple[int, Any],
    bot_b: tuple[int, Any],
    n_deals: int = 64,
    seed: int = 42,
):
    """Play n_deals × 4 rotations, return per-game deltas in both metrics."""
    import sueca_solver

    deals = generate_deals(gen=0, n_deals=n_deals, base_seed=seed)
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
    sweep_weights = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

    gp_deltas, cp_deltas = sueca_solver.run_snr_matchup_rust(
        rust_deals,
        bot_a_type,
        bot_a_network,
        bot_b_type,
        bot_b_network,
        sweep_weights,
        seed,
    )

    return np.array(gp_deltas), np.array(cp_deltas)


# ─── Reporting Helper ───────────────────────────────────────────────────────


def report(name: str, gp_deltas: np.ndarray, cp_deltas: np.ndarray):
    n = len(gp_deltas)
    print(f"\n{'='*60}")
    print(f"  {name}  ({n} games)")
    print(f"{'='*60}")

    for label, d in [("Game Points", gp_deltas), ("Card Points", cp_deltas)]:
        mean = np.mean(d)
        std = np.std(d, ddof=1)
        se = std / np.sqrt(n)
        snr = abs(mean) / se if se > 0 else float("inf")
        print(f"\n  {label}:")
        print(f"    Mean Δ = {mean:+.4f}")
        print(f"    Std    = {std:.4f}")
        print(f"    SE     = {se:.4f}")
        print(f"    SNR    = {snr:.2f}σ")
        print(f"    Range  = [{np.min(d)}, {np.max(d)}]")

        # Distribution of values
        unique, counts = np.unique(d, return_counts=True)
        if len(unique) <= 20:
            print("    Distribution:")
            for v, c in sorted(zip(unique, counts)):
                pct = 100 * c / n
                bar = "#" * int(pct / 2)
                print(f"      {v:+6.0f}: {c:4d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    N_DEALS = 64  # 64 deals × 4 rotations = 256 games

    print("Measuring delta-fitness signal-to-noise ratio...")
    print(
        f"Configuration: {N_DEALS} deals × 4 rotations = {N_DEALS * 4} games per comparison"
    )

    # Bot representation: (bot_type_int, network_or_None)
    heuristic_bot = (1, None)
    random_bot = (0, None)

    # --- HeuristicBot vs RandomBot ---
    gp, cp = measure_delta(heuristic_bot, random_bot, n_deals=N_DEALS)
    report("HeuristicBot vs RandomBot", gp, cp)

    # --- Seed strategies vs RandomBot ---
    for name, conns in SEED_STRATEGIES[:5]:  # top 5 seeds
        network = create_seed_network(conns)
        bot = (2, network)
        gp, cp = measure_delta(bot, random_bot, n_deals=N_DEALS)
        report(f"Seed '{name}' vs RandomBot", gp, cp)

    # --- RandomBot vs RandomBot (noise floor) ---
    gp, cp = measure_delta(random_bot, random_bot, n_deals=N_DEALS)
    report("RandomBot vs RandomBot (noise floor)", gp, cp)
