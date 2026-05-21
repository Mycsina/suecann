"""Fitness evaluation — delta-fitness with Common Random Numbers and weight sweep.

Delta-fitness: each genome is compared against a baseline bot on the exact same
deal/seat/opponents. Fitness = mean(genome_points - baseline_points) + oracle_tax.
This eliminates deal-luck variance (AIVAT / IMP-style scoring).

Weight sweep: each genome is evaluated at W ∈ {0.5, 1.0, 2.0} and the mean
output is used for action selection, following the original WANN paper.
"""

from __future__ import annotations

import numpy as np

from src.engine.belief_state import encode
from src.engine.cards import Card
from src.engine.duplicate_loop import (
    Bot,
    DealRecord,
    GameResultFull,
    evaluate_genome_delta_on_deals,
    generate_deals,
    play_game_with_bots,
)
from src.engine.sueca_engine import SuecaGame
from src.oracle.legal_system import select_card_from_outputs
from src.wann.genome import Genome
from src.wann.network import WannNetwork


# Default shared weight values for weight sweep.
DEFAULT_WEIGHT_SWEEP = [0.5, 1.0, 2.0]


class WannBot:
    """A bot that uses a WANN genome to select cards (weight sweep)."""

    def __init__(self, genome: Genome):
        self.genome = genome
        self.network = WannNetwork(genome)
        self._illegal_count: int = 0

    def select_card(
        self, game: SuecaGame, seat: int, rng: np.random.Generator
    ) -> Card:
        state = game.get_visible_state(seat)
        belief = encode(state)

        # Weight sweep: average outputs across W ∈ {0.5, 1.0, 2.0}.
        _, mean_output = self.network.forward_weight_sweep(belief)

        card, was_illegal = select_card_from_outputs(mean_output, game, seat, rng)
        if was_illegal:
            self._illegal_count += 1
        return card

    def reset(self) -> None:
        self._illegal_count = 0


class WannBotSweep:
    """WannBot that uses the full weight sweep for evaluation (true weight-agnostic)."""

    def __init__(self, genome: Genome, weights: list[float] | None = None):
        self.genome = genome
        self.network = WannNetwork(genome)
        self.weights = weights or DEFAULT_WEIGHT_SWEEP
        self._illegal_count: int = 0

    def select_card(
        self, game: SuecaGame, seat: int, rng: np.random.Generator
    ) -> Card:
        state = game.get_visible_state(seat)
        belief = encode(state)
        _, mean_output = self.network.forward_weight_sweep(belief, self.weights)
        card, was_illegal = select_card_from_outputs(mean_output, game, seat, rng)
        if was_illegal:
            self._illegal_count += 1
        return card

    def reset(self) -> None:
        self._illegal_count = 0


class WannBotSingleWeight:
    """WannBot variant that uses a single shared weight (no sweep). Faster for evaluations."""

    def __init__(self, genome: Genome, weight: float = 1.0):
        self.genome = genome
        self.network = WannNetwork(genome)
        self.weight = weight
        self._illegal_count: int = 0

    def select_card(
        self, game: SuecaGame, seat: int, rng: np.random.Generator
    ) -> Card:
        state = game.get_visible_state(seat)
        belief = encode(state)
        outputs = self.network.forward(belief, self.weight)
        card, was_illegal = select_card_from_outputs(outputs, game, seat, rng)
        if was_illegal:
            self._illegal_count += 1
        return card

    def reset(self) -> None:
        self._illegal_count = 0


def oracle_tax_penalty(
    generation: int, curriculum_gens: int = 50
) -> float:
    """Oracle Tax penalty per illegal intent. Ramps from -0.25 to -3.0."""
    progress = min(generation / curriculum_gens, 1.0)
    return -(0.25 + 2.75 * progress)


def evaluate_genome(
    genome: Genome,
    deals: list[DealRecord],
    opponent_bots: list[Bot],
    baseline_bot: Bot,
    generation: int = 0,
    curriculum_gens: int = 50,
    base_seed: int = 0,
    sweep_weights: list[float] | None = None,
) -> tuple[float, float, float, int, int]:
    """Evaluate a single genome using delta-fitness scoring.

    Plays each deal twice: once with the genome bot, once with the baseline bot
    in the exact same seat/cards/opponents. Fitness = mean(delta) + oracle_tax.

    Args:
        genome: The WANN genome to evaluate.
        deals: Duplicate deals for this generation.
        opponent_bots: 3 bots [partner, opp1, opp2].
        baseline_bot: Bot to compare against (RandomBot for variance reduction).
        generation: Current generation (for Oracle Tax warm-up).
        curriculum_gens: Generations over which to ramp Oracle Tax.
        base_seed: Base RNG seed.
        sweep_weights: W values for weight sweep (default: [0.5, 1.0, 2.0]).

    Returns:
        (fitness, avg_delta, avg_game_points, total_illegal, total_games).
    """
    tax_per = oracle_tax_penalty(generation, curriculum_gens)
    weights = sweep_weights or DEFAULT_WEIGHT_SWEEP
    genome_bot = WannBotSweep(genome, weights=weights)

    all_results = evaluate_genome_delta_on_deals(
        deals, genome_bot, baseline_bot, opponent_bots, base_seed=base_seed
    )

    total_delta = 0.0
    total_game_pts = 0.0
    total_illegal = 0
    n_games = 0

    for genome_results, baseline_results in all_results:
        for g_result, b_result in zip(genome_results, baseline_results):
            delta = g_result.game_points[0] - b_result.game_points[0]
            total_delta += delta
            total_game_pts += g_result.game_points[0]
            total_illegal += g_result.illegal_count
            n_games += 1

    avg_delta = total_delta / n_games if n_games > 0 else 0.0
    avg_game_pts = total_game_pts / n_games if n_games > 0 else 0.0

    # Fitness = average delta + oracle tax * illegal rate.
    illegal_rate = total_illegal / n_games if n_games > 0 else 0.0
    fitness = avg_delta + tax_per * illegal_rate

    return fitness, avg_delta, avg_game_pts, total_illegal, n_games


def _evaluate_single_genome_worker(args) -> tuple[float, float, int]:
    """Helper worker to evaluate a single genome for multiprocessing."""
    (
        genome,
        deals,
        opponent_bots,
        baseline_bot,
        generation,
        curriculum_gens,
        seed,
        sweep_weights,
    ) = args
    fit, avg_delta, _, illegal, _ = evaluate_genome(
        genome,
        deals,
        opponent_bots,
        baseline_bot,
        generation=generation,
        curriculum_gens=curriculum_gens,
        base_seed=seed,
        sweep_weights=sweep_weights,
    )
    return fit, avg_delta, illegal


def evaluate_population(
    genomes: list[Genome],
    deals: list[DealRecord],
    opponent_bots: list[Bot],
    baseline_bot: Bot,
    generation: int = 0,
    curriculum_gens: int = 50,
    base_seed: int = 0,
    sweep_weights: list[float] | None = None,
    parallel: bool = True,
) -> tuple[list[float], list[float], int]:
    """Evaluate all genomes in a population, optionally in parallel.

    Returns:
        (fitnesses, deltas, total_illegal_count).
    """
    weights = sweep_weights or DEFAULT_WEIGHT_SWEEP
    args_list = [
        (
            genome,
            deals,
            opponent_bots,
            baseline_bot,
            generation,
            curriculum_gens,
            base_seed + i,
            weights,
        )
        for i, genome in enumerate(genomes)
    ]

    fitnesses = []
    deltas = []
    total_illegal = 0

    if parallel and len(genomes) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        num_workers = min(multiprocessing.cpu_count(), len(genomes))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_evaluate_single_genome_worker, args_list))
            for fit, delta, illegal in results:
                fitnesses.append(fit)
                deltas.append(delta)
                total_illegal += illegal
    else:
        for args in args_list:
            fit, delta, illegal = _evaluate_single_genome_worker(args)
            fitnesses.append(fit)
            deltas.append(delta)
            total_illegal += illegal

    return fitnesses, deltas, total_illegal
