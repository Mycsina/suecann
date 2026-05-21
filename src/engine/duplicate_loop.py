"""Duplicate deal generator and symmetric seat-rotation evaluator.

For each generation, 16 deals are generated (seeded by gen for reproducibility
but different across generations to prevent overfitting). Each deal is played
4 times, rotating the starting player so every genome sees all seats.

Delta-fitness: each deal is also played with a baseline bot in the same seat
to compute a variance-reduced fitness signal (Common Random Numbers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from src.engine.cards import Card, Suit, deal
from src.engine.sueca_engine import SuecaGame, GameResult


class Bot(Protocol):
    """Protocol for a Sueca-playing agent."""

    def select_card(self, game: SuecaGame, seat: int, rng: np.random.Generator) -> Card:
        """Given a game state and seat, return the card to play."""
        ...

    def reset(self) -> None:
        """Reset any per-game state."""
        ...


@dataclass
class DealRecord:
    """A deal with trump suit and hands for 4 players."""

    hands: list[list[Card]]
    trump: Suit
    seed: int


def generate_deals(gen: int, n_deals: int = 16, base_seed: int = 0) -> list[DealRecord]:
    """Generate duplicate deals for a generation.

    Args:
        gen: Generation number (used in seed for diversity across gens).
        n_deals: Number of deals to generate.
        base_seed: Base seed offset.

    Returns:
        List of DealRecord.
    """
    seed = base_seed + gen
    rng = np.random.default_rng(seed)
    deals: list[DealRecord] = []

    for i in range(n_deals):
        deal_seed = seed * 1000 + i
        deal_rng = np.random.default_rng(deal_seed)
        hands = deal(deal_rng)
        trump = Suit(deal_rng.integers(0, 4))
        deals.append(DealRecord(hands=hands, trump=trump, seed=deal_seed))

    return deals


@dataclass
class GameResultFull:
    """Extended game result with deal info."""

    team_scores: tuple[int, int]
    game_points: tuple[int, int]
    deal_seed: int
    seat_rotation: int  # 0 = original, 1/2/3 = rotated
    illegal_count: int = 0


def rotate_seats(hands: list[list[Card]], rotation: int) -> list[list[Card]]:
    """Rotate hands so that each genome plays from a different seat.

    rotation=0: genome plays from seat 0, 1, 2, 3 (original)
    rotation=1: genome plays from seat 1, 2, 3, 0
    etc.
    """
    if rotation == 0:
        return [list(h) for h in hands]
    n = len(hands)
    return [list(hands[(i - rotation) % n]) for i in range(n)]


def _rotate_first_player(first_player: int, rotation: int) -> int:
    """Adjust first player based on rotation."""
    return (first_player + rotation) % 4


def play_game_with_bots(
    hands: list[list[Card]],
    trump: Suit,
    bots: list[Bot],
    first_player: int = 0,
    seed: int = 0,
) -> GameResult:
    """Play a single game with the given bots.

    Args:
        hands: 4 hands of 10 cards.
        trump: Trump suit.
        bots: 4 Bot instances (one per seat).
        first_player: Seat that leads the first trick.
        seed: RNG seed for tie-breaking.

    Returns:
        GameResult with final scores.
    """
    rng = np.random.default_rng(seed)
    for bot in bots:
        bot.reset()

    game = SuecaGame(hands=hands, trump=trump, first_player=first_player)

    while not game.is_terminal():
        seat = game.current_player
        card = bots[seat].select_card(game, seat, rng)
        game.play_card(seat, card)

    return game.get_result()


def evaluate_genome_on_deals(
    deals: list[DealRecord],
    genome_bot: Bot,
    opponent_bots: list[Bot],
    first_player: int = 0,
    base_seed: int = 0,
) -> list[tuple[GameResultFull, GameResultFull, GameResultFull, GameResultFull]]:
    """Evaluate a genome bot against opponents on all deals with 4 seat rotations.

    Returns one tuple per deal, each tuple containing 4 GameResultFull
    (one per seat rotation). Total: n_deals × 4 = 64 games per genome.
    """
    results: list[
        tuple[GameResultFull, GameResultFull, GameResultFull, GameResultFull]
    ] = []

    for di, deal in enumerate(deals):
        rotations: list[GameResultFull] = []
        for rot in range(4):
            rotated_hands = rotate_seats(deal.hands, rot)
            adjusted_first = _rotate_first_player(first_player, rot)

            # Build bots array: genome_bot always at seat rot (the seat being evaluated).
            # Actually: we want the genome to experience each seat.
            # After rotation, the genome's seat becomes the "primary" seat 0 in rotated hands.
            # So the genome bot should be at seat 0 always in the rotated deal.
            game_bots: list[Bot] = [None] * 4  # type: ignore
            game_bots[0] = genome_bot  # genome always plays seat 0 after rotation

            # Fill opponents at seats 1, 2, 3.
            partner_idx = 2  # partner sits opposite
            opp1_idx, opp2_idx = 1, 3

            game_bots[partner_idx] = opponent_bots[0]  # partner
            game_bots[opp1_idx] = opponent_bots[1]  # opponent 1
            game_bots[opp2_idx] = opponent_bots[2]  # opponent 2

            game_seed = base_seed + deal.seed + rot * 10000
            result = play_game_with_bots(
                rotated_hands,
                deal.trump,
                game_bots,
                first_player=adjusted_first,
                seed=game_seed,
            )

            # Record result from the genome's perspective.
            # After rotation, team 0 and 2 are the genome's team.
            # Team scores: [team_02, team_13] where team_02 = seats 0&2.
            # The genome is at seat 0, so its team is team_02.
            rotations.append(
                GameResultFull(
                    team_scores=result.team_scores,
                    game_points=result.game_points,
                    deal_seed=deal.seed,
                    seat_rotation=rot,
                    illegal_count=getattr(genome_bot, "_illegal_count", 0),
                )
            )

        results.append((rotations[0], rotations[1], rotations[2], rotations[3]))

    return results


def evaluate_genome_delta_on_deals(
    deals: list[DealRecord],
    genome_bot: Bot,
    baseline_bot: Bot,
    opponent_bots: list[Bot],
    first_player: int = 0,
    base_seed: int = 0,
) -> list[tuple[list[GameResultFull], list[GameResultFull]]]:
    """Evaluate a genome vs a baseline on the same deals (Common Random Numbers).

    For each deal × 4 rotations, plays the game twice:
      1. genome_bot at seat 0 (+ opponents)
      2. baseline_bot at seat 0 (+ same opponents, same cards)

    Returns one (genome_results, baseline_results) pair per deal,
    each containing 4 GameResultFull (one per rotation).
    """
    results: list[tuple[list[GameResultFull], list[GameResultFull]]] = []

    for di, deal_rec in enumerate(deals):
        genome_rotations: list[GameResultFull] = []
        baseline_rotations: list[GameResultFull] = []

        for rot in range(4):
            rotated_hands = rotate_seats(deal_rec.hands, rot)
            adjusted_first = _rotate_first_player(first_player, rot)
            game_seed = base_seed + deal_rec.seed + rot * 10000

            # --- Play with genome bot ---
            game_bots_genome: list[Bot] = [None] * 4  # type: ignore
            game_bots_genome[0] = genome_bot
            game_bots_genome[2] = opponent_bots[0]  # partner
            game_bots_genome[1] = opponent_bots[1]  # opp1
            game_bots_genome[3] = opponent_bots[2]  # opp2

            result_genome = play_game_with_bots(
                [list(h) for h in rotated_hands],
                deal_rec.trump,
                game_bots_genome,
                first_player=adjusted_first,
                seed=game_seed,
            )

            genome_rotations.append(
                GameResultFull(
                    team_scores=result_genome.team_scores,
                    game_points=result_genome.game_points,
                    deal_seed=deal_rec.seed,
                    seat_rotation=rot,
                    illegal_count=getattr(genome_bot, "_illegal_count", 0),
                )
            )

            # --- Play with baseline bot (same cards, same opponents, same seed) ---
            game_bots_baseline: list[Bot] = [None] * 4  # type: ignore
            game_bots_baseline[0] = baseline_bot
            game_bots_baseline[2] = opponent_bots[0]  # partner
            game_bots_baseline[1] = opponent_bots[1]  # opp1
            game_bots_baseline[3] = opponent_bots[2]  # opp2

            result_baseline = play_game_with_bots(
                [list(h) for h in rotated_hands],
                deal_rec.trump,
                game_bots_baseline,
                first_player=adjusted_first,
                seed=game_seed,
            )

            baseline_rotations.append(
                GameResultFull(
                    team_scores=result_baseline.team_scores,
                    game_points=result_baseline.game_points,
                    deal_seed=deal_rec.seed,
                    seat_rotation=rot,
                    illegal_count=0,
                )
            )

        results.append((genome_rotations, baseline_rotations))

    return results
