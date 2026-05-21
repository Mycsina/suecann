"""Card primitives for the Portuguese trick-taking game Sueca.

Sueca uses a 40-card deck (standard 52 minus 8s, 9s, 10s).
Rank ordering (high to low): A > 7 > K > J > Q > 6 > 5 > 4 > 3 > 2
Point values: A=11, 7=10, K=4, J=3, Q=2, rest=0.  Total = 120.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import numpy as np


class Suit(IntEnum):
    """Card suits."""

    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3


class Rank(IntEnum):
    """Card ranks in Sueca power ordering (0 = weakest, 9 = strongest)."""

    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    QUEEN = 5
    JACK = 6
    KING = 7
    SEVEN = 8  # Manilha — second strongest
    ACE = 9


class Card(NamedTuple):
    """An immutable card with suit and rank."""

    suit: Suit
    rank: Rank

    def __repr__(self) -> str:
        rank_symbols = {
            Rank.TWO: "2",
            Rank.THREE: "3",
            Rank.FOUR: "4",
            Rank.FIVE: "5",
            Rank.SIX: "6",
            Rank.QUEEN: "Q",
            Rank.JACK: "J",
            Rank.KING: "K",
            Rank.SEVEN: "7",
            Rank.ACE: "A",
        }
        suit_symbols = {
            Suit.HEARTS: "♥",
            Suit.DIAMONDS: "♦",
            Suit.CLUBS: "♣",
            Suit.SPADES: "♠",
        }
        return f"{rank_symbols[self.rank]}{suit_symbols[self.suit]}"


# Point value for each rank.
POINT_VALUES: dict[Rank, int] = {
    Rank.ACE: 11,
    Rank.SEVEN: 10,
    Rank.KING: 4,
    Rank.JACK: 3,
    Rank.QUEEN: 2,
    Rank.SIX: 0,
    Rank.FIVE: 0,
    Rank.FOUR: 0,
    Rank.THREE: 0,
    Rank.TWO: 0,
}

TOTAL_GAME_POINTS = 120

# Maximum points in a single 4-card trick.
# Theoretical max: e.g. A(11) + 7(10) + A(11) + 7(10) = 42, or mixed.
# User-specified ceiling for normalization purposes.
MAX_TRICK_POINTS = 44


def card_points(card: Card) -> int:
    """Return the point value of a card."""
    return POINT_VALUES[card.rank]


def build_deck() -> list[Card]:
    """Return a full 40-card Sueca deck, sorted by suit then rank."""
    return [Card(suit, rank) for suit in Suit for rank in Rank]


def deal(rng: np.random.Generator) -> list[list[Card]]:
    """Shuffle and deal 10 cards to each of 4 players.

    Args:
        rng: A NumPy random generator for reproducibility.

    Returns:
        A list of 4 hands, each a list of 10 Cards.
    """
    deck = build_deck()
    indices = rng.permutation(len(deck))
    shuffled = [deck[i] for i in indices]
    return [shuffled[i * 10 : (i + 1) * 10] for i in range(4)]


def choose_trump(rng: np.random.Generator) -> Suit:
    """Randomly choose a trump suit (simulating the dealer's revealed card)."""
    return Suit(rng.integers(0, 4))
