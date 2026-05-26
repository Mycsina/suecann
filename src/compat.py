# src/compat.py
"""Centralized structures and helpers for Sueca game representations.

Maintains compatibility with Python scripts and PyO3 bindings.
"""

from enum import IntEnum
from typing import NamedTuple
from dataclasses import dataclass
import numpy as np


class Suit(IntEnum):
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3


class Rank(IntEnum):
    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    QUEEN = 5
    JACK = 6
    KING = 7
    SEVEN = 8
    ACE = 9


class Card(NamedTuple):
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


def build_deck() -> list[Card]:
    return [Card(suit, rank) for suit in Suit for rank in Rank]


def deal(rng: np.random.Generator) -> list[list[Card]]:
    deck = build_deck()
    indices = rng.permutation(len(deck))
    shuffled = [deck[i] for i in indices]
    return [shuffled[i * 10 : (i + 1) * 10] for i in range(4)]


@dataclass
class DealRecord:
    hands: list[list[Card]]
    trump: Suit
    seed: int


def generate_deals(gen: int, n_deals: int = 16, base_seed: int = 0) -> list[DealRecord]:
    seed = base_seed + gen
    deals: list[DealRecord] = []
    for i in range(n_deals):
        deal_seed = seed * 1000 + i
        deal_rng = np.random.default_rng(deal_seed)
        hands = deal(deal_rng)
        trump = Suit(deal_rng.integers(0, 4))
        deals.append(DealRecord(hands=hands, trump=trump, seed=deal_seed))
    return deals


class RustDealCompat:
    def __init__(self, hands: list[list[int]], trump: int, seed: int):
        self.hands = hands
        self.trump = trump
        self.seed = seed
