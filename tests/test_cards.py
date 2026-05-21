"""Thorough tests for src/engine/cards.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.engine.cards import (
    Card,
    Rank,
    Suit,
    build_deck,
    card_points,
    choose_trump,
    deal,
    POINT_VALUES,
    TOTAL_GAME_POINTS,
)


# ─── Deck composition ──────────────────────────────────────────────────────

class TestDeckComposition:
    """Verify the 40-card deck is correctly constructed."""

    def test_deck_has_40_cards(self):
        deck = build_deck()
        assert len(deck) == 40

    def test_deck_cards_are_unique(self):
        deck = build_deck()
        assert len(set(deck)) == 40

    def test_deck_has_4_suits(self):
        deck = build_deck()
        suits = {c.suit for c in deck}
        assert suits == {Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES}

    def test_deck_has_10_ranks_per_suit(self):
        deck = build_deck()
        for suit in Suit:
            suit_cards = [c for c in deck if c.suit == suit]
            assert len(suit_cards) == 10

    def test_deck_has_all_ranks_per_suit(self):
        deck = build_deck()
        for suit in Suit:
            ranks = {c.rank for c in deck if c.suit == suit}
            assert ranks == set(Rank)

    def test_no_eights_nines_tens(self):
        """Sueca deck excludes 8, 9, 10 — these aren't in Rank enum."""
        # Verify that the Rank enum has exactly 10 values (2-7, Q, J, K, A).
        assert len(Rank) == 10


# ─── Point values ───────────────────────────────────────────────────────────

class TestPointValues:
    """Verify Sueca point values are correct."""

    def test_total_points_is_120(self):
        """Sum of all 40 cards' point values must be 120."""
        deck = build_deck()
        total = sum(card_points(c) for c in deck)
        assert total == 120

    def test_total_game_points_constant(self):
        assert TOTAL_GAME_POINTS == 120

    def test_ace_is_11(self):
        assert POINT_VALUES[Rank.ACE] == 11

    def test_seven_is_10(self):
        assert POINT_VALUES[Rank.SEVEN] == 10

    def test_king_is_4(self):
        assert POINT_VALUES[Rank.KING] == 4

    def test_jack_is_3(self):
        assert POINT_VALUES[Rank.JACK] == 3

    def test_queen_is_2(self):
        assert POINT_VALUES[Rank.QUEEN] == 2

    @pytest.mark.parametrize("rank", [Rank.SIX, Rank.FIVE, Rank.FOUR, Rank.THREE, Rank.TWO])
    def test_zero_point_cards(self, rank):
        assert POINT_VALUES[rank] == 0

    def test_card_points_function(self):
        assert card_points(Card(Suit.HEARTS, Rank.ACE)) == 11
        assert card_points(Card(Suit.SPADES, Rank.TWO)) == 0

    def test_points_per_suit_is_30(self):
        """Each suit has A(11) + 7(10) + K(4) + J(3) + Q(2) = 30 points."""
        deck = build_deck()
        for suit in Suit:
            suit_pts = sum(card_points(c) for c in deck if c.suit == suit)
            assert suit_pts == 30


# ─── Rank ordering ──────────────────────────────────────────────────────────

class TestRankOrdering:
    """Verify Sueca-specific rank ordering: A > 7 > K > J > Q > 6 > 5 > 4 > 3 > 2."""

    def test_ace_is_highest(self):
        assert Rank.ACE > Rank.SEVEN

    def test_seven_beats_king(self):
        assert Rank.SEVEN > Rank.KING

    def test_king_beats_jack(self):
        assert Rank.KING > Rank.JACK

    def test_jack_beats_queen(self):
        assert Rank.JACK > Rank.QUEEN

    def test_queen_beats_six(self):
        assert Rank.QUEEN > Rank.SIX

    def test_full_ordering(self):
        expected = [
            Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX,
            Rank.QUEEN, Rank.JACK, Rank.KING, Rank.SEVEN, Rank.ACE,
        ]
        assert sorted(Rank) == expected

    def test_ace_value_is_9(self):
        """Ace is rank index 9 (used for normalization /9.0)."""
        assert int(Rank.ACE) == 9


# ─── Dealing ────────────────────────────────────────────────────────────────

class TestDealing:
    """Verify deal() produces correct hands."""

    def test_deal_produces_4_hands(self):
        rng = np.random.default_rng(42)
        hands = deal(rng)
        assert len(hands) == 4

    def test_each_hand_has_10_cards(self):
        rng = np.random.default_rng(42)
        hands = deal(rng)
        for h in hands:
            assert len(h) == 10

    def test_no_card_overlap(self):
        rng = np.random.default_rng(42)
        hands = deal(rng)
        all_cards = [c for h in hands for c in h]
        assert len(set(all_cards)) == 40

    def test_all_cards_from_deck(self):
        rng = np.random.default_rng(42)
        hands = deal(rng)
        all_cards = set(c for h in hands for c in h)
        assert all_cards == set(build_deck())

    def test_deterministic_with_same_seed(self):
        hands1 = deal(np.random.default_rng(123))
        hands2 = deal(np.random.default_rng(123))
        assert hands1 == hands2

    def test_different_with_different_seed(self):
        hands1 = deal(np.random.default_rng(1))
        hands2 = deal(np.random.default_rng(2))
        # Extremely unlikely to be the same.
        assert hands1 != hands2

    def test_deal_is_shuffled(self):
        """Dealing should not produce sorted order."""
        rng = np.random.default_rng(42)
        hands = deal(rng)
        sorted_deck = build_deck()
        sorted_hands = [sorted_deck[i * 10 : (i + 1) * 10] for i in range(4)]
        assert hands != sorted_hands


# ─── Trump selection ────────────────────────────────────────────────────────

class TestTrumpSelection:
    """Verify choose_trump returns valid suits."""

    def test_returns_valid_suit(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            trump = choose_trump(rng)
            assert trump in Suit

    def test_deterministic_with_same_seed(self):
        t1 = choose_trump(np.random.default_rng(42))
        t2 = choose_trump(np.random.default_rng(42))
        assert t1 == t2


# ─── Card repr ──────────────────────────────────────────────────────────────

class TestCardRepr:
    """Verify human-readable card representation."""

    def test_ace_of_hearts(self):
        assert repr(Card(Suit.HEARTS, Rank.ACE)) == "A♥"

    def test_seven_of_spades(self):
        assert repr(Card(Suit.SPADES, Rank.SEVEN)) == "7♠"

    def test_queen_of_diamonds(self):
        assert repr(Card(Suit.DIAMONDS, Rank.QUEEN)) == "Q♦"

    def test_two_of_clubs(self):
        assert repr(Card(Suit.CLUBS, Rank.TWO)) == "2♣"
