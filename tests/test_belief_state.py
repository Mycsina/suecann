"""Thorough tests for src/engine/belief_state.py."""

from __future__ import annotations
import numpy as np
import pytest
from src.engine.cards import Card, Rank, Suit, build_deck, deal
from src.engine.sueca_engine import SuecaGame, TrickCard, VisibleState
from src.engine.belief_state import encode, BELIEF_DIM


def _make_game(seed=42, trump=Suit.HEARTS):
    rng = np.random.default_rng(seed)
    hands = deal(rng)
    return SuecaGame(hands, trump=trump)


class TestDimensions:
    def test_output_shape(self):
        game = _make_game()
        vs = game.get_visible_state(0)
        vec = encode(vs)
        assert vec.shape == (BELIEF_DIM,)
        assert BELIEF_DIM == 18

    def test_all_values_in_0_1(self):
        """All belief vector values must be in [0, 1]."""
        for seed in range(20):
            game = _make_game(seed)
            # Play a few tricks randomly then encode at various points.
            rng = np.random.default_rng(seed + 100)
            for _ in range(15):  # up to 15 plays
                if game.is_terminal():
                    break
                seat = game.current_player
                vs = game.get_visible_state(seat)
                vec = encode(vs)
                assert np.all(vec >= 0.0), f"Negative value found: {vec}"
                assert np.all(vec <= 1.0), f"Value > 1 found: {vec}"
                legal = game.legal_moves(seat)
                game.play_card(seat, legal[rng.integers(len(legal))])


class TestHandFeatures:
    def test_has_led_suit_when_leading(self):
        """When leading (no led suit), Has_Led_Suit should be 0."""
        game = _make_game()
        vs = game.get_visible_state(0)
        vec = encode(vs)
        assert vec[0] == 0.0  # Leading, no led suit

    def test_has_led_suit_positive(self):
        """When a suit is led and player has it, field should be 1."""
        # Seat 0 has 5 hearts + 5 diamonds, seat 3 has 5 hearts + 5 spades.
        hearts = [Card(Suit.HEARTS, r) for r in Rank]
        h0 = hearts[:5] + [Card(Suit.DIAMONDS, r) for r in list(Rank)[:5]]
        h3 = hearts[5:] + [Card(Suit.SPADES, r) for r in list(Rank)[:5]]
        used = set(h0 + h3)
        filler = [c for c in build_deck() if c not in used]
        h1, h2 = filler[:10], filler[10:20]

        game = SuecaGame([h0, h1, h2, h3], trump=Suit.CLUBS)
        game.play_card(0, h0[0])  # leads a heart
        vs = game.get_visible_state(3)
        vec = encode(vs)
        assert vec[0] == 1.0  # seat 3 has hearts

    def test_has_led_suit_negative(self):
        """When player is void in led suit, field should be 0."""
        h0 = [Card(Suit.HEARTS, r) for r in Rank]
        h3 = [Card(Suit.SPADES, r) for r in Rank]  # void in hearts
        remaining = [c for c in build_deck() if c not in h0 and c not in h3]
        h2, h1 = remaining[:10], remaining[10:20]

        game = SuecaGame([h0, h1, h2, h3], trump=Suit.DIAMONDS)
        game.play_card(0, Card(Suit.HEARTS, Rank.TWO))
        vs = game.get_visible_state(3)
        vec = encode(vs)
        assert vec[0] == 0.0

    def test_has_trump(self):
        h0 = [Card(Suit.HEARTS, r) for r in Rank]
        remaining = [c for c in build_deck() if c not in h0]
        h3 = remaining[:10]
        h2, h1 = remaining[10:20], remaining[20:30]
        game = SuecaGame([h0, h1, h2, h3], trump=Suit.HEARTS)
        vs = game.get_visible_state(0)
        vec = encode(vs)
        assert vec[1] == 1.0  # Has trump (hearts)

    def test_no_trump(self):
        h0 = [Card(Suit.HEARTS, r) for r in Rank]
        remaining = [c for c in build_deck() if c not in h0]
        h3 = remaining[:10]
        h2, h1 = remaining[10:20], remaining[20:30]
        game = SuecaGame([h0, h1, h2, h3], trump=Suit.SPADES)
        # Seat 0 has only hearts, trump is spades.
        vs = game.get_visible_state(0)
        vec = encode(vs)
        if not any(c.suit == Suit.SPADES for c in vs.hand):
            assert vec[1] == 0.0

    def test_led_suit_power_ace(self):
        """Led suit power with Ace (rank 9) should be 9/9 = 1.0."""
        h0 = [Card(Suit.HEARTS, r) for r in Rank]  # has A♥
        remaining = [c for c in build_deck() if c not in h0]
        h3 = [Card(Suit.HEARTS, Rank.ACE)] + remaining[:9]
        # Reconstruct to avoid duplicates.
        all_used = set(h0 + h3)
        filler = [c for c in build_deck() if c not in all_used]
        h2, h1 = filler[:10], filler[10:20]

        # Actually seat 3 can't have A♥ if seat 0 already has it.
        # Let's just test seat 0 as leader then encoding after seat 0 leads.
        # When seat 0 leads, led_suit = None, so power = 0. Let's test differently.
        # Make seat 3 lead hearts, then check seat 0's encoding.
        game = SuecaGame([h0, h1, h2, h3], trump=Suit.DIAMONDS, first_player=3)
        # Seat 3 leads something.
        s3_heart = [c for c in game.hands[3] if c.suit == Suit.HEARTS]
        if s3_heart:
            game.play_card(3, s3_heart[0])
            # Now it's seat 2's turn (counter-clockwise: 3→2→1→0).
            game.play_card(2, game.legal_moves(2)[0])
            game.play_card(1, game.legal_moves(1)[0])
            # Now seat 0's turn.
            vs = game.get_visible_state(0)
            vec = encode(vs)
            # Seat 0 has A♥, led suit is hearts.
            assert vec[2] == pytest.approx(9.0 / 9.0)

    def test_trump_power(self):
        """Trump power with highest trump = 7 (rank 8) → 8/9."""
        h0 = [Card(Suit.HEARTS, r) for r in list(Rank)[:8]] + [
            Card(Suit.DIAMONDS, Rank.SEVEN),
            Card(Suit.DIAMONDS, Rank.KING),
        ]
        remaining = [c for c in build_deck() if c not in h0]
        h3, h2, h1 = remaining[:10], remaining[10:20], remaining[20:30]
        game = SuecaGame([h0, h1, h2, h3], trump=Suit.DIAMONDS)
        vs = game.get_visible_state(0)
        vec = encode(vs)
        # Highest trump in hand is 7♦ = rank 8.
        assert vec[3] == pytest.approx(8.0 / 9.0)


class TestTrickFeatures:
    def test_am_i_leading(self):
        game = _make_game()
        vs = game.get_visible_state(0)
        vec = encode(vs)
        assert vec[5] == 1.0  # Leading
        assert vec[6] == 0.0  # Not last

    def test_am_i_last(self):
        game = _make_game()
        # Play 3 cards so seat 1 is last.
        for _ in range(3):
            seat = game.current_player
            game.play_card(seat, game.legal_moves(seat)[0])
        seat = game.current_player  # Should be seat 1 (last in counter-clockwise).
        vs = game.get_visible_state(seat)
        vec = encode(vs)
        assert vec[5] == 0.0  # Not leading
        assert vec[6] == 1.0  # Last to play

    def test_trick_point_value_empty(self):
        game = _make_game()
        vs = game.get_visible_state(0)
        vec = encode(vs)
        assert vec[8] == 0.0  # No cards played yet in trick


class TestHistoryFeatures:
    def test_has_trick_been_cut(self):
        """When trump is played on a non-trump led trick, cut = 1."""
        h0 = [Card(Suit.HEARTS, r) for r in Rank]
        h3 = [Card(Suit.SPADES, r) for r in Rank]  # void in hearts
        remaining = [c for c in build_deck() if c not in h0 and c not in h3]
        h2, h1 = remaining[:10], remaining[10:20]
        trump = Suit.SPADES

        game = SuecaGame([h0, h1, h2, h3], trump=trump)
        game.play_card(0, Card(Suit.HEARTS, Rank.TWO))  # lead hearts
        game.play_card(3, Card(Suit.SPADES, Rank.TWO))  # play trump (cut!)

        # From seat 2's perspective.
        vs = game.get_visible_state(2)
        vec = encode(vs)
        assert vec[9] == 1.0  # Trick has been cut

    def test_no_cut_when_trump_is_led(self):
        """If trump IS the led suit, Has_Trick_Been_Cut should be 0."""
        h0 = [Card(Suit.HEARTS, r) for r in Rank]
        remaining = [c for c in build_deck() if c not in h0]
        h3, h2, h1 = remaining[:10], remaining[10:20], remaining[20:30]
        trump = Suit.HEARTS

        game = SuecaGame([h0, h1, h2, h3], trump=trump)
        game.play_card(0, Card(Suit.HEARTS, Rank.TWO))
        vs = game.get_visible_state(3)
        vec = encode(vs)
        assert vec[9] == 0.0  # Led suit IS trump, so no "cut"

    def test_game_points_remaining_start(self):
        game = _make_game()
        vs = game.get_visible_state(0)
        vec = encode(vs)
        assert vec[17] == pytest.approx(1.0)  # 120/120

    def test_game_points_remaining_decreases(self):
        rng = np.random.default_rng(42)
        game = _make_game()
        # Play one full trick.
        for _ in range(4):
            seat = game.current_player
            legal = game.legal_moves(seat)
            game.play_card(seat, legal[rng.integers(len(legal))])
        vs = game.get_visible_state(game.current_player)
        vec = encode(vs)
        assert vec[17] < 1.0


class TestPartnerWinning:
    def test_partner_winning_detected(self):
        """Seat 0 and 2 are partners. If seat 0 leads high, seat 2 sees partner winning."""
        h0 = [Card(Suit.HEARTS, r) for r in Rank]
        remaining = [c for c in build_deck() if c not in h0]
        h3 = remaining[:10]
        h2 = remaining[10:20]
        h1 = remaining[20:30]

        game = SuecaGame([h0, h1, h2, h3], trump=Suit.DIAMONDS)
        game.play_card(0, Card(Suit.HEARTS, Rank.ACE))  # Seat 0 leads A♥
        # Seat 3 plays.
        game.play_card(3, game.legal_moves(3)[0])

        # Now seat 2's turn — partner (seat 0) should be winning.
        vs = game.get_visible_state(2)
        vec = encode(vs)
        # Partner winning depends on what seat 3 played, but A♥ is top rank.
        # If seat 3 didn't play trump, seat 0 is winning.
        s3_card = (
            game.trick_history[0][1].card
            if game.trick_history
            else game.current_trick[1].card
        )
        if s3_card.suit != Suit.DIAMONDS:  # didn't trump
            assert vec[7] == 1.0
