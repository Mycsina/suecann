"""Thorough tests for src/engine/sueca_engine.py."""

from __future__ import annotations
import numpy as np
import pytest
from src.engine.cards import Card, Rank, Suit, build_deck, card_points, deal
from src.engine.sueca_engine import SuecaGame, TrickCard, _next_seat_after


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_hands_from_deck():
    """Deterministic deal for testing."""
    return deal(np.random.default_rng(42))


def _play_full_game_random(seed: int = 42) -> SuecaGame:
    """Play a full game with random (but legal) moves."""
    rng = np.random.default_rng(seed)
    hands = deal(rng)
    game = SuecaGame(hands, trump=Suit.HEARTS)
    while not game.is_terminal():
        seat = game.current_player
        legal = game.legal_moves(seat)
        card = legal[rng.integers(len(legal))]
        game.play_card(seat, card)
    return game


# ─── Turn order ─────────────────────────────────────────────────────────────


class TestTurnOrder:
    def test_counter_clockwise(self):
        assert _next_seat_after(0) == 3
        assert _next_seat_after(3) == 2
        assert _next_seat_after(2) == 1
        assert _next_seat_after(1) == 0

    def test_full_cycle(self):
        seat = 0
        visited = [seat]
        for _ in range(3):
            seat = _next_seat_after(seat)
            visited.append(seat)
        assert visited == [0, 3, 2, 1]


# ─── Initialization ────────────────────────────────────────────────────────


class TestInit:
    def test_reject_wrong_hand_count(self):
        with pytest.raises(ValueError):
            SuecaGame([[], [], []], trump=Suit.HEARTS)

    def test_reject_wrong_hand_size(self):
        deck = build_deck()
        hands = [deck[:9], deck[9:19], deck[19:29], deck[29:39]]
        with pytest.raises(ValueError):
            SuecaGame(hands, trump=Suit.HEARTS)

    def test_initial_state(self):
        hands = _make_hands_from_deck()
        game = SuecaGame(hands, trump=Suit.HEARTS)
        assert not game.is_terminal()
        assert game.trick_number == 0
        assert game.team_scores == [0, 0]
        assert game.current_player == 0


# ─── Follow-suit enforcement ───────────────────────────────────────────────


class TestFollowSuit:
    def test_must_follow_led_suit(self):
        """If player has cards of the led suit, they must play one."""
        # Construct hands where seat 0 leads hearts, seat 3 has hearts + spades.
        h0 = [Card(Suit.HEARTS, r) for r in list(Rank)[:10]]  # all hearts
        h3 = [Card(Suit.SPADES, r) for r in list(Rank)[:5]] + [
            Card(Suit.HEARTS, r) for r in list(Rank)[:5]
        ]  # mixed
        # Dummy hands for seats 1, 2.
        remaining = [c for c in build_deck() if c not in h0 and c not in h3]
        h2 = remaining[:10]
        h1 = remaining[10:20]
        game = SuecaGame([h0, h1, h2, h3], trump=Suit.DIAMONDS)

        # Seat 0 leads A♥.
        game.play_card(0, Card(Suit.HEARTS, Rank.ACE))

        # Seat 3 has hearts so must follow suit. Playing spades should fail.
        spade_card = Card(Suit.SPADES, Rank.TWO)
        assert spade_card in game.hands[3]
        with pytest.raises(ValueError, match="not a legal play"):
            game.play_card(3, spade_card)

    def test_can_play_any_when_void(self):
        """If player is void in led suit, any card is legal."""
        h0 = [Card(Suit.HEARTS, r) for r in Rank]  # 10 hearts
        h3 = [Card(Suit.SPADES, r) for r in Rank]  # 10 spades, no hearts
        remaining = [c for c in build_deck() if c not in h0 and c not in h3]
        h2 = remaining[:10]
        h1 = remaining[10:20]
        game = SuecaGame([h0, h1, h2, h3], trump=Suit.DIAMONDS)

        game.play_card(0, Card(Suit.HEARTS, Rank.ACE))
        legal = game.legal_moves(3)
        # All 10 spades should be legal (entire hand).
        assert len(legal) == 10

    def test_leader_can_play_any(self):
        hands = _make_hands_from_deck()
        game = SuecaGame(hands, trump=Suit.HEARTS)
        legal = game.legal_moves(0)
        assert legal == list(game.hands[0])


# ─── Trick resolution ──────────────────────────────────────────────────────


class TestTrickResolution:
    def _setup_simple_trick(self, cards_by_seat, trump=Suit.DIAMONDS):
        """Create a game and play a single full trick."""
        # Build minimal hands where each seat has exactly the needed card + 9 fillers.
        deck = build_deck()
        used = set(cards_by_seat.values())
        filler = [c for c in deck if c not in used]

        hands = []
        fi = 0
        for seat in range(4):
            hand = [cards_by_seat[seat]]
            needed = 9
            while needed > 0:
                if filler[fi] not in used:
                    hand.append(filler[fi])
                    needed -= 1
                fi += 1
            hands.append(hand)

        game = SuecaGame(hands, trump=trump)
        order = [0, 3, 2, 1]
        for seat in order:
            game.play_card(seat, cards_by_seat[seat])
        return game

    def test_highest_of_led_suit_wins(self):
        cards = {
            0: Card(Suit.HEARTS, Rank.THREE),
            3: Card(Suit.HEARTS, Rank.ACE),
            2: Card(Suit.HEARTS, Rank.KING),
            1: Card(Suit.HEARTS, Rank.TWO),
        }
        game = self._setup_simple_trick(cards, trump=Suit.DIAMONDS)
        assert game.trick_history[0]  # trick was played
        # Seat 3 played A♥ (highest) → wins.
        assert game.current_leader == 3

    def test_trump_beats_non_trump(self):
        cards = {
            0: Card(Suit.HEARTS, Rank.ACE),  # Led suit, highest
            3: Card(Suit.DIAMONDS, Rank.TWO),  # Trump, lowest
            2: Card(Suit.HEARTS, Rank.SEVEN),
            1: Card(Suit.HEARTS, Rank.KING),
        }
        game = self._setup_simple_trick(cards, trump=Suit.DIAMONDS)
        # Seat 3 played trump (even lowest) beats all non-trump.
        assert game.current_leader == 3

    def test_higher_trump_beats_lower_trump(self):
        cards = {
            0: Card(Suit.HEARTS, Rank.ACE),
            3: Card(Suit.DIAMONDS, Rank.TWO),
            2: Card(Suit.DIAMONDS, Rank.SEVEN),  # Higher trump
            1: Card(Suit.HEARTS, Rank.KING),
        }
        game = self._setup_simple_trick(cards, trump=Suit.DIAMONDS)
        assert game.current_leader == 2

    def test_off_suit_non_trump_loses(self):
        cards = {
            0: Card(Suit.HEARTS, Rank.THREE),
            3: Card(Suit.SPADES, Rank.ACE),  # Off suit, even A loses
            2: Card(Suit.HEARTS, Rank.KING),
            1: Card(Suit.HEARTS, Rank.TWO),
        }
        game = self._setup_simple_trick(cards, trump=Suit.DIAMONDS)
        # K♥ (seat 2) is highest of led suit.
        assert game.current_leader == 2


# ─── Void tracking ─────────────────────────────────────────────────────────


class TestVoidTracking:
    def test_void_recorded(self):
        h0 = [Card(Suit.HEARTS, r) for r in Rank]
        h3 = [Card(Suit.SPADES, r) for r in Rank]  # void in hearts
        remaining = [c for c in build_deck() if c not in h0 and c not in h3]
        h2, h1 = remaining[:10], remaining[10:20]
        game = SuecaGame([h0, h1, h2, h3], trump=Suit.DIAMONDS)

        game.play_card(0, Card(Suit.HEARTS, Rank.TWO))  # leads hearts
        game.play_card(3, Card(Suit.SPADES, Rank.TWO))  # can't follow → void
        assert Suit.HEARTS in game.voids[3]

    def test_following_suit_no_void(self):
        hands = _make_hands_from_deck()
        game = SuecaGame(hands, trump=Suit.HEARTS)
        card = game.hands[0][0]
        game.play_card(0, card)
        # Check if seat 3 follows suit (if they can).
        legal = game.legal_moves(3)
        game.play_card(3, legal[0])
        if legal[0].suit == card.suit:
            assert card.suit not in game.voids[3]


# ─── Full game invariants ──────────────────────────────────────────────────


class TestFullGame:
    def test_total_points_is_120(self):
        """After a full game, total points across both teams must be 120."""
        for seed in range(20):
            game = _play_full_game_random(seed)
            result = game.get_result()
            assert sum(result.team_scores) == 120, f"Seed {seed}: {result.team_scores}"

    def test_10_tricks_played(self):
        game = _play_full_game_random()
        assert len(game.trick_history) == 10

    def test_all_hands_empty(self):
        game = _play_full_game_random()
        for hand in game.hands:
            assert len(hand) == 0

    def test_is_terminal(self):
        game = _play_full_game_random()
        assert game.is_terminal()

    def test_cannot_play_after_terminal(self):
        game = _play_full_game_random()
        with pytest.raises(RuntimeError, match="already finished"):
            game.play_card(0, Card(Suit.HEARTS, Rank.ACE))

    def test_get_result_before_terminal_fails(self):
        hands = _make_hands_from_deck()
        game = SuecaGame(hands, trump=Suit.HEARTS)
        with pytest.raises(RuntimeError, match="not finished"):
            game.get_result()


# ─── Game point tiers ──────────────────────────────────────────────────────


class TestGamePointTiers:
    @pytest.mark.parametrize(
        "t02,t13,exp",
        [
            (120, 0, (4, 0)),  # Bandeira
            (0, 120, (0, 4)),
            (100, 20, (2, 0)),  # 91+ = 2 game pts
            (91, 29, (2, 0)),
            (70, 50, (1, 0)),  # 61-90 = 1 game pt
            (61, 59, (1, 0)),
            (60, 60, (0, 0)),  # Draw
            (50, 70, (0, 1)),
        ],
    )
    def test_game_point_tiers(self, t02, t13, exp):
        from src.engine.sueca_engine import GameResult

        assert GameResult.compute_game_points(t02, t13) == exp


# ─── Visible state ──────────────────────────────────────────────────────────


class TestVisibleState:
    def test_no_opponent_hands_exposed(self):
        hands = _make_hands_from_deck()
        game = SuecaGame(hands, trump=Suit.HEARTS)
        vs = game.get_visible_state(0)
        # Visible state should only contain seat 0's hand.
        assert vs.hand == list(game.hands[0])
        # No attribute gives access to other hands.
        assert not hasattr(vs, "all_hands")

    def test_led_suit_is_none_when_leading(self):
        hands = _make_hands_from_deck()
        game = SuecaGame(hands, trump=Suit.HEARTS)
        vs = game.get_visible_state(0)
        assert vs.led_suit is None

    def test_led_suit_set_after_lead(self):
        hands = _make_hands_from_deck()
        game = SuecaGame(hands, trump=Suit.HEARTS)
        card = game.hands[0][0]
        game.play_card(0, card)
        vs = game.get_visible_state(3)
        assert vs.led_suit == card.suit

    def test_trick_number_increments(self):
        game = _play_full_game_random()
        assert game.trick_number == 10
