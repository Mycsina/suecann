"""Belief state encoder — translates visible game state into an 18-dim vector.

All values are normalized to [0, 1] for direct consumption by the WANN.
See implementation_plan.md for the full field specification.
"""

from __future__ import annotations

import numpy as np

from src.engine.cards import (
    Card,
    Rank,
    Suit,
    card_points,
    POINT_VALUES,
    TOTAL_GAME_POINTS,
)
from src.engine.sueca_engine import VisibleState, TrickCard


# Number of belief state features.
BELIEF_DIM = 18

# Maximum rank index, used for normalizing rank power.
_MAX_RANK = float(Rank.ACE)  # 9.0


def _max_rank_in_suit(hand: list[Card], suit: Suit) -> int:
    """Return the highest rank index of cards in the given suit, or -1 if void."""
    ranks = [c.rank for c in hand if c.suit == suit]
    return max(ranks) if ranks else -1


def _has_suit(hand: list[Card], suit: Suit) -> bool:
    """Check if hand contains any card of the given suit."""
    return any(c.suit == suit for c in hand)


def _sum_points(cards: list[Card]) -> int:
    """Sum the point values of a list of cards."""
    return sum(card_points(c) for c in cards)


def _cards_played_so_far(state: VisibleState) -> list[Card]:
    """Return all cards played in previous tricks."""
    played: list[Card] = []
    for trick in state.trick_history:
        for tc in trick:
            played.append(tc.card)
    return played


def _total_points_remaining(state: VisibleState) -> int:
    """Return total point value of cards not yet played (including current trick)."""
    played = _cards_played_so_far(state)
    played_points = _sum_points(played)
    # Cards in current trick are also "consumed" but haven't been archived yet.
    current_points = sum(card_points(tc.card) for tc in state.current_trick)
    return TOTAL_GAME_POINTS - played_points - current_points


def _trick_winner_seat(trick: list[TrickCard], trump: Suit) -> int | None:
    """Return the seat currently winning the (possibly incomplete) trick."""
    if not trick:
        return None

    led_suit = trick[0].card.suit
    best_seat = trick[0].seat
    best_card = trick[0].card

    for tc in trick[1:]:
        c = tc.card
        best_is_trump = best_card.suit == trump
        c_is_trump = c.suit == trump

        if c_is_trump and not best_is_trump:
            best_seat, best_card = tc.seat, c
        elif c_is_trump and best_is_trump and c.rank > best_card.rank:
            best_seat, best_card = tc.seat, c
        elif not c_is_trump and not best_is_trump:
            if c.suit == led_suit and best_card.suit == led_suit:
                if c.rank > best_card.rank:
                    best_seat, best_card = tc.seat, c
            elif c.suit == led_suit:
                best_seat, best_card = tc.seat, c

    return best_seat


def _is_partner(seat_a: int, seat_b: int) -> bool:
    """Check if two seats are on the same team (0&2 or 1&3)."""
    return (seat_a % 2) == (seat_b % 2)


def _any_card_played(state: VisibleState, suit: Suit, rank: Rank) -> bool:
    """Check if a specific card has been played in a previous trick."""
    for trick in state.trick_history:
        for tc in trick:
            if tc.card.suit == suit and tc.card.rank == rank:
                return True
    return False


def _player_position_in_trick(state: VisibleState) -> int:
    """Return 0-based position of the player in the current trick (0=leading, 3=last)."""
    return len(state.current_trick)


def encode(state: VisibleState) -> np.ndarray:
    """Encode a visible game state into an 18-dimensional belief vector.

    Returns:
        np.ndarray of shape (18,) with all values in [0, 1].
    """
    vec = np.zeros(BELIEF_DIM, dtype=np.float64)
    hand = state.hand
    trump = state.trump
    led_suit = state.led_suit
    position = _player_position_in_trick(state)

    # --- Hand features (5) ---

    # [0] Has_Led_Suit: 1 if hand contains cards of led suit.
    #     0 if leading (led_suit is None).
    if led_suit is not None:
        vec[0] = 1.0 if _has_suit(hand, led_suit) else 0.0
    else:
        vec[0] = 0.0

    # [1] Has_Trump: 1 if hand contains trump cards.
    vec[1] = 1.0 if _has_suit(hand, trump) else 0.0

    # [2] Led_Suit_Power: max rank in led suit / 9.0. 0 if void or leading.
    if led_suit is not None:
        max_r = _max_rank_in_suit(hand, led_suit)
        vec[2] = max_r / _MAX_RANK if max_r >= 0 else 0.0
    else:
        vec[2] = 0.0

    # [3] Trump_Power: max rank in trump / 9.0. 0 if void.
    max_trump = _max_rank_in_suit(hand, trump)
    vec[3] = max_trump / _MAX_RANK if max_trump >= 0 else 0.0

    # [4] Hand_Point_Density: sum of points in hand / remaining game points.
    remaining = _total_points_remaining(state)
    hand_points = _sum_points(hand)
    if remaining > 0:
        vec[4] = hand_points / remaining
    else:
        vec[4] = 0.0

    # --- Trick features (4) ---

    # [5] Am_I_Leading: 1 if agent is first to play.
    vec[5] = 1.0 if position == 0 else 0.0

    # [6] Am_I_Last_To_Play: 1 if agent is 4th to play.
    vec[6] = 1.0 if position == 3 else 0.0

    # [7] Is_Partner_Winning: 1 if partner has the highest card so far.
    winner_seat = _trick_winner_seat(state.current_trick, trump)
    if winner_seat is not None:
        vec[7] = 1.0 if _is_partner(state.seat, winner_seat) else 0.0
    else:
        vec[7] = 0.0

    # [8] Trick_Point_Value: sum of points in current trick / 44.
    trick_pts = sum(card_points(tc.card) for tc in state.current_trick)
    vec[8] = trick_pts / 44.0

    # --- History features (9) ---

    # [9] Has_Trick_Been_Cut: 1 if trump played in current trick (when led != trump).
    if led_suit is not None and led_suit != trump:
        vec[9] = (
            1.0 if any(tc.card.suit == trump for tc in state.current_trick) else 0.0
        )
    else:
        vec[9] = 0.0

    # [10] Partner_Void_Led: 1 if partner void in led suit.
    partner_seat = (state.seat + 2) % 4
    if led_suit is not None:
        vec[10] = 1.0 if led_suit in state.voids.get(partner_seat, set()) else 0.0
    else:
        vec[10] = 0.0

    # [11] Partner_Void_Trump: 1 if partner void in trump.
    vec[11] = 1.0 if trump in state.voids.get(partner_seat, set()) else 0.0

    # [12] Any_Opp_Void_Led: 1 if either opponent void in led suit.
    opp_seats = [(state.seat + 1) % 4, (state.seat + 3) % 4]
    if led_suit is not None:
        vec[12] = (
            1.0
            if any(led_suit in state.voids.get(s, set()) for s in opp_seats)
            else 0.0
        )
    else:
        vec[12] = 0.0

    # [13] Any_Opp_Void_Trump: 1 if either opponent void in trump.
    vec[13] = 1.0 if any(trump in state.voids.get(s, set()) for s in opp_seats) else 0.0

    # [14] Led_Suit_Ace_Played: 1 if Ace of led suit played previously.
    if led_suit is not None:
        vec[14] = 1.0 if _any_card_played(state, led_suit, Rank.ACE) else 0.0
    else:
        vec[14] = 0.0

    # [15] Led_Suit_7_Played: 1 if 7 (manilha) of led suit played previously.
    if led_suit is not None:
        vec[15] = 1.0 if _any_card_played(state, led_suit, Rank.SEVEN) else 0.0
    else:
        vec[15] = 0.0

    # [16] Trump_Ace_Played: 1 if trump Ace played previously.
    vec[16] = 1.0 if _any_card_played(state, trump, Rank.ACE) else 0.0

    # [17] Game_Pts_Remaining: total unplayed points / 120.
    vec[17] = remaining / TOTAL_GAME_POINTS

    return vec
