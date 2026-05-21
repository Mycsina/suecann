"""Heuristic Sueca bot — rule-based player for curriculum training.

Strategy:
  - Leading: play highest card of longest non-trump suit (or Ace/7), or
    lead trump if holding many trumps.
  - Following (can follow suit): play lowest card that beats current winner,
    otherwise dump lowest.
  - Void (can't follow): cut with lowest trump if partner isn't winning,
    otherwise dump lowest off-suit card.
"""

from __future__ import annotations

import numpy as np

from src.engine.cards import Card, Rank, Suit, card_points
from src.engine.sueca_engine import SuecaGame, TrickCard


def _is_partner(seat_a: int, seat_b: int) -> bool:
    return (seat_a % 2) == (seat_b % 2)


def _trick_winner_seat(
    trick: list[TrickCard], trump: Suit
) -> int:
    """Return seat currently winning the incomplete trick."""
    if not trick:
        return -1
    led_suit = trick[0].card.suit
    best = trick[0]
    for tc in trick[1:]:
        c = tc.card
        b = best.card
        c_trump = c.suit == trump
        b_trump = b.suit == trump
        if c_trump and not b_trump:
            best = tc
        elif c_trump and b_trump and c.rank > b.rank:
            best = tc
        elif not c_trump and not b_trump:
            if c.suit == led_suit and b.suit == led_suit:
                if c.rank > b.rank:
                    best = tc
            elif c.suit == led_suit:
                best = tc
    return best.seat


class HeuristicBot:
    """Rule-based Sueca player."""

    def select_card(
        self, game: SuecaGame, seat: int, rng: np.random.Generator
    ) -> Card:
        legal = game.legal_moves(seat)
        trump = game.trump
        trick = game.current_trick

        if not trick:
            return self._lead(legal, trump, rng)

        led_suit = trick[0].card.suit
        has_led = any(c.suit == led_suit for c in legal)

        if has_led:
            return self._follow(legal, trick, trump, led_suit, seat, rng)
        else:
            return self._void(legal, trick, trump, led_suit, seat, rng)

    def reset(self) -> None:
        pass

    def _lead(
        self, legal: list[Card], trump: Suit, rng: np.random.Generator
    ) -> Card:
        """Lead with strongest non-trump card, preferring longest suit."""
        # Count cards per suit.
        non_trump = [c for c in legal if c.suit != trump]
        trump_cards = [c for c in legal if c.suit == trump]

        if non_trump:
            # Prefer longest suit, then highest rank.
            from collections import Counter
            suit_counts = Counter(c.suit for c in non_trump)
            best_suit = max(suit_counts, key=lambda s: (suit_counts[s], s))
            suited = [c for c in non_trump if c.suit == best_suit]
            suited.sort(key=lambda c: (c.rank, card_points(c)), reverse=True)
            return suited[0]

        # Only trump — lead lowest trump.
        trump_cards.sort(key=lambda c: c.rank)
        return trump_cards[0]

    def _follow(
        self,
        legal: list[Card],
        trick: list[TrickCard],
        trump: Suit,
        led_suit: Suit,
        seat: int,
        rng: np.random.Generator,
    ) -> Card:
        """Follow suit: try to win cheaply, otherwise dump."""
        suited = [c for c in legal if c.suit == led_suit]

        # Try to beat current winner with the cheapest card.
        winner_seat = _trick_winner_seat(trick, trump)
        winner_card = None
        for tc in trick:
            if tc.seat == winner_seat:
                winner_card = tc.card
                break

        if winner_card is not None:
            # If partner is winning and we can't add points, dump.
            if _is_partner(seat, winner_seat):
                # Play lowest card (save strength).
                suited.sort(key=lambda c: (c.rank, -card_points(c)))
                return suited[0]

            # Try to beat winner.
            beating = [
                c for c in suited
                if winner_card.suit == trump and c.suit == trump and c.rank > winner_card.rank
                or winner_card.suit != trump and c.rank > winner_card.rank
                or (winner_card.suit != trump and winner_card.suit != led_suit and c.suit == led_suit)
            ]

            if winner_card.suit == trump:
                # Can only beat with higher trump.
                beating = [
                    c for c in suited
                    if c.suit == trump and c.rank > winner_card.rank
                ]
            elif winner_card.suit == led_suit:
                beating = [
                    c for c in suited
                    if c.rank > winner_card.rank
                ]
            else:
                # Winner is off-suit non-trump — any led suit card beats it.
                beating = list(suited)

            if beating:
                # Play cheapest beating card.
                beating.sort(key=lambda c: (c.rank, -card_points(c)))
                return beating[0]

        # Can't beat winner — dump lowest.
        suited.sort(key=lambda c: (c.rank, -card_points(c)))
        return suited[0]

    def _void(
        self,
        legal: list[Card],
        trick: list[TrickCard],
        trump: Suit,
        led_suit: Suit,
        seat: int,
        rng: np.random.Generator,
    ) -> Card:
        """Void in led suit: cut with trump if needed, otherwise dump."""
        winner_seat = _trick_winner_seat(trick, trump)
        trump_cards = [c for c in legal if c.suit == trump]
        non_trump = [c for c in legal if c.suit != trump]

        # If partner is winning, dump.
        if _is_partner(seat, winner_seat):
            # Dump lowest off-suit card.
            if non_trump:
                non_trump.sort(key=lambda c: (c.rank, card_points(c)))
                return non_trump[0]
            # Only trump — play lowest.
            trump_cards.sort(key=lambda c: c.rank)
            return trump_cards[0]

        # Opponent is winning — cut if possible.
        if trump_cards:
            # Cut with lowest trump that beats current winner, or lowest trump.
            winner_card = None
            for tc in trick:
                if tc.seat == winner_seat:
                    winner_card = tc.card
                    break

            if winner_card is not None and winner_card.suit == trump:
                # Need higher trump to over-cut.
                over_trumps = [c for c in trump_cards if c.rank > winner_card.rank]
                if over_trumps:
                    over_trumps.sort(key=lambda c: c.rank)
                    return over_trumps[0]
                # Can't over-cut — dump.
                if non_trump:
                    non_trump.sort(key=lambda c: (c.rank, card_points(c)))
                    return non_trump[0]
            else:
                # Any trump wins.
                trump_cards.sort(key=lambda c: c.rank)
                return trump_cards[0]

        # No trump — dump lowest.
        if non_trump:
            non_trump.sort(key=lambda c: (c.rank, card_points(c)))
            return non_trump[0]

        # Fallback.
        legal_sorted = sorted(legal, key=lambda c: (c.rank, card_points(c)))
        return legal_sorted[0]
