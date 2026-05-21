"""Oracle intent resolver — maps 5 abstract intents to legal card selections.

Intent IDs:
  0. DUCK_OR_DUMP  — Lowest legal card (fallback, never illegal)
  1. TAKE_CHEAPLY  — Min card that beats current winner
  2. FORCE_HIGH    — Highest power card
  3. FEED_PARTNER  — Highest point-value card
  4. CUT_LOW       — Lowest trump

Illegal intent → fallback to DUCK_OR_DUMP + Oracle Tax penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from src.engine.cards import Card, Rank, card_points
from src.engine.sueca_engine import SuecaGame, TrickCard


class Intent(IntEnum):
    DUCK_OR_DUMP = 0
    TAKE_CHEAPLY = 1
    FORCE_HIGH = 2
    FEED_PARTNER = 3
    CUT_LOW = 4


INTENT_COUNT = 5


@dataclass
class IntentResult:
    """Result of resolving an intent to a card."""

    card: Card
    intent_used: int  # the actual intent that was executed
    was_illegal: bool  # True if the original intent was illegal


def _card_strength(card: Card, trump_suit: int, led_suit: int | None) -> float:
    """Compute a heuristic strength score for a card in a trick context.

    Trump cards get a large bonus. Non-trump cards of the led suit
    get their rank value. Off-suit non-trump cards get -1 (can't win).
    """
    if card.suit == trump_suit:
        return 100.0 + float(card.rank)
    if led_suit is not None and card.suit == led_suit:
        return float(card.rank)
    return -1.0


def _would_beat(
    card: Card,
    current_trick: list[TrickCard],
    trump: int,
    led_suit: int,
) -> bool:
    """Check whether `card` would win the trick if played now (as last player)."""
    if not current_trick:
        return True

    best_so_far = current_trick[0].card
    for tc in current_trick[1:]:
        c = tc.card
        c_is_trump = c.suit == trump
        best_is_trump = best_so_far.suit == trump

        if c_is_trump and not best_is_trump:
            best_so_far = c
        elif c_is_trump and best_is_trump and c.rank > best_so_far.rank:
            best_so_far = c
        elif not c_is_trump and not best_is_trump:
            if c.suit == led_suit and best_so_far.suit == led_suit:
                if c.rank > best_so_far.rank:
                    best_so_far = c
            elif c.suit == led_suit:
                best_so_far = c

    # Check if challenger beats best_so_far.
    c_is_trump = card.suit == trump
    best_is_trump = best_so_far.suit == trump

    if c_is_trump and not best_is_trump:
        return True
    if not c_is_trump and best_is_trump:
        return False
    if c_is_trump and best_is_trump:
        return card.rank > best_so_far.rank

    # Neither is trump.
    if card.suit == led_suit and best_so_far.suit == led_suit:
        return card.rank > best_so_far.rank
    if card.suit == led_suit:
        return True
    return False


def resolve_intent(
    intent: int,
    game: SuecaGame,
    seat: int,
    rng: np.random.Generator,
) -> IntentResult:
    """Resolve an abstract intent to a concrete card.

    Args:
        intent: One of Intent values (0-4) or the argmax of WANN output.
        game: Current game state.
        seat: The seat making the decision.
        rng: Random generator for tie-breaking.

    Returns:
        IntentResult with the chosen card and legality flag.
    """
    legal = game.legal_moves(seat)
    trump = game.trump
    current_trick = game.current_trick

    led_suit: int | None = current_trick[0].card.suit if current_trick else None

    # --- Intent 0: DUCK_OR_DUMP (always legal, always the fallback) ---
    def _duck_or_dump() -> Card:
        # Lowest card by rank (weakest first).
        legal_sorted = sorted(legal, key=lambda c: (c.rank, c.suit))
        return legal_sorted[0]

    # --- Intent 1: TAKE_CHEAPLY ---
    def _take_cheaply() -> Card | None:
        """Return cheapest card that beats current winner, or None."""
        if not current_trick:
            # Leading — can't "take" since there's nothing to beat.
            return None
        takers = [c for c in legal if _would_beat(c, current_trick, trump, led_suit)]
        if not takers:
            return None
        # Cheapest = lowest rank among those that would win.
        takers.sort(key=lambda c: (c.rank, c.suit))
        return takers[0]

    # --- Intent 2: FORCE_HIGH ---
    def _force_high() -> Card:
        """Highest power card among legal moves."""
        legal_sorted = sorted(
            legal,
            key=lambda c: _card_strength(c, trump, led_suit),
            reverse=True,
        )
        return legal_sorted[0]

    # --- Intent 3: FEED_PARTNER ---
    def _feed_partner() -> Card:
        """Highest point-value card among legal moves."""
        legal_sorted = sorted(
            legal,
            key=lambda c: (card_points(c), c.rank),
            reverse=True,
        )
        return legal_sorted[0]

    # --- Intent 4: CUT_LOW ---
    def _cut_low() -> Card | None:
        """Lowest trump card. Illegal when holding cards of led suit."""
        if led_suit is None:
            # Can't cut when leading.
            return None
        trump_cards = [c for c in legal if c.suit == trump]
        if not trump_cards:
            return None
        trump_cards.sort(key=lambda c: c.rank)
        return trump_cards[0]

    # Check legality and resolve.
    was_illegal = False
    card: Card | None = None

    if intent == Intent.DUCK_OR_DUMP:
        card = _duck_or_dump()
    elif intent == Intent.TAKE_CHEAPLY:
        card = _take_cheaply()
        if card is None:
            was_illegal = True
            card = _duck_or_dump()
    elif intent == Intent.FORCE_HIGH:
        card = _force_high()
    elif intent == Intent.FEED_PARTNER:
        card = _feed_partner()
    elif intent == Intent.CUT_LOW:
        # CUT_LOW is illegal when player holds cards of led suit (which is not trump)
        # because they must follow suit, OR when they have no trump card to cut with.
        if led_suit is not None:
            if led_suit == trump:
                # If led suit is trump, playing trump is following suit. Lowest trump is legal.
                card = _cut_low()
                if card is None:
                    was_illegal = True
                    card = _duck_or_dump()
            else:
                has_led_suit = any(c.suit == led_suit for c in legal)
                if has_led_suit:
                    # Must follow suit — can't cut.
                    was_illegal = True
                    card = _duck_or_dump()
                else:
                    card = _cut_low()
                    if card is None:
                        was_illegal = True
                        card = _duck_or_dump()
        else:
            # Leading — can't cut.
            was_illegal = True
            card = _duck_or_dump()
    else:
        # Unknown intent, fallback.
        was_illegal = True
        card = _duck_or_dump()

    return IntentResult(card=card, intent_used=intent, was_illegal=was_illegal)


def select_card_from_outputs(
    outputs: np.ndarray,
    game: SuecaGame,
    seat: int,
    rng: np.random.Generator,
) -> tuple[Card, bool]:
    """Given WANN outputs (5 floats), pick the intent with highest activation.

    Returns:
        (card, was_illegal) — was_illegal is True if the top intent was illegal
        and a fallback was used.
    """
    assert len(outputs) == INTENT_COUNT
    max_val = np.max(outputs)
    best_intents = np.where(outputs == max_val)[0]
    intent = int(rng.choice(best_intents))
    result = resolve_intent(intent, game, seat, rng)
    return result.card, result.was_illegal
