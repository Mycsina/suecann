"""Sueca game engine — 4-player trick-taking state machine.

Manages a single deal: 4 hands of 10 cards, 10 tricks, follow-suit rules,
trick resolution, score accumulation, and void tracking.

Teams: seats 0 & 2 vs seats 1 & 3.
Turn order: counter-clockwise (seat order 0 → 3 → 2 → 1 → 0 …).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from src.engine.cards import Card, Rank, Suit, card_points


class TrickCard(NamedTuple):
    """A card played in a trick, tagged with who played it."""

    seat: int
    card: Card


@dataclass
class VisibleState:
    """Information visible to a specific player at decision time."""

    seat: int
    hand: list[Card]
    trump: Suit
    led_suit: Suit | None  # None when this player is leading
    current_trick: list[TrickCard]
    trick_history: list[list[TrickCard]]
    team_scores: tuple[int, int]  # (team_02, team_13)
    voids: dict[int, set[Suit]]  # seat → set of suits they've shown void in
    trick_number: int  # 0-indexed, which trick we're on


@dataclass
class GameResult:
    """Final result of a completed game."""

    team_scores: tuple[int, int]  # (team_02, team_13)
    game_points: tuple[int, int]  # game-point tier per team

    @staticmethod
    def compute_game_points(team_02: int, team_13: int) -> tuple[int, int]:
        """Convert trick-point totals to game points (1/2/4 system)."""
        def _tier(pts: int) -> int:
            if pts == 120:
                return 4
            if pts >= 91:
                return 2
            if pts >= 61:
                return 1
            return 0

        return (_tier(team_02), _tier(team_13))


# Counter-clockwise turn order from seat 0.
_TURN_ORDER = [0, 3, 2, 1]


def _next_seat_after(seat: int) -> int:
    """Return the next seat in counter-clockwise order."""
    idx = _TURN_ORDER.index(seat)
    return _TURN_ORDER[(idx + 1) % 4]


class SuecaGame:
    """State machine for a single Sueca deal."""

    def __init__(
        self,
        hands: list[list[Card]],
        trump: Suit,
        first_player: int = 0,
    ) -> None:
        """Initialize a new game.

        Args:
            hands: 4 lists of 10 cards each.
            trump: The trump suit for this deal.
            first_player: Seat that leads the first trick (default 0).
        """
        if len(hands) != 4 or any(len(h) != 10 for h in hands):
            raise ValueError("Need exactly 4 hands of 10 cards each.")

        self.hands: list[list[Card]] = [list(h) for h in hands]
        self.trump: Suit = trump
        self.current_trick: list[TrickCard] = []
        self.trick_history: list[list[TrickCard]] = []
        self.team_scores: list[int] = [0, 0]  # [team_02, team_13]
        self.current_leader: int = first_player
        self.current_player: int = first_player
        self.voids: dict[int, set[Suit]] = {i: set() for i in range(4)}
        self._finished: bool = False

    @property
    def trick_number(self) -> int:
        """Current trick index (0-based)."""
        return len(self.trick_history)

    def is_terminal(self) -> bool:
        """True if all 10 tricks have been played."""
        return self._finished

    def get_result(self) -> GameResult:
        """Return the final game result. Only valid after game ends."""
        if not self._finished:
            raise RuntimeError("Game is not finished yet.")
        scores = (self.team_scores[0], self.team_scores[1])
        game_pts = GameResult.compute_game_points(*scores)
        return GameResult(team_scores=scores, game_points=game_pts)

    def legal_moves(self, seat: int) -> list[Card]:
        """Return the list of legal cards for a player.

        Must follow the led suit if possible; otherwise, any card is legal.
        """
        if seat != self.current_player:
            raise ValueError(
                f"It's seat {self.current_player}'s turn, not seat {seat}'s."
            )

        hand = self.hands[seat]
        if not self.current_trick:
            # Leading the trick — any card is legal.
            return list(hand)

        led_suit = self.current_trick[0].card.suit
        suited = [c for c in hand if c.suit == led_suit]
        return suited if suited else list(hand)

    def get_visible_state(self, seat: int) -> VisibleState:
        """Return the information visible to the given player."""
        led_suit = (
            self.current_trick[0].card.suit if self.current_trick else None
        )
        # If this player is leading, led_suit is None.
        if seat == self.current_leader and not self.current_trick:
            led_suit = None

        return VisibleState(
            seat=seat,
            hand=list(self.hands[seat]),
            trump=self.trump,
            led_suit=led_suit,
            current_trick=list(self.current_trick),
            trick_history=[list(t) for t in self.trick_history],
            team_scores=(self.team_scores[0], self.team_scores[1]),
            voids={s: set(v) for s, v in self.voids.items()},
            trick_number=self.trick_number,
        )

    def play_card(self, seat: int, card: Card) -> int | None:
        """Play a card from the given seat.

        Args:
            seat: The seat playing.
            card: The card to play.

        Returns:
            The seat that won the trick if the trick is complete, else None.

        Raises:
            ValueError: If it's not this seat's turn, the card isn't in hand,
                or the card is illegal (not following suit when possible).
        """
        if self._finished:
            raise RuntimeError("Game is already finished.")
        if seat != self.current_player:
            raise ValueError(
                f"It's seat {self.current_player}'s turn, not seat {seat}'s."
            )
        if card not in self.hands[seat]:
            raise ValueError(f"Card {card!r} is not in seat {seat}'s hand.")

        # Enforce follow-suit.
        legal = self.legal_moves(seat)
        if card not in legal:
            raise ValueError(
                f"Card {card!r} is not a legal play. Legal moves: {legal}"
            )

        # Track voids: if player doesn't follow the led suit, mark void.
        if self.current_trick:
            led_suit = self.current_trick[0].card.suit
            if card.suit != led_suit:
                self.voids[seat].add(led_suit)

        # Remove card from hand and add to trick.
        self.hands[seat].remove(card)
        self.current_trick.append(TrickCard(seat, card))

        # If trick is not complete, advance to next player.
        if len(self.current_trick) < 4:
            self.current_player = _next_seat_after(seat)
            return None

        # Trick is complete — resolve winner.
        winner = self._resolve_trick()
        trick_points = sum(card_points(tc.card) for tc in self.current_trick)

        # Award points to the winning team.
        team_idx = 0 if winner in (0, 2) else 1
        self.team_scores[team_idx] += trick_points

        # Archive trick and reset.
        self.trick_history.append(list(self.current_trick))
        self.current_trick = []
        self.current_leader = winner
        self.current_player = winner

        # Check if game is over.
        if len(self.trick_history) == 10:
            self._finished = True

        return winner

    def _resolve_trick(self) -> int:
        """Determine the winner of the current (complete) trick.

        Highest trump wins. If no trump played, highest of led suit wins.
        """
        led_suit = self.current_trick[0].card.suit

        best_seat = self.current_trick[0].seat
        best_card = self.current_trick[0].card

        for tc in self.current_trick[1:]:
            if self._beats(tc.card, best_card, led_suit):
                best_seat = tc.seat
                best_card = tc.card

        return best_seat

    def _beats(self, challenger: Card, current: Card, led_suit: Suit) -> bool:
        """Check if challenger card beats the current best card.

        Rules:
        - Trump beats non-trump.
        - Higher trump beats lower trump.
        - If neither is trump, only cards of the led suit compete;
          off-suit non-trump cards never win.
        """
        challenger_is_trump = challenger.suit == self.trump
        current_is_trump = current.suit == self.trump

        if challenger_is_trump and not current_is_trump:
            return True
        if not challenger_is_trump and current_is_trump:
            return False
        if challenger_is_trump and current_is_trump:
            return challenger.rank > current.rank

        # Neither is trump.
        if challenger.suit == led_suit and current.suit == led_suit:
            return challenger.rank > current.rank
        if challenger.suit == led_suit:
            return True
        # Challenger is off-suit and non-trump — can't beat anything.
        return False
