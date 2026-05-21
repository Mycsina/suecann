"""Perfect Information Monte Carlo (PIMC) solver for Sueca.

Wraps the optimized Rust sueca_solver engine to perform high-speed hand sampling,
Alpha-Beta minimax search, and Rayon parallelization.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Set, Tuple

import numpy as np

import sueca_solver
from src.engine.cards import Card, Suit, Rank
from src.engine.duplicate_loop import Bot
from src.engine.sueca_engine import SuecaGame
from src.engine.belief_state import encode


def card_to_u8(card: Card) -> int:
    return int(card.suit) * 10 + int(card.rank)


def u8_to_card(idx: int) -> Card:
    return Card(Suit(idx // 10), Rank(idx % 10))


def get_trick_winner_and_best(game: SuecaGame) -> Tuple[int, int]:
    if not game.current_trick:
        return game.current_player, 40

    trump = game.trump
    led_suit = game.current_trick[0].card.suit

    best_seat = game.current_trick[0].seat
    best_card = game.current_trick[0].card

    def beats(challenger: Card, current: Card) -> bool:
        ch_suit = challenger.suit
        ch_rank = challenger.rank
        cur_suit = current.suit
        cur_rank = current.rank

        ch_is_trump = (ch_suit == trump)
        cur_is_trump = (cur_suit == trump)

        if ch_is_trump and not cur_is_trump:
            return True
        if not ch_is_trump and cur_is_trump:
            return False
        if ch_is_trump and cur_is_trump:
            return ch_rank > cur_rank

        # Neither is trump
        if ch_suit == led_suit and cur_suit == led_suit:
            return ch_rank > cur_rank
        if ch_suit == led_suit:
            return True
        return False

    for tc in game.current_trick[1:]:
        if beats(tc.card, best_card):
            best_seat = tc.seat
            best_card = tc.card

    return best_seat, card_to_u8(best_card)


class PIMCBot(Bot):
    """PIMC Bot leveraging the high-performance Rust sueca_solver engine."""

    def __init__(
        self,
        n_worlds: int = 10,
        search_depth: int = 1,  # tricks to search (1 trick = 4 plies)
        record_file: str | None = None,
        seed: int = 42,
    ):
        self.n_worlds = n_worlds
        self.search_depth = search_depth
        self.record_file = record_file
        self.seed = seed

    def select_card(
        self, game: SuecaGame, seat: int, rng: np.random.Generator
    ) -> Card:
        legal = game.legal_moves(seat)
        if len(legal) == 1:
            return legal[0]

        # Calculate EV for each legal move using Rust solver
        move_evs = self.evaluate_moves(game, seat)

        # Choose best move (argmax EV)
        best_move = max(move_evs, key=move_evs.get)  # type: ignore

        # Record data if requested
        if self.record_file:
            self._record_decision(game, seat, move_evs)

        return best_move

    def evaluate_moves(self, game: SuecaGame, seat: int) -> Dict[Card, float]:
        """Compute the Expected Value (EV) of each legal move using PIMC via Rust."""
        state = self.to_rust_state(game, seat)
        
        # Call the Rust solver
        # Rust solver returns a list of (card_u8, ev_score) tuples
        rust_evs = sueca_solver.solve(
            state,
            n_worlds=self.n_worlds,
            search_depth=self.search_depth,
            seed=self.seed
        )

        ev_dict = {}
        for c_idx, ev in rust_evs:
            ev_dict[u8_to_card(c_idx)] = ev

        return ev_dict

    def to_rust_state(self, game: SuecaGame, seat: int) -> sueca_solver.SuecaState:
        """Convert SuecaGame Python state to sueca_solver.SuecaState."""
        my_hand_u8 = [card_to_u8(c) for c in game.hands[seat]]
        
        played_cards_u8 = []
        for trick in game.trick_history:
            for tc in trick:
                played_cards_u8.append(card_to_u8(tc.card))
        for tc in game.current_trick:
            played_cards_u8.append(card_to_u8(tc.card))

        voids_mask = [0] * 4
        for s in range(4):
            if s in game.voids:
                for suit in game.voids[s]:
                    voids_mask[s] |= (1 << int(suit))

        target_sizes = [len(game.hands[s]) for s in range(4)]

        trump_val = int(game.trump)

        led_suit_val = 4
        if game.current_trick:
            led_suit_val = int(game.current_trick[0].card.suit)

        current_trick_u8 = [card_to_u8(tc.card) for tc in game.current_trick]

        current_player_val = game.current_player
        current_trick_winner_val, current_trick_best_card_val = get_trick_winner_and_best(game)

        team_scores = [game.team_scores[0], game.team_scores[1]]
        trick_number_val = game.trick_number

        return sueca_solver.SuecaState(
            my_seat=seat,
            my_hand=my_hand_u8,
            played_cards=played_cards_u8,
            voids=voids_mask,
            target_sizes=target_sizes,
            trump=trump_val,
            led_suit=led_suit_val,
            current_trick=current_trick_u8,
            current_player=current_player_val,
            current_trick_winner=current_trick_winner_val,
            current_trick_best_card=current_trick_best_card_val,
            team_scores=team_scores,
            trick_number=trick_number_val,
        )

    def _record_decision(self, game: SuecaGame, seat: int, move_evs: Dict[Card, float]) -> None:
        """Record the belief state and move EVs for offline training."""
        assert self.record_file is not None
        state = game.get_visible_state(seat)
        belief = encode(state).tolist()

        # Serialize Card objects to string keys
        ev_record = {str(card): ev for card, ev in move_evs.items()}

        record = {
            "belief": belief,
            "evs": ev_record,
            "seat": seat,
            "trump": str(game.trump),
        }

        # Append to jsonl file
        try:
            with open(self.record_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except IOError:
            pass

    def reset(self) -> None:
        pass
