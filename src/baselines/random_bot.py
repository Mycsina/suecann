"""Random baseline bot — uniformly picks a legal card."""

from __future__ import annotations

import numpy as np

from src.engine.cards import Card
from src.engine.sueca_engine import SuecaGame


class RandomBot:
    """Bot that selects uniformly among legal moves."""

    def select_card(self, game: SuecaGame, seat: int, rng: np.random.Generator) -> Card:
        legal = game.legal_moves(seat)
        return legal[rng.integers(len(legal))]

    def reset(self) -> None:
        pass
