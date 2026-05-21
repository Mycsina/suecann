"""Unit tests for PIMCBot."""

from __future__ import annotations

import os
import tempfile
import numpy as np

from src.baselines.pimc_bot import PIMCBot
from src.engine.cards import Card, Suit
from src.engine.duplicate_loop import generate_deals
from src.engine.sueca_engine import SuecaGame


def test_pimc_bot_select_card():
    # Generate a single deal
    deals = generate_deals(gen=0, n_deals=1, base_seed=123)
    deal = deals[0]
    
    # Initialize game
    game = SuecaGame(hands=deal.hands, trump=deal.trump, first_player=0)
    
    # Instantiate PIMCBot
    bot = PIMCBot(n_worlds=3, search_depth=1, seed=42)
    
    # Select card for first player (seat 0)
    rng = np.random.default_rng(42)
    card = bot.select_card(game, seat=0, rng=rng)
    
    assert isinstance(card, Card)
    assert card in game.legal_moves(0)


def test_pimc_bot_recording():
    deals = generate_deals(gen=0, n_deals=1, base_seed=123)
    deal = deals[0]
    game = SuecaGame(hands=deal.hands, trump=deal.trump, first_player=0)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        record_file = os.path.join(tmpdir, "dataset.jsonl")
        bot = PIMCBot(n_worlds=2, search_depth=0, record_file=record_file, seed=42)
        rng = np.random.default_rng(42)
        
        card = bot.select_card(game, seat=0, rng=rng)
        assert os.path.exists(record_file)
        
        with open(record_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        import json
        record = json.loads(lines[0])
        assert "belief" in record
        assert "evs" in record
        assert len(record["belief"]) == 18
        assert str(card) in record["evs"]
