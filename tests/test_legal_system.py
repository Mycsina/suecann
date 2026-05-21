import pytest
import numpy as np
from src.engine.cards import Card, Suit, Rank
from src.engine.sueca_engine import SuecaGame
from src.oracle.legal_system import resolve_intent, Intent


def test_cut_low_legality_when_trump_led():
    # Trump is HEARTS. Led suit is HEARTS.
    # Player 0 has some hearts.
    h0 = [Card(Suit.HEARTS, Rank.TWO), Card(Suit.HEARTS, Rank.THREE)] + [
        Card(Suit.SPADES, r) for r in list(Rank)[:8]
    ]
    h1 = [Card(Suit.DIAMONDS, r) for r in list(Rank)[:10]]
    h2 = [Card(Suit.CLUBS, r) for r in list(Rank)[:10]]
    h3 = [Card(Suit.HEARTS, Rank.ACE), Card(Suit.HEARTS, Rank.KING)] + [
        Card(Suit.SPADES, r) for r in list(Rank)[2:]
    ]

    # Leader is seat 3, who leads A♥
    game = SuecaGame([h0, h1, h2, h3], trump=Suit.HEARTS, first_player=3)
    game.play_card(3, Card(Suit.HEARTS, Rank.ACE))

    # Seats 2 and 1 play
    game.play_card(2, game.legal_moves(2)[0])
    game.play_card(1, game.legal_moves(1)[0])

    # Now it is seat 0's turn. Hearts is led, they have Hearts.
    # CUT_LOW should pick 2♥ (lowest trump) and it should NOT be flagged as illegal.
    rng = np.random.default_rng(42)
    result = resolve_intent(Intent.CUT_LOW, game, 0, rng)

    assert result.card == Card(Suit.HEARTS, Rank.TWO)
    assert not result.was_illegal


def test_cut_low_illegal_when_non_trump_led_and_holding_led_suit():
    # Trump is HEARTS. Led suit is SPADES.
    # Player 0 has Spades (led suit) and Hearts (trump).
    h0 = [Card(Suit.HEARTS, Rank.TWO), Card(Suit.SPADES, Rank.THREE)] + [
        Card(Suit.DIAMONDS, r) for r in list(Rank)[:8]
    ]
    h1 = [Card(Suit.DIAMONDS, r) for r in list(Rank)[:10]]
    h2 = [Card(Suit.CLUBS, r) for r in list(Rank)[:10]]
    h3 = [Card(Suit.SPADES, Rank.ACE), Card(Suit.HEARTS, Rank.KING)] + [
        Card(Suit.SPADES, r) for r in list(Rank)[2:]
    ]

    # Leader is seat 3, who leads A♠
    game = SuecaGame([h0, h1, h2, h3], trump=Suit.HEARTS, first_player=3)
    game.play_card(3, Card(Suit.SPADES, Rank.ACE))

    # Seats 2 and 1 play
    game.play_card(2, game.legal_moves(2)[0])
    game.play_card(1, game.legal_moves(1)[0])

    # Now it is seat 0's turn. Spades is led, they have Spades.
    # CUT_LOW should fail (since they must follow suit) and pick a duck card (3♠)
    # and mark it as illegal.
    rng = np.random.default_rng(42)
    result = resolve_intent(Intent.CUT_LOW, game, 0, rng)

    assert result.card == Card(Suit.SPADES, Rank.THREE)
    assert result.was_illegal


def test_cut_low_legal_when_non_trump_led_and_void_in_led_suit():
    # Trump is HEARTS. Led suit is SPADES.
    # Player 0 has no Spades (void), but has Hearts (trump).
    h0 = [Card(Suit.HEARTS, Rank.TWO), Card(Suit.HEARTS, Rank.THREE)] + [
        Card(Suit.DIAMONDS, r) for r in list(Rank)[:8]
    ]
    h1 = [Card(Suit.DIAMONDS, r) for r in list(Rank)[:10]]
    h2 = [Card(Suit.CLUBS, r) for r in list(Rank)[:10]]
    h3 = [Card(Suit.SPADES, Rank.ACE), Card(Suit.HEARTS, Rank.KING)] + [
        Card(Suit.SPADES, r) for r in list(Rank)[2:]
    ]

    # Leader is seat 3, who leads A♠
    game = SuecaGame([h0, h1, h2, h3], trump=Suit.HEARTS, first_player=3)
    game.play_card(3, Card(Suit.SPADES, Rank.ACE))

    # Seats 2 and 1 play
    game.play_card(2, game.legal_moves(2)[0])
    game.play_card(1, game.legal_moves(1)[0])

    # Now it is seat 0's turn. Spades is led, they are void.
    # They want to CUT_LOW (play lowest trump). They should play 2♥ and it should be legal.
    rng = np.random.default_rng(42)
    result = resolve_intent(Intent.CUT_LOW, game, 0, rng)

    assert result.card == Card(Suit.HEARTS, Rank.TWO)
    assert not result.was_illegal
