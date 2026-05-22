import pytest

from src.benchmark import run_tournament


def test_mini_tournament():
    # Set up a mini tournament between two bots to verify that the logic executes successfully.
    bots = {
        "Random": (0, None),
        "Heuristic": (1, None),
    }

    # Run with 2 duplicate deals
    win_rates, ci_margins, pts, gpts = run_tournament(
        bots,
        n_deals=2,
        base_seed=999,
        use_multiprocessing=False,  # Disable multiprocessing in unit tests for speed and simplicity
    )

    assert win_rates.shape == (2, 2)
    assert ci_margins.shape == (2, 2)
    assert pts.shape == (2, 2)
    assert gpts.shape == (2, 2)

    # Diagonals should be reference values
    assert win_rates[0, 0] == 0.5
    assert win_rates[1, 1] == 0.5
    assert ci_margins[0, 0] == 0.0
    assert ci_margins[1, 1] == 0.0

    # Win rate of bot 0 vs bot 1 should be the exact complement of bot 1 vs bot 0
    assert win_rates[0, 1] + win_rates[1, 0] == pytest.approx(1.0)
