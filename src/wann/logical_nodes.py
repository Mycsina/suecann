"""Logical node primitives for the WANN framework.

Provides aggregation functions (how inputs combine) and activation functions
(post-aggregation transform).  All activation outputs are clamped to [0, 1]
to preserve Boolean semantics needed for clean IF/THEN rule extraction.

Design decisions:
  - SIGMOID removed: breaks human-readable rule extraction.
  - MEAN removed: mean([1,0]) = 0.5 at THRESHOLD boundary causes float
    precision bit-flipping.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


# ---------------------------------------------------------------------------
# Aggregation functions — combine a list of incoming signals into one value.
# ---------------------------------------------------------------------------

class AggregationFn(IntEnum):
    """Available aggregation functions for WANN nodes."""

    SUM = 0
    MIN = 1  # Logical AND
    MAX = 2  # Logical OR


def aggregate(fn_id: int, inputs: list[float] | np.ndarray) -> float:
    """Apply an aggregation function to a list of input signals.

    Args:
        fn_id: AggregationFn enum value.
        inputs: List of float input signals.

    Returns:
        Single aggregated float value.
    """
    if len(inputs) == 0:
        return 0.0

    arr = np.asarray(inputs, dtype=np.float64)

    if fn_id == AggregationFn.SUM:
        return float(np.sum(arr))
    elif fn_id == AggregationFn.MIN:
        return float(np.min(arr))
    elif fn_id == AggregationFn.MAX:
        return float(np.max(arr))
    else:
        raise ValueError(f"Unknown aggregation function ID: {fn_id}")


# ---------------------------------------------------------------------------
# Activation functions — transform aggregated value, clamped to [0, 1].
# ---------------------------------------------------------------------------

class ActivationFn(IntEnum):
    """Available activation functions for WANN nodes."""

    IDENTITY = 0
    NOT = 1
    THRESHOLD = 2


def activate(fn_id: int, x: float, shared_weight: float = 1.0) -> float:
    """Apply an activation function and clamp the result to [0, 1].

    Args:
        fn_id: ActivationFn enum value.
        x: The aggregated input value.
        shared_weight: The shared weight W value.

    Returns:
        Activated float value in [0, 1].
    """
    if fn_id == ActivationFn.IDENTITY:
        value = x
    elif fn_id == ActivationFn.NOT:
        value = 1.0 - _clamp01(x)
    elif fn_id == ActivationFn.THRESHOLD:
        return 1.0 if x > 0.5 * abs(shared_weight) else 0.0
    else:
        raise ValueError(f"Unknown activation function ID: {fn_id}")

    return _clamp01(value)


def _clamp01(x: float) -> float:
    """Clamp a value to the [0, 1] range."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# ---------------------------------------------------------------------------
# Sign-only connection logic.
# ---------------------------------------------------------------------------

def apply_sign(signal: float, sign: int) -> float:
    """Apply sign-based inversion to a signal.

    When sign == +1: pass through unchanged.
    When sign == -1: invert the signal (1.0 - x).

    Args:
        signal: Input signal value (should be in [0, 1]).
        sign: +1 or -1.

    Returns:
        Transformed signal in [0, 1].
    """
    if sign == -1:
        return 1.0 - _clamp01(signal)
    return signal


# Convenience: all available function IDs for mutation sampling.
ALL_AGGREGATIONS = list(AggregationFn)
ALL_ACTIVATIONS = list(ActivationFn)
