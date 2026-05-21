"""Thorough tests for src/wann/logical_nodes.py."""
from __future__ import annotations
import pytest
from src.wann.logical_nodes import (
    AggregationFn, ActivationFn,
    aggregate, activate, apply_sign, _clamp01,
    ALL_AGGREGATIONS, ALL_ACTIVATIONS,
)


# ─── Clamping ───────────────────────────────────────────────────────────────

class TestClamp:
    def test_clamp_negative(self):
        assert _clamp01(-0.5) == 0.0

    def test_clamp_above_one(self):
        assert _clamp01(1.5) == 1.0

    def test_clamp_in_range(self):
        assert _clamp01(0.5) == 0.5

    def test_clamp_boundaries(self):
        assert _clamp01(0.0) == 0.0
        assert _clamp01(1.0) == 1.0


# ─── Aggregation functions ─────────────────────────────────────────────────

class TestAggregationEmpty:
    """All aggregation functions return 0.0 for empty inputs."""

    @pytest.mark.parametrize("fn", list(AggregationFn))
    def test_empty_returns_zero(self, fn):
        assert aggregate(fn, []) == 0.0


class TestAggregationSingle:
    """With a single input, behavior should be predictable."""

    def test_sum_single(self):
        assert aggregate(AggregationFn.SUM, [0.7]) == pytest.approx(0.7)

    def test_min_single(self):
        assert aggregate(AggregationFn.MIN, [0.7]) == pytest.approx(0.7)

    def test_max_single(self):
        assert aggregate(AggregationFn.MAX, [0.7]) == pytest.approx(0.7)


class TestAggregationMultiple:
    def test_sum(self):
        assert aggregate(AggregationFn.SUM, [0.3, 0.7]) == pytest.approx(1.0)

    def test_sum_multiple(self):
        assert aggregate(AggregationFn.SUM, [0.2, 0.3, 0.5]) == pytest.approx(1.0)

    def test_min_and_semantics(self):
        """MIN acts as logical AND: min(1, 1) = 1, min(1, 0) = 0."""
        assert aggregate(AggregationFn.MIN, [1.0, 1.0]) == 1.0
        assert aggregate(AggregationFn.MIN, [1.0, 0.0]) == 0.0
        assert aggregate(AggregationFn.MIN, [0.0, 0.0]) == 0.0

    def test_max_or_semantics(self):
        """MAX acts as logical OR: max(0, 1) = 1, max(0, 0) = 0."""
        assert aggregate(AggregationFn.MAX, [0.0, 1.0]) == 1.0
        assert aggregate(AggregationFn.MAX, [0.0, 0.0]) == 0.0
        assert aggregate(AggregationFn.MAX, [1.0, 1.0]) == 1.0

    def test_min_with_floats(self):
        assert aggregate(AggregationFn.MIN, [0.3, 0.7, 0.5]) == pytest.approx(0.3)

    def test_max_with_floats(self):
        assert aggregate(AggregationFn.MAX, [0.3, 0.7, 0.5]) == pytest.approx(0.7)


class TestAggregationInvalid:
    def test_unknown_id_raises(self):
        with pytest.raises(ValueError, match="Unknown aggregation"):
            aggregate(99, [0.5])


# ─── Activation functions ──────────────────────────────────────────────────

class TestActivationIdentity:
    def test_passthrough_in_range(self):
        assert activate(ActivationFn.IDENTITY, 0.5) == pytest.approx(0.5)

    def test_clamps_negative(self):
        assert activate(ActivationFn.IDENTITY, -0.3) == 0.0

    def test_clamps_above_one(self):
        assert activate(ActivationFn.IDENTITY, 1.5) == 1.0

    def test_zero(self):
        assert activate(ActivationFn.IDENTITY, 0.0) == 0.0

    def test_one(self):
        assert activate(ActivationFn.IDENTITY, 1.0) == 1.0


class TestActivationNot:
    def test_not_zero(self):
        assert activate(ActivationFn.NOT, 0.0) == 1.0

    def test_not_one(self):
        assert activate(ActivationFn.NOT, 1.0) == 0.0

    def test_not_half(self):
        assert activate(ActivationFn.NOT, 0.5) == pytest.approx(0.5)

    def test_not_clamps_input(self):
        """NOT of negative should be NOT(0) = 1."""
        assert activate(ActivationFn.NOT, -0.5) == 1.0

    def test_not_clamps_high_input(self):
        """NOT of >1 should be NOT(1) = 0."""
        assert activate(ActivationFn.NOT, 1.5) == 0.0


class TestActivationThreshold:
    def test_above_threshold(self):
        assert activate(ActivationFn.THRESHOLD, 0.6) == 1.0

    def test_below_threshold(self):
        assert activate(ActivationFn.THRESHOLD, 0.4) == 0.0

    def test_at_threshold(self):
        """Exactly 0.5 → 0.0 (strictly greater than)."""
        assert activate(ActivationFn.THRESHOLD, 0.5) == 0.0

    def test_just_above(self):
        assert activate(ActivationFn.THRESHOLD, 0.500001) == 1.0

    def test_negative(self):
        assert activate(ActivationFn.THRESHOLD, -1.0) == 0.0

    def test_large_positive(self):
        assert activate(ActivationFn.THRESHOLD, 100.0) == 1.0


class TestActivationThresholdScaling:
    def test_above_threshold_scaled(self):
        # With shared_weight = 0.5, threshold is 0.25. 0.3 should fire.
        assert activate(ActivationFn.THRESHOLD, 0.3, shared_weight=0.5) == 1.0

    def test_below_threshold_scaled(self):
        # With shared_weight = 0.5, threshold is 0.25. 0.2 should not fire.
        assert activate(ActivationFn.THRESHOLD, 0.2, shared_weight=0.5) == 0.0

    def test_above_threshold_scaled_large(self):
        # With shared_weight = 2.0, threshold is 1.0. 1.1 should fire.
        assert activate(ActivationFn.THRESHOLD, 1.1, shared_weight=2.0) == 1.0

    def test_below_threshold_scaled_large(self):
        # With shared_weight = 2.0, threshold is 1.0. 0.9 should not fire.
        assert activate(ActivationFn.THRESHOLD, 0.9, shared_weight=2.0) == 0.0


class TestActivationInvalid:
    def test_unknown_id_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            activate(99, 0.5)


# ─── Compound logical behavior ─────────────────────────────────────────────

class TestCompoundLogic:
    """Verify that aggregation + activation compose to correct logical gates."""

    def test_and_gate(self):
        """MIN + THRESHOLD = AND gate."""
        for a, b, expected in [(1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 0)]:
            agg = aggregate(AggregationFn.MIN, [float(a), float(b)])
            result = activate(ActivationFn.THRESHOLD, agg)
            assert result == float(expected), f"AND({a},{b}) = {result}, expected {expected}"

    def test_or_gate(self):
        """MAX + THRESHOLD = OR gate."""
        for a, b, expected in [(1, 1, 1), (1, 0, 1), (0, 1, 1), (0, 0, 0)]:
            agg = aggregate(AggregationFn.MAX, [float(a), float(b)])
            result = activate(ActivationFn.THRESHOLD, agg)
            assert result == float(expected), f"OR({a},{b}) = {result}, expected {expected}"

    def test_not_gate(self):
        """NOT activation inverts Boolean signals."""
        assert activate(ActivationFn.NOT, 1.0) == 0.0
        assert activate(ActivationFn.NOT, 0.0) == 1.0

    def test_nand_via_min_not(self):
        """MIN + NOT = NAND gate."""
        for a, b, expected in [(1, 1, 0), (1, 0, 1), (0, 1, 1), (0, 0, 1)]:
            agg = aggregate(AggregationFn.MIN, [float(a), float(b)])
            result = activate(ActivationFn.NOT, agg)
            assert result == float(expected), f"NAND({a},{b}) = {result}, expected {expected}"


# ─── Sign-only connections ──────────────────────────────────────────────────

class TestSignLogic:
    def test_positive_passthrough(self):
        assert apply_sign(0.7, +1) == 0.7

    def test_negative_inverts(self):
        assert apply_sign(0.7, -1) == pytest.approx(0.3)

    def test_negative_of_zero(self):
        assert apply_sign(0.0, -1) == 1.0

    def test_negative_of_one(self):
        assert apply_sign(1.0, -1) == 0.0

    def test_sign_plus_min_gives_and_not(self):
        """MIN(NOT(a), b) via sign inversion + MIN aggregation."""
        a, b = 1.0, 1.0
        inv_a = apply_sign(a, -1)  # NOT(1) = 0
        result = aggregate(AggregationFn.MIN, [inv_a, b])
        assert result == 0.0  # AND(NOT(1), 1) = 0

        a, b = 0.0, 1.0
        inv_a = apply_sign(a, -1)  # NOT(0) = 1
        result = aggregate(AggregationFn.MIN, [inv_a, b])
        assert result == 1.0  # AND(NOT(0), 1) = 1


# ─── Enum completeness ─────────────────────────────────────────────────────

class TestEnumCompleteness:
    def test_all_aggregations_list(self):
        assert len(ALL_AGGREGATIONS) == 3
        assert set(ALL_AGGREGATIONS) == {AggregationFn.SUM, AggregationFn.MIN, AggregationFn.MAX}

    def test_all_activations_list(self):
        assert len(ALL_ACTIVATIONS) == 3
        assert set(ALL_ACTIVATIONS) == {ActivationFn.IDENTITY, ActivationFn.NOT, ActivationFn.THRESHOLD}
