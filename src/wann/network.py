"""WANN network — forward pass over a genome with shared-weight sweep.

Pipeline per connection:
  signal → sign inversion (±1) → shared weight scaling (W) → aggregate → activate → clamp [0,1]
"""

from __future__ import annotations

import numpy as np

from src.wann.genome import (
    BIAS_ID,
    INPUT_COUNT,
    INPUT_START,
    OUTPUT_COUNT,
    OUTPUT_START,
    Genome,
    NodeType,
)
from src.wann.logical_nodes import activate, aggregate, apply_sign


class WannNetwork:
    """A compiled network ready for inference."""

    def __init__(self, genome: Genome) -> None:
        self.genome = genome
        self._order = genome.topological_order()

        # Precompute incoming connections per node (keyed by dst).
        self._incoming: dict[int, list[tuple[int, int]]] = {}  # dst → [(src, sign)]
        for c in genome.enabled_connections():
            self._incoming.setdefault(c.dst, []).append((c.src, c.sign))

    def forward(self, inputs: np.ndarray, shared_weight: float) -> np.ndarray:
        """Evaluate the network.

        Args:
            inputs: Shape (INPUT_COUNT,) belief state vector in [0, 1].
            shared_weight: W value for weight sweep (0.5, 1.0, or 2.0).

        Returns:
            Shape (OUTPUT_COUNT,) output vector in [0, 1].
        """
        values: dict[int, float] = {}

        # Set input values.
        for i in range(INPUT_COUNT):
            values[INPUT_START + i] = float(np.clip(inputs[i], 0.0, 1.0))

        # Bias node always outputs 1.0.
        values[BIAS_ID] = 1.0

        # Evaluate all nodes in topological order.
        for nid in self._order:
            if nid in values:
                continue  # already set (input or bias)

            ng = self.genome.node_genes.get(nid)
            if ng is None:
                continue

            incoming = self._incoming.get(nid, [])

            if not incoming:
                values[nid] = 0.0
                continue

            # Collect signals: apply sign inversion then shared weight.
            signals: list[float] = []
            for src, sign in incoming:
                src_val = values.get(src, 0.0)
                inverted = apply_sign(src_val, sign)
                signals.append(inverted * shared_weight)

            # Aggregate.
            agg_val = aggregate(ng.aggregation_fn, signals)

            # Activate and clamp.
            out_val = activate(ng.activation_fn, agg_val, shared_weight)

            values[nid] = out_val

        # Gather output values.
        outputs = np.zeros(OUTPUT_COUNT, dtype=np.float64)
        for i in range(OUTPUT_COUNT):
            outputs[i] = values.get(OUTPUT_START + i, 0.0)

        return outputs

    def forward_weight_sweep(
        self, inputs: np.ndarray, weights: list[float] | None = None
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Evaluate at each shared weight and return all outputs + mean.

        Args:
            inputs: Shape (INPUT_COUNT,) belief state vector.
            weights: Sweep values, default [0.5, 1.0, 2.0].

        Returns:
            (per_weight_outputs, mean_output) — mean is for action selection.
        """
        if weights is None:
            weights = [0.5, 1.0, 2.0]

        all_outputs = []
        for w in weights:
            all_outputs.append(self.forward(inputs, w))

        mean_output = np.mean(all_outputs, axis=0)
        return all_outputs, mean_output
