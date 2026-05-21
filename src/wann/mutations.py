"""WANN mutation operators.

Available mutations:
  - Add node: split an existing enabled connection, insert hidden node.
  - Add connection: create a new connection between two unconnected nodes.
  - Toggle connection: enable/disable a random connection.
  - Flip sign: flip the sign of a random connection.
  - Change activation: re-roll activation function of a random node.
  - Change aggregation: re-roll aggregation function of a random node.

No weight mutation — weights are sign-only and shared across the network.
"""

from __future__ import annotations

import numpy as np

from src.wann.genome import (
    FIRST_HIDDEN_ID,
    OUTPUT_START,
    BIAS_ID,
    ConnGene,
    Genome,
    NodeGene,
    NodeType,
)
from src.wann.logical_nodes import (
    ALL_ACTIVATIONS,
    ALL_AGGREGATIONS,
    ActivationFn,
    AggregationFn,
)


def _pick_random_connection(
    genome: Genome, rng: np.random.Generator
) -> ConnGene | None:
    """Pick a random connection from the genome."""
    conns = list(genome.conn_genes.values())
    if not conns:
        return None
    return conns[rng.integers(len(conns))]


def _pick_random_enabled_connection(
    genome: Genome, rng: np.random.Generator
) -> ConnGene | None:
    """Pick a random enabled connection."""
    enabled = [c for c in genome.conn_genes.values() if c.enabled]
    if not enabled:
        return None
    return enabled[rng.integers(len(enabled))]


def _pick_random_node(
    genome: Genome,
    rng: np.random.Generator,
    exclude_inputs: bool = True,
    exclude_bias: bool = True,
    exclude_outputs: bool = True,
) -> NodeGene | None:
    """Pick a random node gene."""
    candidates = list(genome.node_genes.values())
    if exclude_inputs:
        candidates = [n for n in candidates if n.node_type != NodeType.INPUT]
    if exclude_bias:
        candidates = [n for n in candidates if n.node_type != NodeType.BIAS]
    if exclude_outputs:
        candidates = [n for n in candidates if n.node_type != NodeType.OUTPUT]
    if not candidates:
        return None
    return candidates[rng.integers(len(candidates))]


def mutate_add_node(genome: Genome, rng: np.random.Generator) -> bool:
    """Split a random enabled connection, inserting a new hidden node.

    The original connection (src→dst) is disabled. Two new connections are
    added: src→new_node (sign=+1) and new_node→dst (sign=original_sign).
    The new node uses IDENTITY activation and SUM aggregation.
    """
    conn = _pick_random_enabled_connection(genome, rng)
    if conn is None:
        return False

    # Disable the old connection.
    genome.conn_genes[conn.innovation] = ConnGene.make(
        conn.innovation, conn.src, conn.dst, conn.sign, enabled=False
    )
    genome.invalidate_topo_order()

    # Create a new hidden node.
    new_id = genome.node_ids
    new_id_max = max(new_id) if new_id else FIRST_HIDDEN_ID - 1
    node_id = max(new_id_max + 1, FIRST_HIDDEN_ID)
    new_node = NodeGene.make(
        node_id, NodeType.HIDDEN, ActivationFn.IDENTITY, AggregationFn.SUM
    )
    genome.add_node(new_node)

    # Add src → new_node (sign=+1).
    inno1 = genome.next_innovation
    genome.add_connection(ConnGene.make(inno1, conn.src, node_id, sign=1))
    genome.next_innovation = max(genome.next_innovation, inno1 + 1)

    # Add new_node → dst (sign=original_sign).
    inno2 = genome.next_innovation
    genome.add_connection(ConnGene.make(inno2, node_id, conn.dst, sign=conn.sign))
    genome.next_innovation = max(genome.next_innovation, inno2 + 1)

    return True


def mutate_add_connection(genome: Genome, rng: np.random.Generator) -> bool:
    """Add a new connection between two unconnected nodes.

    Avoids creating cycles by only allowing connections from lower-depth
    nodes to higher-depth nodes (or same depth for recurrent — but we
    keep it feed-forward for simplicity).
    """
    order = genome.topological_order()
    depth = {nid: i for i, nid in enumerate(order)}

    # Candidate pairs: src must come before dst in topological order.
    candidates: list[tuple[int, int]] = []
    existing = {(c.src, c.dst) for c in genome.conn_genes.values()}

    for src in order:
        for dst in order:
            if depth[src] >= depth[dst]:
                continue
            if (src, dst) in existing:
                continue
            # Don't connect output->anything.
            src_node = genome.node_genes.get(src)
            if src_node is not None and src_node.node_type == NodeType.OUTPUT:
                continue
            # Don't connect anything->input or anything->bias.
            dst_node = genome.node_genes.get(dst)
            if dst_node is not None and dst_node.node_type in (
                NodeType.INPUT,
                NodeType.BIAS,
            ):
                continue
            candidates.append((src, dst))

    if not candidates:
        return False

    src, dst = candidates[rng.integers(len(candidates))]
    sign = rng.choice([-1, 1])
    inno = genome.next_innovation
    genome.add_connection(ConnGene.make(inno, src, dst, sign=sign))
    genome.next_innovation = max(genome.next_innovation, inno + 1)
    return True


def mutate_toggle_connection(genome: Genome, rng: np.random.Generator) -> bool:
    """Enable or disable a random connection."""
    conn = _pick_random_connection(genome, rng)
    if conn is None:
        return False
    genome.conn_genes[conn.innovation] = ConnGene.make(
        conn.innovation, conn.src, conn.dst, conn.sign, enabled=not conn.enabled
    )
    genome.invalidate_topo_order()
    return True


def mutate_flip_sign(genome: Genome, rng: np.random.Generator) -> bool:
    """Flip the sign of a random connection (+1 → -1 or vice versa)."""
    conn = _pick_random_connection(genome, rng)
    if conn is None:
        return False
    new_sign = -conn.sign
    genome.conn_genes[conn.innovation] = ConnGene.make(
        conn.innovation, conn.src, conn.dst, sign=new_sign, enabled=conn.enabled
    )
    genome.invalidate_topo_order()
    return True


def mutate_change_activation(genome: Genome, rng: np.random.Generator) -> bool:
    """Change the activation function of a random non-input node."""
    node = _pick_random_node(genome, rng)
    if node is None:
        return False
    current = node.activation_fn
    choices = [a for a in ALL_ACTIVATIONS if a != current]
    new_fn = choices[rng.integers(len(choices))]
    genome.node_genes[node.id] = NodeGene.make(
        node.id, node.node_type, new_fn, node.aggregation_fn
    )
    return True


def mutate_change_aggregation(genome: Genome, rng: np.random.Generator) -> bool:
    """Change the aggregation function of a random non-input node."""
    node = _pick_random_node(genome, rng)
    if node is None:
        return False
    current = node.aggregation_fn
    choices = [a for a in ALL_AGGREGATIONS if a != current]
    new_fn = choices[rng.integers(len(choices))]
    genome.node_genes[node.id] = NodeGene.make(
        node.id, node.node_type, node.activation_fn, new_fn
    )
    return True


# All mutation operators for sampling.
MUTATIONS = [
    mutate_add_node,
    mutate_add_connection,
    mutate_toggle_connection,
    mutate_flip_sign,
    mutate_change_activation,
    mutate_change_aggregation,
]
