"""Unit tests for evaluate_wann_accuracy."""

from __future__ import annotations

import numpy as np
import sueca_solver
from src.export.export_flowchart import Genome, ConnGene


def to_rust_network(genome: Genome):
    import sueca_solver

    node_ids = []
    node_types = []
    node_activations = []
    node_aggregations = []
    for nid in sorted(genome.node_genes.keys()):
        ng = genome.node_genes[nid]
        node_ids.append(ng.id)
        node_types.append(ng.node_type)
        node_activations.append(ng.activation_fn)
        node_aggregations.append(ng.aggregation_fn)

    conn_srcs = []
    conn_dsts = []
    conn_signs = []
    conn_enableds = []
    for c in genome.conn_genes.values():
        conn_srcs.append(c.src)
        conn_dsts.append(c.dst)
        conn_signs.append(c.sign)
        conn_enableds.append(c.enabled)

    return sueca_solver.PyWannNetwork.from_genome(
        node_ids,
        node_types,
        node_activations,
        node_aggregations,
        conn_srcs,
        conn_dsts,
        conn_signs,
        conn_enableds,
    )


def test_evaluate_wann_accuracy():
    # 1. Create a couple of genomes
    g1 = Genome.initial()
    g1.add_connection(
        ConnGene.make(100, 0, 22, sign=1, enabled=True)
    )  # connect input 0 to output 22 (DUCK_OR_DUMP)

    g2 = Genome.initial()
    g2.add_connection(
        ConnGene.make(200, 1, 23, sign=1, enabled=True)
    )  # connect input 1 to output 23 (TAKE_CHEAPLY)

    genomes = [g1, g2]
    candidate_networks = [to_rust_network(g) for g in genomes]

    # 2. Setup mock dataset: 10 states
    # Input dimension: 21
    states = np.zeros((10, 21), dtype=np.float64)
    # Let state 0 have input 0 positive, and intent 0 legal.
    states[0, 0] = 1.0
    # Let state 1 have input 1 positive, and intent 1 legal.
    states[1, 1] = 1.0

    # Targets:
    intents = np.zeros(10, dtype=np.uint8)
    intents[0] = 0  # target: DUCK_OR_DUMP
    intents[1] = 1  # target: TAKE_CHEAPLY

    # Legal masks:
    # bit 0 = 1 (1), bit 1 = 2 (2) -> 3 (both legal)
    legal_masks = np.ones(10, dtype=np.uint8) * 3

    sweep_weights = [1.0]

    # 3. Call evaluate_wann_accuracy
    accuracies = sueca_solver.evaluate_wann_accuracy(
        candidate_networks, states, intents, legal_masks, sweep_weights
    )

    assert len(accuracies) == 2
    # Both accuracies should be float values in range [0, 1]
    for acc in accuracies:
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0
