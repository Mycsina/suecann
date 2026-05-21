import pytest
import numpy as np
from src.wann.genome import Genome, ConnGene, NodeGene, NodeType
from src.wann.species import compatibility_distance, Species, speciate
from src.wann.mutations import mutate_toggle_connection, mutate_flip_sign
from src.oracle.fitness import evaluate_population
from src.baselines.random_bot import RandomBot
from src.engine.duplicate_loop import generate_deals

def test_topological_sort_caching():
    g = Genome.initial()
    assert g._topo_order is None
    
    # 1. Accessing should compute and cache
    order1 = g.topological_order()
    assert g._topo_order is not None
    assert len(order1) > 0
    
    # 2. Modifying via add_node / add_connection should invalidate cache
    g.add_node(NodeGene.make(50, NodeType.HIDDEN))
    assert g._topo_order is None
    
    g.topological_order()
    assert g._topo_order is not None
    
    g.add_connection(ConnGene.make(100, 0, 50, sign=1, enabled=True))
    assert g._topo_order is None

    g.topological_order()
    assert g._topo_order is not None

    # 3. Mutating properties should invalidate cache
    rng = np.random.default_rng(42)
    mutate_toggle_connection(g, rng)
    assert g._topo_order is None

    g.topological_order()
    assert g._topo_order is not None

    mutate_flip_sign(g, rng)
    assert g._topo_order is None


def test_speciation_coefficients():
    # Create two genomes with matching hidden node but different activation functions
    # and some disjoint connections
    g1 = Genome.initial()
    g2 = Genome.initial()
    
    # Add a dummy matching connection gene to prevent early exit of compatibility distance
    g1.add_connection(ConnGene.make(1, 0, 50, sign=1, enabled=True))
    g2.add_connection(ConnGene.make(1, 0, 50, sign=1, enabled=True))
    
    g1.add_node(NodeGene.make(50, NodeType.HIDDEN, activation_fn=1, aggregation_fn=1))
    g2.add_node(NodeGene.make(50, NodeType.HIDDEN, activation_fn=2, aggregation_fn=1))
    
    # Distance with default coefficients (c1=1, c2=1, c3=0.4)
    dist_default = compatibility_distance(g1, g2, c1=1.0, c2=1.0, c3=0.4)
    
    # Distance with c3=2.0 (mismatch coefficient)
    dist_high_mismatch = compatibility_distance(g1, g2, c1=1.0, c2=1.0, c3=2.0)
    
    # Mismatch should make the distance significantly higher since c3 is larger
    assert dist_high_mismatch > dist_default


def test_parallel_vs_sequential_evaluation():
    g1 = Genome.initial()
    g2 = Genome.initial()
    genomes = [g1, g2]
    
    deals = generate_deals(0, n_deals=2, base_seed=42)
    opponents = [RandomBot(), RandomBot(), RandomBot()]
    
    baseline = RandomBot()

    # Evaluate sequentially
    seq_fit, seq_deltas, seq_illegal = evaluate_population(
        genomes, deals, opponents, baseline, generation=0, base_seed=42, parallel=False
    )

    # Evaluate in parallel
    par_fit, par_deltas, par_illegal = evaluate_population(
        genomes, deals, opponents, baseline, generation=0, base_seed=42, parallel=True
    )

    assert seq_fit == par_fit
    assert seq_deltas == par_deltas
    assert seq_illegal == par_illegal
