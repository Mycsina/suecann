import os
import numpy as np

from src.export.export_flowchart import (
    Genome,
    NodeGene,
    ConnGene,
    NodeType,
    ActivationFn,
    AggregationFn,
    compile_to_python,
    compile_rules,
    export_topology_graphviz,
)


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


def test_compiler_correctness():
    # 1. Build a small WANN genome with custom logical nodes.
    # We will hook up input 0 (Has_Led_Suit), input 1 (Has_Trump), input 5 (Am_I_Leading)
    # and bias to output 23 (TAKE_CHEAPLY) and output 26 (CUT_LOW).
    genome = Genome.initial()

    # Add a hidden node (ID 27) with MIN (AND) aggregation and IDENTITY activation.
    genome.add_node(
        NodeGene.make(
            27,
            NodeType.HIDDEN,
            ActivationFn.IDENTITY,
            AggregationFn.MIN,
        )
    )

    # Connections to hidden node 27
    # conn from input 0 to hidden 27 (sign=1)
    genome.add_connection(ConnGene.make(1, 0, 27, sign=1, enabled=True))
    # conn from input 1 to hidden 27 (sign=-1)
    genome.add_connection(ConnGene.make(2, 1, 27, sign=-1, enabled=True))

    # Add a connection from hidden 27 to output 23 (TAKE_CHEAPLY)
    genome.add_connection(ConnGene.make(3, 27, 23, sign=1, enabled=True))

    # Add a connection from bias (21) to output 23 (TAKE_CHEAPLY)
    genome.add_connection(ConnGene.make(4, 21, 23, sign=1, enabled=True))

    # Let's add a connection to output 26 (CUT_LOW) with THRESHOLD activation.
    # Output node 26 default activation is IDENTITY, let's modify it to THRESHOLD.
    genome.node_genes[26] = NodeGene.make(
        26,
        NodeType.OUTPUT,
        ActivationFn.THRESHOLD,
        AggregationFn.SUM,
    )
    # conn from input 5 to output 26 (sign=1)
    genome.add_connection(ConnGene.make(5, 5, 26, sign=1, enabled=True))

    # 2. Compile to Python code
    python_code = compile_to_python(genome)

    # 3. Dynamically execute the compiled Python code to get the evaluate_compiled function
    namespace = {}
    exec(python_code, namespace)
    evaluate_compiled = namespace["evaluate_compiled"]

    # 4. Generate WannNetwork model
    net = to_rust_network(genome)

    # 5. Evaluate and verify equality over a grid of weights and random belief states
    rng = np.random.default_rng(12345)
    weights_to_test = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

    for W in weights_to_test:
        for _ in range(20):
            # Generate random belief state
            belief = rng.uniform(0.0, 1.0, size=21)

            # Expected output from reference network
            expected = net.forward(belief.tolist(), W)

            # Output from compiled Python code
            actual = evaluate_compiled(belief, W)

            # Assert they are equal with absolute tolerance
            np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)


def test_export_rules_and_dot_no_crash():
    # Verify that calling visualizer and rule compiler works without throwing exceptions.
    genome = Genome.initial()
    rules_text = compile_rules(genome, W=1.0)
    assert "Active Inputs Referenced:" in rules_text
    assert "DUCK_OR_DUMP" in rules_text

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        dot_path = os.path.join(tmpdir, "test_graph")
        export_topology_graphviz(genome, dot_path)
        assert os.path.exists(f"{dot_path}.dot")
