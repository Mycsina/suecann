# src/export/export_flowchart.py
"""WANN Rule Compiler and Topology Visualizer.

Compiles weight-agnostic neural networks into human-readable logical rules
and generates executable Python code. Generates Graphviz visualizations of network topologies.
"""

import os
import sys

import json
import numpy as np
from enum import IntEnum
from typing import NamedTuple, List, Dict, Set

import sueca_solver


class NodeType(IntEnum):
    INPUT = 0
    BIAS = 1
    HIDDEN = 2
    OUTPUT = 3


class AggregationFn(IntEnum):
    SUM = 0
    MIN = 1
    MAX = 2


class ActivationFn(IntEnum):
    IDENTITY = 0
    NOT = 1
    THRESHOLD = 2


class ConnGene(NamedTuple):
    innovation: int
    src: int
    dst: int
    sign: int
    enabled: bool

    @staticmethod
    def make(innovation: int, src: int, dst: int, sign: int = 1, enabled: bool = True):
        return ConnGene(innovation, src, dst, sign, enabled)


class NodeGene(NamedTuple):
    id: int
    node_type: int
    activation_fn: int
    aggregation_fn: int

    @staticmethod
    def make(
        node_id: int, node_type: int, activation_fn: int = 0, aggregation_fn: int = 0
    ):
        return NodeGene(node_id, node_type, activation_fn, aggregation_fn)


INPUT_START = sueca_solver.INPUT_START
INPUT_COUNT = sueca_solver.INPUT_COUNT
BIAS_ID = sueca_solver.BIAS_ID
OUTPUT_START = sueca_solver.OUTPUT_START
OUTPUT_COUNT = sueca_solver.OUTPUT_COUNT
FIRST_HIDDEN_ID = sueca_solver.FIRST_HIDDEN_ID

OUTPUT_NAMES = {
    OUTPUT_START + 0: "MAX_FORCE",
    OUTPUT_START + 1: "MIN_FORCE",
    OUTPUT_START + 2: "EFFICIENT_WIN",
    OUTPUT_START + 3: "EQUITY_BUILDER",
}


def _initial_node_genes() -> List[NodeGene]:
    nodes = []
    for i in range(INPUT_COUNT):
        nodes.append(NodeGene.make(INPUT_START + i, NodeType.INPUT))
    nodes.append(NodeGene.make(BIAS_ID, NodeType.BIAS))
    for i in range(OUTPUT_COUNT):
        nodes.append(NodeGene.make(OUTPUT_START + i, NodeType.OUTPUT))
    return nodes


def _topological_order(node_ids: Set[int], connections: List[ConnGene]) -> List[int]:
    adj = {nid: [] for nid in node_ids}
    in_degree = {nid: 0 for nid in node_ids}
    for c in connections:
        if c.enabled and c.src in node_ids and c.dst in node_ids:
            adj[c.src].append(c.dst)
            in_degree[c.dst] += 1
    hidden_ids = sorted(nid for nid in node_ids if nid >= FIRST_HIDDEN_ID)
    priority = {}
    for i, nid in enumerate(
        list(range(INPUT_START, INPUT_START + INPUT_COUNT))
        + [BIAS_ID]
        + hidden_ids
        + list(range(OUTPUT_START, OUTPUT_START + OUTPUT_COUNT))
    ):
        priority[nid] = i

    queue = sorted(
        [nid for nid in node_ids if in_degree[nid] == 0],
        key=lambda n: priority.get(n, 999),
    )
    order = []
    visited = set()
    while queue:
        nid = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        order.append(nid)
        for neighbor in adj.get(nid, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                queue.sort(key=lambda n: priority.get(n, 999))
    for nid in sorted(node_ids, key=lambda n: priority.get(n, 999)):
        if nid not in visited:
            order.append(nid)
    return order


class Genome:
    def __init__(
        self,
        node_genes: List[NodeGene] = None,
        conn_genes: List[ConnGene] = None,
        next_innovation: int = 0,
    ):
        self.node_genes: Dict[int, NodeGene] = {}
        self.conn_genes: Dict[int, ConnGene] = {}
        self.next_innovation = next_innovation
        self._topo_order = None

        if node_genes is not None:
            for ng in node_genes:
                self.node_genes[ng.id] = ng
        else:
            for ng in _initial_node_genes():
                self.node_genes[ng.id] = ng

        if conn_genes is not None:
            for cg in conn_genes:
                self.conn_genes[cg.innovation] = cg

        if self.conn_genes:
            self.next_innovation = max(self.conn_genes.keys()) + 1

    @staticmethod
    def initial() -> Genome:
        return Genome()

    @property
    def node_ids(self) -> Set[int]:
        return set(self.node_genes.keys())

    @property
    def hidden_ids(self) -> List[int]:
        return sorted(
            nid
            for nid, ng in self.node_genes.items()
            if ng.node_type == NodeType.HIDDEN
        )

    def enabled_connections(self) -> List[ConnGene]:
        return [c for c in self.conn_genes.values() if c.enabled]

    def topological_order(self) -> List[int]:
        if self._topo_order is None:
            self._topo_order = _topological_order(
                self.node_ids, list(self.conn_genes.values())
            )
        return self._topo_order

    def add_node(self, node: NodeGene) -> None:
        self.node_genes[node.id] = node
        self._topo_order = None

    def add_connection(self, conn: ConnGene) -> None:
        self.conn_genes[conn.innovation] = conn
        self._topo_order = None


def load_genome(filepath: str) -> Genome:
    if str(filepath).endswith(".json"):
        with open(filepath, "r") as f:
            data = json.load(f)
    else:
        data = np.load(filepath, allow_pickle=False)

    node_ids = data["node_ids"]
    node_types = data["node_types"]
    node_acts = data["node_acts"]
    node_aggs = data["node_aggs"]

    node_genes = []
    for i in range(len(node_ids)):
        node_genes.append(
            NodeGene.make(
                int(node_ids[i]),
                int(node_types[i]),
                int(node_acts[i]),
                int(node_aggs[i]),
            )
        )

    if isinstance(data, dict):
        conn_innovs = data.get("conn_innovs", [])
        conn_srcs = data.get("conn_srcs", [])
        conn_dsts = data.get("conn_dsts", [])
        conn_signs = data.get("conn_signs", [])
        conn_enabled = data.get("conn_enabled", [])
        next_innovation = int(data.get("next_innovation", 0))
    else:
        conn_innovs = data.get("conn_innovs", np.array([], dtype=np.int32))
        conn_srcs = data.get("conn_srcs", np.array([], dtype=np.int32))
        conn_dsts = data.get("conn_dsts", np.array([], dtype=np.int32))
        conn_signs = data.get("conn_signs", np.array([], dtype=np.int32))
        conn_enabled = data.get("conn_enabled", np.array([], dtype=np.int32))
        next_innovation = int(data.get("next_innovation", 0))

    conn_genes = []
    for i in range(len(conn_innovs)):
        conn_genes.append(
            ConnGene.make(
                int(conn_innovs[i]),
                int(conn_srcs[i]),
                int(conn_dsts[i]),
                int(conn_signs[i]),
                bool(conn_enabled[i]),
            )
        )

    return Genome(
        node_genes=node_genes, conn_genes=conn_genes, next_innovation=next_innovation
    )


# Map belief state feature indices to names
FEATURE_NAMES = {
    0: "Has_Led_Suit",
    1: "Has_Trump",
    2: "Led_Suit_Power",
    3: "Trump_Power",
    4: "Hand_Point_Density",
    5: "Am_I_Leading",
    6: "Am_I_Last_To_Play",
    7: "Is_Partner_Winning",
    8: "Trick_Point_Value",
    9: "Has_Trick_Been_Cut",
    10: "Partner_Void_Led",
    11: "Partner_Void_Trump",
    12: "Any_Opp_Void_Led",
    13: "Any_Opp_Void_Trump",
    14: "Led_Suit_Ace_Played",
    15: "Led_Suit_7_Played",
    16: "Trump_Ace_Played",
    17: "Game_Pts_Remaining",
    18: "Trick_Number",
    19: "Trumps_Remaining",
    20: "Score_Delta",
    21: "Side0_Depletion",
    22: "Side0_Ace_Played",
    23: "Side0_7_Played",
    24: "Side1_Depletion",
    25: "Side1_Ace_Played",
    26: "Side1_7_Played",
    27: "Side2_Depletion",
    28: "Side2_Ace_Played",
    29: "Side2_7_Played",
}

# OUTPUT_NAMES is defined at the top using relative keys


def get_reachable_nodes(genome: Genome) -> set[int]:
    """Find all node IDs that are reachable backwards from the output nodes."""
    output_ids = [
        nid for nid, ng in genome.node_genes.items() if ng.node_type == NodeType.OUTPUT
    ]
    reachable = set(output_ids)
    queue = list(reachable)

    incoming = {}
    for c in genome.enabled_connections():
        incoming.setdefault(c.dst, []).append(c.src)

    while queue:
        curr = queue.pop(0)
        for src in incoming.get(curr, []):
            if src not in reachable:
                reachable.add(src)
                queue.append(src)
    return reachable


def compile_rules(genome: Genome, W: float | None = 1.0) -> str:
    """Compile the WANN genome into a human-readable string representation of the rules.

    Args:
        genome: The WANN genome.
        W: The shared weight multiplier. If None, rules are parameterized.

    Returns:
        A formatted text string of the compiled rule logic.
    """
    reachable = get_reachable_nodes(genome)
    order = genome.topological_order()

    output_ids = sorted(
        nid for nid, ng in genome.node_genes.items() if ng.node_type == NodeType.OUTPUT
    )
    output_names = {
        nid: name
        for nid, name in zip(
            output_ids, ["MAX_FORCE", "MIN_FORCE", "EFFICIENT_WIN", "EQUITY_BUILDER"]
        )
    }

    bias_ids = [
        nid for nid, ng in genome.node_genes.items() if ng.node_type == NodeType.BIAS
    ]
    bias_id = bias_ids[0] if bias_ids else BIAS_ID

    # We only need to show active hidden and output nodes
    active_nodes = [
        nid
        for nid in order
        if nid in reachable
        and genome.node_genes[nid].node_type in (NodeType.HIDDEN, NodeType.OUTPUT)
    ]

    lines = []
    w_str = f"W = {W}" if W is not None else "parameterized W"
    lines.append(f"=== Evolved Sueca WANN Strategy Rules ({w_str}) ===")
    lines.append("")

    lines.append("Active Inputs Referenced:")
    referenced_inputs = sorted(
        nid
        for nid in reachable
        if genome.node_genes[nid].node_type in (NodeType.INPUT, NodeType.BIAS)
    )
    for r in referenced_inputs:
        if r == bias_id:
            lines.append(f"  BIAS (Node {bias_id}) = 1.0")
        else:
            lines.append(f"  {FEATURE_NAMES[r]} (Node {r})")
    lines.append("")

    # Separate hidden and output nodes
    hidden_nodes = [
        nid
        for nid in active_nodes
        if genome.node_genes[nid].node_type == NodeType.HIDDEN
    ]

    # Topological order guarantees inputs/hidden evaluated first
    if hidden_nodes:
        lines.append("Active Hidden Logic Steps:")
        for nid in hidden_nodes:
            lines.append(f"  {_format_node_expression(genome, nid, W)}")
        lines.append("")

    lines.append("Decision Rules for Play Intents:")
    for nid in output_ids:
        if nid in reachable:
            lines.append(
                f"  {output_names[nid]} = {_format_node_expression_rhs(genome, nid, W)}"
            )
        else:
            lines.append(f"  {output_names[nid]} = 0.0 (Inactive)")

    return "\n".join(lines)


def _format_node_expression(genome: Genome, nid: int, W: float | None) -> str:
    """Format the full expression of a node: 'name = rhs'."""
    name = f"hidden_{nid}"
    rhs = _format_node_expression_rhs(genome, nid, W)
    return f"{name} = {rhs}"


def _format_node_expression_rhs(genome: Genome, nid: int, W: float | None) -> str:
    """Format the right-hand-side expression of a node."""
    ng = genome.node_genes.get(nid)
    if ng is None:
        return "0.0"

    conns = [c for c in genome.enabled_connections() if c.dst == nid]
    if not conns:
        return "0.0"

    bias_ids = [
        n.id for n in genome.node_genes.values() if n.node_type == NodeType.BIAS
    ]
    bias_id = bias_ids[0] if bias_ids else BIAS_ID

    signals = []
    for c in conns:
        # Format source
        if c.src == bias_id:
            src_str = "1.0"
        elif genome.node_genes[c.src].node_type == NodeType.INPUT:
            src_str = FEATURE_NAMES[c.src]
        else:
            src_str = f"hidden_{c.src}"

        # Apply sign inversion
        if c.sign == -1:
            src_str = f"NOT({src_str})"

        # Apply shared weight scale
        if W is not None:
            if W == 1.0:
                sig_str = src_str
            else:
                sig_str = f"({src_str} * {W})"
        else:
            sig_str = f"({src_str} * W)"
        signals.append(sig_str)

    # Format aggregation function
    if ng.aggregation_fn == AggregationFn.SUM:
        agg_str = " + ".join(signals)
        if len(signals) > 1:
            agg_str = f"({agg_str})"
    elif ng.aggregation_fn == AggregationFn.MIN:
        agg_str = f"AND({', '.join(signals)})"
    elif ng.aggregation_fn == AggregationFn.MAX:
        agg_str = f"OR({', '.join(signals)})"
    else:
        agg_str = "0.0"

    # Format activation function
    if ng.activation_fn == ActivationFn.IDENTITY:
        return agg_str
    elif ng.activation_fn == ActivationFn.NOT:
        return f"NOT({agg_str})"
    elif ng.activation_fn == ActivationFn.THRESHOLD:
        if W is not None:
            threshold_val = 0.5 * W
            if W >= 0:
                return f"THRESHOLD({agg_str} > {threshold_val:.4f})"
            else:
                return f"THRESHOLD({agg_str} < {threshold_val:.4f})"
        else:
            return f"THRESHOLD({agg_str} passes 0.5 * W)"

    return agg_str


def compile_to_python(genome: Genome) -> str:
    """Compile the genome into executable Python code.

    Returns:
        A string containing a valid python function `evaluate_compiled(belief, W)`.
    """
    reachable = get_reachable_nodes(genome)
    order = genome.topological_order()

    output_ids = sorted(
        nid for nid, ng in genome.node_genes.items() if ng.node_type == NodeType.OUTPUT
    )
    output_names = {
        nid: name
        for nid, name in zip(
            output_ids, ["MAX_FORCE", "MIN_FORCE", "EFFICIENT_WIN", "EQUITY_BUILDER"]
        )
    }

    bias_ids = [
        nid for nid, ng in genome.node_genes.items() if ng.node_type == NodeType.BIAS
    ]
    bias_id = bias_ids[0] if bias_ids else BIAS_ID

    active_nodes = [
        nid
        for nid in order
        if nid in reachable
        and genome.node_genes[nid].node_type in (NodeType.HIDDEN, NodeType.OUTPUT)
    ]

    lines = []
    lines.append("import numpy as np")
    lines.append("")
    lines.append("def evaluate_compiled(belief: np.ndarray, W: float) -> np.ndarray:")
    lines.append("    # Helpers")
    lines.append("    def NOT(x): return 1.0 - np.clip(x, 0.0, 1.0)")
    lines.append("    def THRESHOLD(x, w):")
    lines.append("        if w >= 0:")
    lines.append("            return 1.0 if x > 0.5 * w else 0.0")
    lines.append("        else:")
    lines.append("            return 1.0 if x < 0.5 * w else 0.0")
    lines.append("    def SUM(*args): return sum(args)")
    lines.append("    def AND(*args): return min(args) if args else 0.0")
    lines.append("    def OR(*args): return max(args) if args else 0.0")
    lines.append("    def CLAMP(x): return np.clip(x, 0.0, 1.0)")
    lines.append("")

    lines.append("    # Inputs mapping")
    input_ids = sorted(
        nid for nid, ng in genome.node_genes.items() if ng.node_type == NodeType.INPUT
    )
    for r in input_ids:
        lines.append(f"    {FEATURE_NAMES[r]} = float(belief[{r}])")
    lines.append("    BIAS = 1.0")
    lines.append("")

    # Compile logic
    hidden_nodes = [
        nid
        for nid in active_nodes
        if genome.node_genes[nid].node_type == NodeType.HIDDEN
    ]
    if hidden_nodes:
        lines.append("    # Hidden node logic")
        for nid in hidden_nodes:
            lines.append(f"    hidden_{nid} = {_python_node_assignment(genome, nid)}")
        lines.append("")

    lines.append("    # Output node logic")
    for nid in output_ids:
        name = output_names[nid]
        if nid in reachable:
            expr = _python_node_assignment(genome, nid)
            lines.append(f"    {name} = {expr}")
        else:
            lines.append(f"    {name} = 0.0")

    lines.append("")
    lines.append("    return np.array([")
    for nid in output_ids:
        lines.append(f"        {output_names[nid]},")
    lines.append("    ], dtype=np.float64)")

    return "\n".join(lines)


def _python_node_assignment(genome: Genome, nid: int) -> str:
    """Generate the right-hand-side expression for a node in compiled Python code."""
    ng = genome.node_genes.get(nid)
    if ng is None:
        return "0.0"

    conns = [c for c in genome.enabled_connections() if c.dst == nid]
    if not conns:
        return "0.0"

    bias_ids = [
        n.id for n in genome.node_genes.values() if n.node_type == NodeType.BIAS
    ]
    bias_id = bias_ids[0] if bias_ids else BIAS_ID

    signals = []
    for c in conns:
        # Format source
        if c.src == bias_id:
            src_str = "BIAS"
        elif genome.node_genes[c.src].node_type == NodeType.INPUT:
            src_str = FEATURE_NAMES[c.src]
        else:
            src_str = f"hidden_{c.src}"

        # Apply sign inversion
        if c.sign == -1:
            src_str = f"NOT({src_str})"

        # Multiply by W
        sig_str = f"({src_str} * W)"
        signals.append(sig_str)

    # Format aggregation function
    if ng.aggregation_fn == AggregationFn.SUM:
        agg_expr = f"SUM({', '.join(signals)})"
    elif ng.aggregation_fn == AggregationFn.MIN:
        agg_expr = f"AND({', '.join(signals)})"
    elif ng.aggregation_fn == AggregationFn.MAX:
        agg_expr = f"OR({', '.join(signals)})"
    else:
        agg_expr = "0.0"

    # Format activation function
    if ng.activation_fn == ActivationFn.IDENTITY:
        return f"CLAMP({agg_expr})"
    elif ng.activation_fn == ActivationFn.NOT:
        return f"NOT({agg_expr})"
    elif ng.activation_fn == ActivationFn.THRESHOLD:
        return f"THRESHOLD({agg_expr}, W)"

    return agg_expr


def compute_depths(genome: Genome) -> dict[int, int]:
    """Compute the depth of each node in the DAG for visual leveling."""
    depths = {}
    for nid, ng in genome.node_genes.items():
        if ng.node_type in (NodeType.INPUT, NodeType.BIAS):
            depths[nid] = 0

    order = genome.topological_order()
    for nid in order:
        ng = genome.node_genes.get(nid)
        if ng is None or ng.node_type in (NodeType.INPUT, NodeType.BIAS):
            continue

        incoming = [c for c in genome.enabled_connections() if c.dst == nid]
        if not incoming:
            depths[nid] = 1
        else:
            depths[nid] = 1 + max(depths.get(c.src, 0) for c in incoming)

    # Place output nodes at max depth + 1 for clean layout
    max_hidden_depth = max(
        [
            d
            for nid, d in depths.items()
            if genome.node_genes[nid].node_type == NodeType.HIDDEN
        ],
        default=0,
    )
    for nid, ng in genome.node_genes.items():
        if ng.node_type == NodeType.OUTPUT:
            depths[nid] = max_hidden_depth + 1

    return depths


def export_topology_graphviz(genome: Genome, output_path: str) -> None:
    """Export the evolved network topology to a Graphviz PDF/PNG and raw .dot file.

    Handles missing system dependency (dot binary) by gracefully falling back
    to just writing the .dot code.
    """
    try:
        import graphviz
    except ImportError:
        print(
            "Warning: python 'graphviz' library not installed. Cannot export visualization."
        )
        return

    dot = graphviz.Digraph(comment="Evolved WANN Topology", format="png")
    dot.attr(
        rankdir="LR", splines="true", overlap="false", ranksep="1.2", nodesep="0.4"
    )

    # Group by depth
    depths = compute_depths(genome)
    max_depth = max(depths.values(), default=1)
    reachable = get_reachable_nodes(genome)

    output_ids = sorted(
        nid for nid, ng in genome.node_genes.items() if ng.node_type == NodeType.OUTPUT
    )
    output_names = {
        nid: name
        for nid, name in zip(
            output_ids, ["MAX_FORCE", "MIN_FORCE", "EFFICIENT_WIN", "EQUITY_BUILDER"]
        )
    }

    bias_ids = [
        nid for nid, ng in genome.node_genes.items() if ng.node_type == NodeType.BIAS
    ]
    bias_id = bias_ids[0] if bias_ids else BIAS_ID

    # Compile node labels and styles
    for nid, ng in genome.node_genes.items():
        # Only render reachable nodes or output nodes
        if nid not in reachable:
            continue

        if ng.node_type == NodeType.INPUT:
            label = f"{FEATURE_NAMES[nid]}\n(I{nid})"
            dot.node(
                str(nid),
                label,
                shape="ellipse",
                color="#4682b4",
                fillcolor="#f0f8ff",
                style="filled",
                fontname="Helvetica",
            )
        elif ng.node_type == NodeType.BIAS:
            dot.node(
                str(nid),
                f"BIAS\n({bias_id})",
                shape="ellipse",
                color="#7f7f7f",
                fillcolor="#f5f5f5",
                style="filled",
                fontname="Helvetica",
            )
        elif ng.node_type == NodeType.OUTPUT:
            label = f"{output_names[nid]}\n(O{nid})"
            dot.node(
                str(nid),
                label,
                shape="box",
                color="#b22222",
                fillcolor="#fff0f5",
                style="filled,bold",
                fontname="Helvetica-Bold",
            )
        elif ng.node_type == NodeType.HIDDEN:
            # Format activation / aggregation name
            agg_name = AggregationFn(ng.aggregation_fn).name
            act_name = ActivationFn(ng.activation_fn).name
            label = f"H{nid}\n{agg_name}\n{act_name}"
            dot.node(
                str(nid),
                label,
                shape="box",
                color="#d2691e",
                fillcolor="#fff5ee",
                style="filled",
                fontname="Helvetica",
            )

    # Connections
    for c in genome.conn_genes.values():
        if not c.enabled:
            # Omit disabled connections to reduce clutter (Semiology of Graphics: Graphic Density)
            continue

        if c.src not in reachable or c.dst not in reachable:
            # Skip connections involving unreachable nodes
            continue

        # Color and style according to sign
        if c.sign == 1:
            color = "#2e8b57"  # Soft emerald green (positive impact)
            style = "solid"
        else:
            color = "#cd5c5c"  # Soft crimson red (inhibitory impact)
            style = "dashed"

        dot.edge(
            str(c.src),
            str(c.dst),
            color=color,
            style=style,
            penwidth="2.0",
            arrowsize="0.8",
        )

    # Enforce sequence ordering for inputs and outputs via invisible edges to prevent layout shuffling
    active_inputs = sorted(
        [
            nid
            for nid in reachable
            if genome.node_genes[nid].node_type in (NodeType.INPUT, NodeType.BIAS)
        ]
    )
    for i in range(len(active_inputs) - 1):
        dot.edge(str(active_inputs[i]), str(active_inputs[i + 1]), style="invis")

    active_outputs = sorted(
        [
            nid
            for nid in reachable
            if genome.node_genes[nid].node_type == NodeType.OUTPUT
        ]
    )
    for i in range(len(active_outputs) - 1):
        dot.edge(str(active_outputs[i]), str(active_outputs[i + 1]), style="invis")

    # Force grouping by depth levels
    for d in range(max_depth + 1):
        with dot.subgraph() as s:
            s.attr(rank="same")
            for nid, depth in depths.items():
                if depth == d and nid in reachable:
                    s.node(str(nid))

    # Save .dot representation
    dot_dir = os.path.dirname(output_path)
    if dot_dir:
        os.makedirs(dot_dir, exist_ok=True)

    base_path, _ = os.path.splitext(output_path)
    dot_source_path = f"{base_path}.dot"
    with open(dot_source_path, "w") as f:
        f.write(dot.source)
    print(f"Saved raw dot source to {dot_source_path}")

    # Render image
    try:
        dot.render(base_path, cleanup=True)
        print(f"Rendered network topology to {base_path}.png")
    except Exception as e:
        print(
            f"Warning: Could not render image due to Graphviz 'dot' binary missing: {e}"
        )
        print(
            "Note: The raw .dot file remains available and can be viewed online at viz-js.com"
        )


def plot_training_stats(csv_path: str, output_path: str) -> None:
    """Plot the training run data using premium Semiology of Graphics principles."""
    if not os.path.exists(csv_path):
        print(f"Warning: Stats CSV not found at {csv_path}. Skipping stats plotting.")
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print(
            "Warning: matplotlib or pandas library not installed. Cannot plot training stats."
        )
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading training stats CSV: {e}")
        return

    # Clean styling
    plt.rcParams["font.sans-serif"] = "Helvetica, Arial, DejaVu Sans"
    plt.rcParams["font.family"] = "sans-serif"

    # 5-panel layout (Row 1 has 2 panels, Row 2 has 3 panels)
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 6, hspace=0.35, wspace=0.45)

    ax1 = fig.add_subplot(gs[0, 0:3])  # Row 0, cols 0-2 (Phase 0)
    ax2 = fig.add_subplot(gs[0, 3:6])  # Row 0, cols 3-5 (Phase 1)
    ax3 = fig.add_subplot(gs[1, 0:2])  # Row 1, cols 0-1 (Species Count)
    ax4 = fig.add_subplot(gs[1, 2:4])  # Row 1, cols 2-3 (Connections)
    ax5 = fig.add_subplot(gs[1, 4:6])  # Row 1, cols 4-5 (Hidden Nodes)

    # Colors
    c_best = "#1f77b4"  # Premium blue
    c_avg = "#aec7e8"  # Light blue
    c_median = "#ff7f0e"  # Orange
    c_species = "#2ca02c"  # Green
    c_conn = "#9467bd"  # Purple
    c_hidden = "#d62728"  # Red

    # Filter phase data
    phase0_df = df[df["phase"] == 0]
    phase1_df = df[df["phase"] == 1]

    # 1. Phase 0 Supervised Accuracy
    if not phase0_df.empty:
        ax1.plot(
            phase0_df["generation"],
            phase0_df["best_fitness"],
            label="Best Accuracy",
            color=c_best,
            linewidth=2.0,
        )
        if "avg_fitness" in phase0_df.columns:
            ax1.plot(
                phase0_df["generation"],
                phase0_df["avg_fitness"],
                label="Avg Accuracy",
                color=c_avg,
                linewidth=1.5,
                linestyle="--",
            )
        ax1.fill_between(
            phase0_df["generation"], phase0_df["best_fitness"], color=c_best, alpha=0.08
        )
        ax1.set_title(
            "Phase 0: Supervised Pretraining Accuracy",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax1.set_xlabel("Generation", fontsize=10)
        ax1.set_ylabel("Accuracy (PIMC Match)", fontsize=10)
        ax1.legend(loc="lower right", frameon=False, fontsize=9)
    else:
        ax1.text(0.5, 0.5, "No Phase 0 Data Available", ha="center", va="center")
        ax1.set_title(
            "Phase 0: Supervised Pretraining Accuracy",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

    # 2. Phase 1 Self-Play Delta vs HeuristicBot
    if not phase1_df.empty:
        ax2.plot(
            phase1_df["generation"],
            phase1_df["best_delta"],
            label="Best Delta",
            color=c_best,
            linewidth=2.0,
        )
        if "median_delta" in phase1_df.columns:
            ax2.plot(
                phase1_df["generation"],
                phase1_df["median_delta"],
                label="Median Delta",
                color=c_median,
                linewidth=1.5,
            )
        ax2.fill_between(
            phase1_df["generation"], phase1_df["best_delta"], color=c_best, alpha=0.08
        )
        ax2.axhline(
            y=0.0,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label="HeuristicBot Baseline",
        )
        ax2.set_title(
            "Phase 1: Self-Play Delta vs HeuristicBot",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax2.set_xlabel("Generation", fontsize=10)
        ax2.set_ylabel("Game Points Delta", fontsize=10)
        ax2.legend(loc="lower right", frameon=False, fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No Phase 1 Data Available", ha="center", va="center")
        ax2.set_title(
            "Phase 1: Self-Play Delta vs HeuristicBot",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

    # 3. Species count (all generations)
    ax3.plot(df["generation"], df["n_species"], color=c_species, linewidth=1.8)
    ax3.fill_between(df["generation"], df["n_species"], color=c_species, alpha=0.08)
    ax3.set_title("Species Diversity", fontsize=11, fontweight="bold", pad=10)
    ax3.set_xlabel("Generation", fontsize=9)
    ax3.set_ylabel("Number of Species", fontsize=9)

    # Draw phase transition vertical line
    if not phase1_df.empty:
        phase1_start = phase1_df["generation"].min()
        for ax in [ax3, ax4, ax5]:
            ax.axvline(
                x=phase1_start, color="gray", linestyle="--", alpha=0.5, linewidth=1.0
            )

    # 4. Network Connections (best genome)
    ax4.plot(df["generation"], df["n_connections_best"], color=c_conn, linewidth=1.8)
    ax4.fill_between(
        df["generation"], df["n_connections_best"], color=c_conn, alpha=0.08
    )
    ax4.set_title("Best Genome Connections", fontsize=11, fontweight="bold", pad=10)
    ax4.set_xlabel("Generation", fontsize=9)
    ax4.set_ylabel("Connection Count", fontsize=9)

    # 5. Network Hidden Nodes (best genome)
    ax5.plot(df["generation"], df["n_hidden_best"], color=c_hidden, linewidth=1.8)
    ax5.fill_between(df["generation"], df["n_hidden_best"], color=c_hidden, alpha=0.08)
    ax5.set_title("Best Genome Hidden Nodes", fontsize=11, fontweight="bold", pad=10)
    ax5.set_xlabel("Generation", fontsize=9)
    ax5.set_ylabel("Hidden Node Count", fontsize=9)

    # Apply global aesthetics to all subplots
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.grid(True, which="both", color="#f4f4f4", linestyle="-", linewidth=0.8)
        # Hide top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")
        ax.tick_params(colors="#555555", labelsize=9)

    plt.suptitle(
        "Sueca WANN Training Evolution Performance & Structure Summary",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )

    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved training stats visualization to {output_path}")


if __name__ == "__main__":
    import argparse

    pass  # load_genome defined in this module

    parser = argparse.ArgumentParser(
        description="Compile WANN rules and visualize topology"
    )
    parser.add_argument(
        "--genome", type=str, required=True, help="Path to genome .npz file"
    )
    parser.add_argument(
        "--weight", type=float, default=1.0, help="Weight to compile rules for"
    )
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints", help="Output directory"
    )

    args = parser.parse_args()

    if not os.path.exists(args.genome):
        print(f"Error: Genome file not found: {args.genome}")
        sys.exit(1)

    genome = load_genome(args.genome)
    print(
        f"Loaded genome with {len(genome.node_genes)} nodes and {len(genome.conn_genes)} connections."
    )

    # Compile rules
    rules = compile_rules(genome, W=args.weight)
    rules_path = os.path.join(args.output_dir, "compiled_rules.txt")
    with open(rules_path, "w") as f:
        f.write(rules)
    print(rules)
    print(f"\nSaved text rules to {rules_path}")

    # Compile python code
    python_code = compile_to_python(genome)
    code_path = os.path.join(args.output_dir, "evaluate_compiled.py")
    with open(code_path, "w") as f:
        f.write(python_code)
    print(f"Saved executable Python code to {code_path}")

    # Export topology graphviz
    viz_path = os.path.join(args.output_dir, "topology_graph")
    export_topology_graphviz(genome, viz_path)

    # Plot training stats if available
    genome_dir = os.path.dirname(args.genome) or "."
    if os.path.basename(genome_dir) == "genomes":
        parent_dir = os.path.dirname(genome_dir) or "."
    else:
        parent_dir = genome_dir
    stats_csv_path = os.path.join(parent_dir, "training_stats.csv")
    plot_path = os.path.join(args.output_dir, "run_plot.png")
    plot_training_stats(stats_csv_path, plot_path)
