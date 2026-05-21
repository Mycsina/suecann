"""WANN genome — gene arrays and network expression via topological sort.

Connection genes [5,N]: (innovation, src, dst, sign, enabled)
Node genes [4,M]: (id, type, activation_fn, aggregation_fn)

Zero-link initialization: 18 input + 1 bias + 5 output nodes, zero connections.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import numpy as np


class NodeType(IntEnum):
    INPUT = 0
    BIAS = 1
    HIDDEN = 2
    OUTPUT = 3


from src.wann.logical_nodes import AggregationFn, ActivationFn


class ConnGene(NamedTuple):
    """Connection gene: (innovation, src, dst, sign, enabled)."""

    innovation: int
    src: int
    dst: int
    sign: int  # +1 or -1
    enabled: bool

    @staticmethod
    def make(
        innovation: int,
        src: int,
        dst: int,
        sign: int = 1,
        enabled: bool = True,
    ) -> "ConnGene":
        if sign not in (-1, 1):
            raise ValueError(f"Sign must be +1 or -1, got {sign}")
        return ConnGene(innovation, src, dst, sign, enabled)


class NodeGene(NamedTuple):
    """Node gene: (id, type, activation_fn, aggregation_fn)."""

    id: int
    node_type: int  # NodeType
    activation_fn: int  # ActivationFn
    aggregation_fn: int  # AggregationFn

    @staticmethod
    def make(
        node_id: int,
        node_type: int,
        activation_fn: int = ActivationFn.IDENTITY,
        aggregation_fn: int = AggregationFn.SUM,
    ) -> "NodeGene":
        return NodeGene(node_id, node_type, activation_fn, aggregation_fn)


# Reserved node IDs.
INPUT_START = 0
INPUT_COUNT = 18  # belief state dimension
BIAS_ID = INPUT_START + INPUT_COUNT  # 18
OUTPUT_START = BIAS_ID + 1  # 19
OUTPUT_COUNT = 5
FIRST_HIDDEN_ID = OUTPUT_START + OUTPUT_COUNT  # 24


def _initial_node_genes() -> list[NodeGene]:
    """Build the 24 fixed nodes (18 inputs, 1 bias, 5 outputs)."""
    nodes: list[NodeGene] = []

    for i in range(INPUT_COUNT):
        nodes.append(
            NodeGene.make(
                INPUT_START + i,
                NodeType.INPUT,
                ActivationFn.IDENTITY,
                AggregationFn.SUM,
            )
        )

    nodes.append(
        NodeGene.make(
            BIAS_ID,
            NodeType.BIAS,
            ActivationFn.IDENTITY,
            AggregationFn.SUM,
        )
    )

    for i in range(OUTPUT_COUNT):
        nodes.append(
            NodeGene.make(
                OUTPUT_START + i,
                NodeType.OUTPUT,
                ActivationFn.IDENTITY,
                AggregationFn.SUM,
            )
        )

    return nodes


def _topological_order(
    node_ids: set[int], connections: list[ConnGene]
) -> list[int]:
    """Return node IDs in topological order (inputs/ bias first, then by depth).

    Uses Kahn's algorithm. All nodes not in connections get depth 0.
    """
    # Build adjacency and in-degree.
    adj: dict[int, list[int]] = {nid: [] for nid in node_ids}
    in_degree: dict[int, int] = {nid: 0 for nid in node_ids}

    for c in connections:
        if c.enabled and c.src in node_ids and c.dst in node_ids:
            adj[c.src].append(c.dst)
            in_degree[c.dst] += 1

    # Start with nodes that have fixed priority ordering.
    fixed_order = (
        list(range(INPUT_START, INPUT_START + INPUT_COUNT))
        + [BIAS_ID]
        + list(range(OUTPUT_START, OUTPUT_START + OUTPUT_COUNT))
    )
    hidden_ids = sorted(
        nid for nid in node_ids if nid >= FIRST_HIDDEN_ID
    )

    # Priority: inputs first, then bias, then hidden, then outputs.
    priority: dict[int, int] = {}
    for i, nid in enumerate(
        list(range(INPUT_START, INPUT_START + INPUT_COUNT))
        + [BIAS_ID]
        + hidden_ids
        + list(range(OUTPUT_START, OUTPUT_START + OUTPUT_COUNT))
    ):
        priority[nid] = i

    # Kahn's algorithm with priority tiebreaker.
    queue = sorted(
        [nid for nid in node_ids if in_degree[nid] == 0],
        key=lambda n: priority.get(n, 999),
    )
    order: list[int] = []
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

    # Any remaining nodes not in adjacency graph.
    for nid in sorted(node_ids, key=lambda n: priority.get(n, 999)):
        if nid not in visited:
            order.append(nid)

    return order


class Genome:
    """A WANN genome: node genes + connection genes + innovation counter."""

    def __init__(
        self,
        node_genes: list[NodeGene] | None = None,
        conn_genes: list[ConnGene] | None = None,
        next_innovation: int = 0,
    ) -> None:
        self.node_genes: dict[int, NodeGene] = {}
        self.conn_genes: dict[int, ConnGene] = {}  # keyed by innovation
        self.next_innovation = next_innovation
        self._topo_order: list[int] | None = None

        if node_genes is not None:
            for ng in node_genes:
                self.node_genes[ng.id] = ng
        else:
            for ng in _initial_node_genes():
                self.node_genes[ng.id] = ng

        if conn_genes is not None:
            for cg in conn_genes:
                self.conn_genes[cg.innovation] = cg

        # Ensure next_innovation is ahead of all existing innovations.
        if self.conn_genes:
            max_inno = max(self.conn_genes.keys())
            if self.next_innovation <= max_inno:
                self.next_innovation = max_inno + 1

    @staticmethod
    def initial() -> "Genome":
        """Create a zero-link initial genome."""
        return Genome()

    @property
    def node_ids(self) -> set[int]:
        return set(self.node_genes.keys())

    @property
    def hidden_ids(self) -> list[int]:
        return sorted(
            nid
            for nid, ng in self.node_genes.items()
            if ng.node_type == NodeType.HIDDEN
        )

    def invalidate_topo_order(self) -> None:
        """Invalidate the cached topological order."""
        self._topo_order = None

    def add_node(self, node: NodeGene) -> None:
        self.node_genes[node.id] = node
        self.invalidate_topo_order()

    def add_connection(self, conn: ConnGene) -> None:
        self.conn_genes[conn.innovation] = conn
        if conn.innovation >= self.next_innovation:
            self.next_innovation = conn.innovation + 1
        self.invalidate_topo_order()

    def has_connection(self, src: int, dst: int) -> bool:
        """Check if a connection exists between src and dst."""
        for c in self.conn_genes.values():
            if c.src == src and c.dst == dst:
                return True
        return False

    def get_connection(self, src: int, dst: int) -> ConnGene | None:
        for c in self.conn_genes.values():
            if c.src == src and c.dst == dst:
                return c
        return None

    def enabled_connections(self) -> list[ConnGene]:
        return [c for c in self.conn_genes.values() if c.enabled]

    def topological_order(self) -> list[int]:
        """Return node IDs in evaluation order."""
        if self._topo_order is None:
            self._topo_order = _topological_order(self.node_ids, list(self.conn_genes.values()))
        return self._topo_order

    def copy(self) -> "Genome":
        """Deep copy."""
        new = Genome(
            node_genes=[
                NodeGene.make(ng.id, ng.node_type, ng.activation_fn, ng.aggregation_fn)
                for ng in self.node_genes.values()
            ],
            conn_genes=[
                ConnGene.make(c.innovation, c.src, c.dst, c.sign, c.enabled)
                for c in self.conn_genes.values()
            ],
            next_innovation=self.next_innovation,
        )
        return new

    def num_enabled(self) -> int:
        return sum(1 for c in self.conn_genes.values() if c.enabled)

    def num_nodes(self) -> int:
        return len(self.node_genes)

    def __repr__(self) -> str:
        return (
            f"Genome(nodes={len(self.node_genes)}, "
            f"conns={len(self.conn_genes)}, "
            f"enabled={self.num_enabled()}, "
            f"hidden={len(self.hidden_ids)})"
        )
