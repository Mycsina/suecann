"""Hall of Fame — frozen champion archive.

Stores champion genomes from past generations, provides opponent sampling
for curriculum and self-play training.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field

import numpy as np

from src.wann.genome import (
    BIAS_ID,
    FIRST_HIDDEN_ID,
    INPUT_COUNT,
    INPUT_START,
    OUTPUT_COUNT,
    OUTPUT_START,
    ConnGene,
    Genome,
    NodeGene,
    NodeType,
)
from src.wann.logical_nodes import ActivationFn, AggregationFn


@dataclass
class HoFEntry:
    """A single Hall of Fame entry."""

    genome: Genome
    fitness: float
    generation: int


class HallOfFame:
    """Frozen champion archive."""

    def __init__(self, max_size: int = 20) -> None:
        self.max_size = max_size
        self.entries: list[HoFEntry] = []

    def add(self, genome: Genome, fitness: float, generation: int) -> None:
        """Add a champion genome to the archive."""
        # Avoid duplicates — check if similar fitness already exists.
        entry = HoFEntry(genome=genome.copy(), fitness=fitness, generation=generation)
        self.entries.append(entry)
        # Keep sorted by fitness descending.
        self.entries.sort(key=lambda e: e.fitness, reverse=True)
        # Trim to max_size.
        if len(self.entries) > self.max_size:
            self.entries = self.entries[: self.max_size]

    def sample(self, rng: np.random.Generator, n: int = 1) -> list[Genome]:
        """Sample n champions uniformly at random."""
        if not self.entries:
            return []
        indices = rng.choice(len(self.entries), size=min(n, len(self.entries)), replace=False)
        return [self.entries[i].genome.copy() for i in indices]

    def best(self) -> HoFEntry | None:
        """Return the best entry or None."""
        return self.entries[0] if self.entries else None

    def __len__(self) -> int:
        return len(self.entries)

    def save(self, filepath: str) -> None:
        """Save Hall of Fame to a .npz file."""
        # Serialize each genome into arrays.
        all_node_ids = []
        all_node_types = []
        all_node_acts = []
        all_node_aggs = []
        all_conn_innovs = []
        all_conn_srcs = []
        all_conn_dsts = []
        all_conn_signs = []
        all_conn_enabled = []
        node_entry_indices = []
        conn_entry_indices = []

        for ei, entry in enumerate(self.entries):
            g = entry.genome
            for nid, ng in g.node_genes.items():
                all_node_ids.append(ng.id)
                all_node_types.append(ng.node_type)
                all_node_acts.append(ng.activation_fn)
                all_node_aggs.append(ng.aggregation_fn)
                node_entry_indices.append(ei)
            for c in g.conn_genes.values():
                all_conn_innovs.append(c.innovation)
                all_conn_srcs.append(c.src)
                all_conn_dsts.append(c.dst)
                all_conn_signs.append(c.sign)
                all_conn_enabled.append(1 if c.enabled else 0)
                conn_entry_indices.append(ei)

        fitnesses = np.array([e.fitness for e in self.entries], dtype=np.float32)
        generations = np.array([e.generation for e in self.entries], dtype=np.int32)

        np.savez(
            filepath,
            fitnesses=fitnesses,
            generations=generations,
            node_ids=np.array(all_node_ids, dtype=np.int32),
            node_types=np.array(all_node_types, dtype=np.int32),
            node_acts=np.array(all_node_acts, dtype=np.int32),
            node_aggs=np.array(all_node_aggs, dtype=np.int32),
            conn_innovs=np.array(all_conn_innovs, dtype=np.int32),
            conn_srcs=np.array(all_conn_srcs, dtype=np.int32),
            conn_dsts=np.array(all_conn_dsts, dtype=np.int32),
            conn_signs=np.array(all_conn_signs, dtype=np.int32),
            conn_enabled=np.array(all_conn_enabled, dtype=np.int32),
            entry_indices=np.array(node_entry_indices, dtype=np.int32),
            conn_entry_indices=np.array(conn_entry_indices, dtype=np.int32),
        )

    @staticmethod
    def load(filepath: str) -> "HallOfFame":
        """Load Hall of Fame from a .npz file."""
        if not os.path.exists(filepath):
            return HallOfFame()

        data = np.load(filepath, allow_pickle=False)
        fitnesses = data["fitnesses"]
        generations = data["generations"]
        node_ids = data["node_ids"]
        node_types = data["node_types"]
        node_acts = data["node_acts"]
        node_aggs = data["node_aggs"]
        conn_innovs = data.get("conn_innovs", np.array([], dtype=np.int32))
        conn_srcs = data.get("conn_srcs", np.array([], dtype=np.int32))
        conn_dsts = data.get("conn_dsts", np.array([], dtype=np.int32))
        conn_signs = data.get("conn_signs", np.array([], dtype=np.int32))
        conn_enabled = data.get("conn_enabled", np.array([], dtype=np.int32))
        node_entry_indices = data.get("entry_indices", np.array([], dtype=np.int32))
        conn_entry_indices = data.get("conn_entry_indices", np.array([], dtype=np.int32))

        hof = HallOfFame()
        n_entries = len(fitnesses)

        # Rebuild entries.
        node_offset = 0
        conn_offset = 0
        for ei in range(n_entries):
            g = Genome(node_genes=[], conn_genes=[])
            
            # Override with stored node genes.
            stored_nodes: dict[int, NodeGene] = {}
            while node_offset < len(node_entry_indices) and node_entry_indices[node_offset] == ei:
                nid = int(node_ids[node_offset])
                stored_nodes[nid] = NodeGene.make(
                    nid,
                    int(node_types[node_offset]),
                    int(node_acts[node_offset]),
                    int(node_aggs[node_offset]),
                )
                node_offset += 1
            g.node_genes = stored_nodes

            # Rebuild connections for this entry.
            stored_conns: dict[int, ConnGene] = {}
            while conn_offset < len(conn_entry_indices) and conn_entry_indices[conn_offset] == ei:
                inno = int(conn_innovs[conn_offset])
                stored_conns[inno] = ConnGene.make(
                    inno,
                    int(conn_srcs[conn_offset]),
                    int(conn_dsts[conn_offset]),
                    int(conn_signs[conn_offset]),
                    bool(conn_enabled[conn_offset]),
                )
                conn_offset += 1
            g.conn_genes = stored_conns
            
            if g.conn_genes:
                g.next_innovation = max(g.conn_genes.keys()) + 1
            else:
                g.next_innovation = 0

            hof.add(g, float(fitnesses[ei]), int(generations[ei]))

        return hof
