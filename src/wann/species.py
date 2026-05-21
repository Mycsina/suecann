"""WANN speciation — compatibility distance and species management."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.wann.genome import ConnGene, Genome, NodeGene, NodeType


def compatibility_distance(
    genome_a: Genome,
    genome_b: Genome,
    c1: float = 1.0,
    c2: float = 1.0,
    c3: float = 0.4,
) -> float:
    """Compute compatibility distance between two genomes.

    Based on NEAT metric:
      delta = c1*E/N + c2*D/N + c3*W

    Where E = excess genes, D = disjoint genes, W = average sign difference
    for matching connection genes.

    For node gene differences we add a small bonus: +0.5 per differing
    activation/aggregation in matching hidden nodes.

    Args:
        genome_a: First genome.
        genome_b: Second genome.
        c1: Coefficient for excess genes.
        c2: Coefficient for disjoint genes.
        c3: Coefficient for sign/function differences.

    Returns:
        Compatibility distance (non-negative float).
    """
    conn_innov_a = set(genome_a.conn_genes.keys())
    conn_innov_b = set(genome_b.conn_genes.keys())

    if not conn_innov_a and not conn_innov_b:
        return 0.0

    all_innov = conn_innov_a | conn_innov_b
    max_innov_a = max(conn_innov_a) if conn_innov_a else 0
    max_innov_b = max(conn_innov_b) if conn_innov_b else 0

    # Shared, disjoint, excess.
    shared = conn_innov_a & conn_innov_b
    disjoint_a = {i for i in conn_innov_a if i not in shared and i < max_innov_b}
    disjoint_b = {i for i in conn_innov_b if i not in shared and i < max_innov_a}
    excess_a = {i for i in conn_innov_a if i > max_innov_b}
    excess_b = {i for i in conn_innov_b if i > max_innov_a}

    excess = len(excess_a) + len(excess_b)
    disjoint = len(disjoint_a) + len(disjoint_b)

    # Normalization factor N (number of genes in larger genome, min 1).
    n_conns = max(len(conn_innov_a), len(conn_innov_b), 1)

    # Sign difference for matching connections.
    sign_diff_total = 0.0
    for i in shared:
        ca = genome_a.conn_genes[i]
        cb = genome_b.conn_genes[i]
        if ca.sign != cb.sign:
            sign_diff_total += 1.0
        if ca.enabled != cb.enabled:
            sign_diff_total += 0.5

    avg_sign_diff = sign_diff_total / len(shared) if shared else 0.0

    # Node gene differences for matching hidden nodes.
    hidden_a = {
        nid: ng
        for nid, ng in genome_a.node_genes.items()
        if ng.node_type == NodeType.HIDDEN
    }
    hidden_b = {
        nid: ng
        for nid, ng in genome_b.node_genes.items()
        if ng.node_type == NodeType.HIDDEN
    }
    shared_hidden = set(hidden_a.keys()) & set(hidden_b.keys())
    node_diff = 0.0
    for nid in shared_hidden:
        na = hidden_a[nid]
        nb = hidden_b[nid]
        if na.activation_fn != nb.activation_fn:
            node_diff += 0.5
        if na.aggregation_fn != nb.aggregation_fn:
            node_diff += 0.5

    dist = (
        c1 * excess / n_conns
        + c2 * disjoint / n_conns
        + c3 * avg_sign_diff
        + c3 * node_diff
    )

    return dist


@dataclass
class Species:
    """A species within the population."""

    id: int
    representative: Genome
    members: list[int] = field(default_factory=list)  # indices into population
    best_fitness: float = float("-inf")
    generations_no_improvement: int = 0
    created_at_gen: int = 0

    def update_best(self, fitness: float) -> bool:
        """Update best fitness. Returns True if improved."""
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.generations_no_improvement = 0
            return True
        return False

    def increment_stagnation(self) -> None:
        self.generations_no_improvement += 1


def speciate(
    genomes: list[Genome],
    fitnesses: list[float],
    species_list: list[Species],
    threshold: float,
    next_species_id: int,
    rng: np.random.Generator,
    c1: float = 1.0,
    c2: float = 1.0,
    c3: float = 0.4,
) -> tuple[list[Species], int]:
    """Assign genomes to species using compatibility distance.

    Args:
        genomes: All genomes in the population.
        fitnesses: Fitness of each genome (may be partial placeholder).
        species_list: Existing species (may be empty on first call).
        threshold: Compatibility distance threshold.
        next_species_id: Next available species ID.
        rng: Random generator.
        c1: Coefficient for excess genes.
        c2: Coefficient for disjoint genes.
        c3: Coefficient for sign/function differences.

    Returns:
        (updated_species_list, next_species_id).
    """
    # Reset all species members.
    for sp in species_list:
        sp.members = []

    unassigned = list(range(len(genomes)))

    for sp in species_list:
        if not unassigned:
            break
        rep = sp.representative
        new_members = []
        still_unassigned = []
        for idx in unassigned:
            if compatibility_distance(rep, genomes[idx], c1, c2, c3) < threshold:
                new_members.append(idx)
            else:
                still_unassigned.append(idx)
        unassigned = still_unassigned
        if new_members:
            sp.members = new_members

    # Create new species for remaining genomes.
    while unassigned:
        idx = unassigned[0]
        # Pick a random unassigned genome as new representative.
        new_sp = Species(
            id=next_species_id,
            representative=genomes[idx].copy(),
            members=[idx],
        )
        next_species_id += 1
        species_list.append(new_sp)
        unassigned = unassigned[1:]

        # Try to absorb similar genomes into this new species.
        rep = new_sp.representative
        new_members = []
        still_unassigned = []
        for other_idx in unassigned:
            if compatibility_distance(rep, genomes[other_idx], c1, c2, c3) < threshold:
                new_members.append(other_idx)
            else:
                still_unassigned.append(other_idx)
        unassigned = still_unassigned
        new_sp.members.extend(new_members)

    # Remove empty species (but keep if they had members and just went extinct).
    # Actually keep all species for stagnation tracking.

    return species_list, next_species_id
