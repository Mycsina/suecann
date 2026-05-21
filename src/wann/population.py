"""WANN population — NEAT-style evolution with speciation.

Includes:
- Seed genomes encoding known Sueca heuristics (10% of population)
- Rank-based fitness for noise-robust selection
- Multi-objective Pareto ranking (performance + complexity)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.wann.genome import (
    BIAS_ID,
    INPUT_START,
    OUTPUT_START,
    ConnGene,
    Genome,
    NodeGene,
    NodeType,
)
from src.wann.mutations import MUTATIONS
from src.wann.species import Species, compatibility_distance, speciate


@dataclass
class PopConfig:
    """Configuration for the NEAT-style population."""

    pop_size: int = 100
    elitism: int = 2  # top N per species kept unchanged
    survival_threshold: float = 0.2  # fraction of species to keep for breeding
    compatibility_threshold: float = 2.0
    stagnation_limit: int = 15  # gens without improvement before species removal
    min_species_size: int = 3
    # Speciation compatibility distance coefficients
    c_excess: float = 1.0
    c_disjoint: float = 1.0
    c_mismatch: float = 0.4
    # Mutation probabilities per genome (applied independently).
    p_add_node: float = 0.03
    p_add_conn: float = 0.15
    p_toggle_conn: float = 0.05
    p_flip_sign: float = 0.10
    p_change_act: float = 0.02
    p_change_agg: float = 0.02
    # Crossover
    p_crossover: float = 0.25  # prob of crossover vs asexual
    # Seed genomes
    seed_fraction: float = 0.10  # fraction of pop seeded with heuristic strategies
    # Multi-objective ranking
    pareto_complexity_prob: float = (
        0.80  # prob of ranking by (perf, complexity) vs (perf, max_perf)
    )


# --- Seed genome strategies ---
# Each strategy is a list of (src_node_id, dst_output_id, sign) tuples.
# These encode known-good Sueca heuristics as WANN connections.
# Belief state indices: see CLAUDE.md
#   5 = Am_I_Leading, 7 = Is_Partner_Winning
# Output indices: 19=DUCK_OR_DUMP, 20=TAKE_CHEAPLY, 21=FORCE_HIGH, 22=FEED_PARTNER, 23=CUT_LOW

SEED_STRATEGIES: list[tuple[str, list[tuple[int, int, int]]]] = [
    # Aggressive: always play strongest card (bias → FORCE_HIGH)
    ("aggressive", [(BIAS_ID, OUTPUT_START + 2, +1)]),
    # Take Cheaply: always try to win cheaply (bias → TAKE_CHEAPLY)
    ("take_cheaply", [(BIAS_ID, OUTPUT_START + 1, +1)]),
    # Partner Aware: duck when partner wins, else force high
    (
        "partner_aware",
        [
            (
                INPUT_START + 7,
                OUTPUT_START + 0,
                +1,
            ),  # Is_Partner_Winning(+1) → DUCK_OR_DUMP
            (
                INPUT_START + 7,
                OUTPUT_START + 2,
                -1,
            ),  # Is_Partner_Winning(-1) → FORCE_HIGH
        ],
    ),
    # Trump Cutter: cut when void in led suit and have trump
    (
        "trump_cutter",
        [
            (INPUT_START + 0, OUTPUT_START + 4, -1),  # Has_Led_Suit(-1) → CUT_LOW
            (INPUT_START + 1, OUTPUT_START + 4, +1),  # Has_Trump(+1) → CUT_LOW
        ],
    ),
    # Feeder: feed points to winning partner
    (
        "feeder",
        [
            (
                INPUT_START + 7,
                OUTPUT_START + 3,
                +1,
            ),  # Is_Partner_Winning(+1) → FEED_PARTNER
        ],
    ),
    # Position-aware attacker: force high when leading
    (
        "lead_attacker",
        [
            (INPUT_START + 5, OUTPUT_START + 2, +1),  # Am_I_Leading(+1) → FORCE_HIGH
        ],
    ),
    # Last-to-play optimizer: take cheaply when last to play
    (
        "last_taker",
        [
            (
                INPUT_START + 6,
                OUTPUT_START + 1,
                +1,
            ),  # Am_I_Last_To_Play(+1) → TAKE_CHEAPLY
        ],
    ),
    # Combined: partner aware + position aware
    (
        "combined_basic",
        [
            (
                INPUT_START + 7,
                OUTPUT_START + 0,
                +1,
            ),  # Is_Partner_Winning(+1) → DUCK_OR_DUMP
            (
                INPUT_START + 7,
                OUTPUT_START + 2,
                -1,
            ),  # Is_Partner_Winning(-1) → FORCE_HIGH
            (INPUT_START + 5, OUTPUT_START + 2, +1),  # Am_I_Leading(+1) → FORCE_HIGH
        ],
    ),
]


def _create_seed_genome(strategy: list[tuple[int, int, int]]) -> Genome:
    """Create a genome from a seed strategy specification."""
    g = Genome.initial()
    for src, dst, sign in strategy:
        inno = g.next_innovation
        g.add_connection(ConnGene.make(inno, src, dst, sign=sign))
    return g


def _rank_values(values: list[float]) -> list[float]:
    """Convert raw values to normalized ranks in [0, 1].

    Higher value → higher rank. Ties get average rank.
    """
    n = len(values)
    if n == 0:
        return []

    # Argsort ascending, then assign ranks.
    indices = np.argsort(values)
    ranks = np.zeros(n, dtype=np.float64)
    for rank_pos, idx in enumerate(indices):
        ranks[idx] = rank_pos

    # Handle ties: average the ranks of tied values.
    sorted_vals = np.array(values)
    unique_vals = np.unique(sorted_vals)
    for val in unique_vals:
        mask = sorted_vals == val
        if np.sum(mask) > 1:
            mean_rank = np.mean(ranks[mask])
            ranks[mask] = mean_rank

    # Normalize to [0, 1].
    if n > 1:
        ranks = ranks / (n - 1)

    return ranks.tolist()


def _pareto_rank(
    fitnesses: list[float],
    complexities: list[int],
) -> list[float]:
    """Non-dominated Pareto ranking on (fitness ↑, simplicity ↑) with lexicographic tie-breaking.

    Returns a rank score for each genome (higher = better).
    Genomes on the Pareto front get the highest base rank.
    Within the same Pareto level, tie-breaking is done using normalized raw fitness.
    """
    n = len(fitnesses)
    if n == 0:
        return []

    # Convert complexity to simplicity (lower complexity = higher simplicity).
    max_complexity = max(complexities) if complexities else 1
    simplicities = [max_complexity - c for c in complexities]

    # Compute domination count and dominated-by sets.
    domination_count = [0] * n
    dominated_by: list[list[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            # i dominates j if better on both objectives.
            i_dom_j = (
                fitnesses[i] >= fitnesses[j]
                and simplicities[i] >= simplicities[j]
                and (fitnesses[i] > fitnesses[j] or simplicities[i] > simplicities[j])
            )
            j_dom_i = (
                fitnesses[j] >= fitnesses[i]
                and simplicities[j] >= simplicities[i]
                and (fitnesses[j] > fitnesses[i] or simplicities[j] > simplicities[i])
            )

            if i_dom_j:
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif j_dom_i:
                dominated_by[j].append(i)
                domination_count[i] += 1

    # Assign Pareto front levels (lower level = better).
    levels = [0] * n
    current_front = [i for i in range(n) if domination_count[i] == 0]
    level = 0

    while current_front:
        for i in current_front:
            levels[i] = level
        next_front = []
        for i in current_front:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front = next_front
        level += 1

    # Min-Max normalize fitnesses to [0, 1] for fair tie-breaking
    min_fit = min(fitnesses)
    max_fit = max(fitnesses)
    fit_range = max_fit - min_fit

    perf_scores = [
        (f - min_fit) / fit_range if fit_range > 0 else 1.0
        for f in fitnesses
    ]

    # Convert levels to scores (lower level = higher score).
    max_level = max(levels) if levels else 0
    scores = []

    for lvl, perf in zip(levels, perf_scores):
        base_score = max_level - lvl
        # Add 0.5 * normalized fitness to break ties
        scores.append(base_score + 0.5 * perf)

    # Normalize final scores to [0, 1]
    max_score = max(scores) if scores else 1.0
    min_score = min(scores) if scores else 0.0

    if max_score > min_score:
        scores = [(s - min_score) / (max_score - min_score) for s in scores]
    else:
        scores = [1.0] * n

    return scores


class Population:
    """A population of WANN genomes with speciation."""

    def __init__(
        self,
        config: PopConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.config = config or PopConfig()
        self.rng = np.random.default_rng(seed)
        self.genomes: list[Genome] = []
        self.fitnesses: list[float] = []
        self.species_list: list[Species] = []
        self.generation: int = 0
        self.next_species_id: int = 0
        self.global_best_fitness: float = float("-inf")
        self.global_best_genome: Genome | None = None

        self._init_population()

    def _init_population(self) -> None:
        """Create initial population with seed genomes and random-link genomes.

        10% of the population is seeded with known-good Sueca heuristics.
        The rest get random connections + mutations for diversity.
        """
        cfg = self.config
        n_seeds = max(1, int(cfg.pop_size * cfg.seed_fraction))

        # --- Seed genomes (10% of population) ---
        for i in range(n_seeds):
            strategy_idx = i % len(SEED_STRATEGIES)
            _, strategy_conns = SEED_STRATEGIES[strategy_idx]
            g = _create_seed_genome(strategy_conns)
            # Add 1-2 random mutations on top for diversity.
            n_extra = self.rng.integers(0, 2)
            for _ in range(n_extra):
                self._apply_mutations(g)
            self.genomes.append(g)
            self.fitnesses.append(0.0)

        # --- Random-link genomes (remaining 90%) ---
        base = Genome.initial()
        for i in range(cfg.pop_size - n_seeds):
            g = base.copy()
            # Every genome gets at least one connection.
            self._apply_single_mutation(g, force_add_conn=True)
            # Top 30% of the random genomes get extra mutations.
            if i < (cfg.pop_size - n_seeds) * 0.3:
                n_mutations = self.rng.integers(1, 3)
                for _ in range(n_mutations):
                    self._apply_mutations(g)
            self.genomes.append(g)
            self.fitnesses.append(0.0)

    def _apply_single_mutation(
        self, genome: Genome, force_add_conn: bool = False
    ) -> int:
        """Apply a single mutation. With force_add_conn, only adds a connection."""
        if force_add_conn:
            MUTATIONS[1](genome, self.rng)  # add_connection
            return 1
        return self._apply_mutations(genome)

    def _apply_mutations(self, genome: Genome) -> int:
        """Apply mutations probabilistically. Returns number of mutations applied."""
        n = 0
        cfg = self.config
        if self.rng.random() < cfg.p_add_node:
            MUTATIONS[0](genome, self.rng)
            n += 1
        if self.rng.random() < cfg.p_add_conn:
            MUTATIONS[1](genome, self.rng)
            n += 1
        if self.rng.random() < cfg.p_toggle_conn:
            MUTATIONS[2](genome, self.rng)
            n += 1
        if self.rng.random() < cfg.p_flip_sign:
            MUTATIONS[3](genome, self.rng)
            n += 1
        if self.rng.random() < cfg.p_change_act:
            MUTATIONS[4](genome, self.rng)
            n += 1
        if self.rng.random() < cfg.p_change_agg:
            MUTATIONS[5](genome, self.rng)
            n += 1
        return n

    def speciate_and_evolve(self) -> list[int]:
        """Speciate current (evaluated) population and breed the next generation.

        Call this AFTER setting fitnesses for the current population.

        Returns:
            List of genome indices in the BRED next generation to evaluate.
        """
        # Speciate current population based on fitnesses.
        self._update_species()
        self._update_stagnation()

        # Breed the next generation.
        new_genomes = self._breed_next_generation()
        self.genomes = new_genomes
        self.fitnesses = [0.0] * len(self.genomes)
        self.generation += 1

        return list(range(len(self.genomes)))

    def tell_fitnesses(self, fitnesses: list[float]) -> None:
        """Report fitness values for the current population."""
        assert len(fitnesses) == len(self.genomes)
        self.fitnesses = list(fitnesses)

        # Track global best.
        best_idx = int(np.argmax(fitnesses))
        if fitnesses[best_idx] > self.global_best_fitness:
            self.global_best_fitness = fitnesses[best_idx]
            self.global_best_genome = self.genomes[best_idx].copy()

    def _update_species(self) -> None:
        """Run speciation on current population."""
        self.species_list, self.next_species_id = speciate(
            self.genomes,
            self.fitnesses,
            self.species_list,
            self.config.compatibility_threshold,
            self.next_species_id,
            self.rng,
            c1=getattr(self.config, "c_excess", 1.0),
            c2=getattr(self.config, "c_disjoint", 1.0),
            c3=getattr(self.config, "c_mismatch", 0.4),
        )

    def _update_stagnation(self) -> None:
        """Update stagnation counters and remove stale species."""
        for sp in self.species_list:
            if not sp.members:
                sp.increment_stagnation()
                continue
            # Get best fitness in this species.
            best_f = max(self.fitnesses[i] for i in sp.members)
            improved = sp.update_best(best_f)
            if not improved:
                sp.increment_stagnation()

        # Remove species that have stagnated too long (unless it's the last one).
        active = [sp for sp in self.species_list if sp.members]
        if len(active) > 1:
            surviving = [
                sp
                for sp in active
                if sp.generations_no_improvement < self.config.stagnation_limit
            ]
            if surviving:
                # Remove stale ones.
                stale_ids = {sp.id for sp in active if sp not in surviving}
                self.species_list = [
                    sp for sp in self.species_list if sp.id not in stale_ids
                ]

    def _breed_next_generation(self) -> list[Genome]:
        """Create the next generation via multi-objective ranking and rank-based selection."""
        cfg = self.config
        new_genomes: list[Genome] = []

        # Sort species by best fitness.
        active = [sp for sp in self.species_list if sp.members]
        if not active:
            # Fallback: keep current population.
            return [g.copy() for g in self.genomes]

        active.sort(
            key=lambda sp: max(self.fitnesses[i] for i in sp.members), reverse=True
        )

        # --- Multi-objective ranking (Pareto or fitness-only) ---
        complexities = [g.num_enabled() for g in self.genomes]
        if self.rng.random() < cfg.pareto_complexity_prob:
            # 80%: rank by (performance, simplicity) Pareto front.
            pareto_scores = _pareto_rank(self.fitnesses, complexities)
            selection_fitness = pareto_scores
        else:
            # 20%: rank by performance only (allows complex but high-performing networks).
            selection_fitness = _rank_values(self.fitnesses)

        # Compute number of offspring per species (fitness-proportional).
        total_fitness = 0.0
        species_best = []
        for sp in active:
            best_f = max(selection_fitness[i] for i in sp.members)
            adj_f = (
                max(best_f, 0.0) + 0.1
            )  # small offset so all species get at least something
            total_fitness += adj_f * len(sp.members)
            species_best.append(adj_f)

        # Allocate offspring.
        offspring_counts: dict[int, int] = {}
        remaining = cfg.pop_size
        for i, sp in enumerate(active):
            if sp is active[-1]:
                count = remaining
            else:
                adj_f = species_best[i]
                count = max(
                    cfg.min_species_size,
                    min(
                        remaining - cfg.min_species_size,
                        int(cfg.pop_size * (adj_f * len(sp.members)) / total_fitness),
                    ),
                )
                count = min(count, remaining - cfg.min_species_size)
            offspring_counts[sp.id] = min(count, remaining)
            remaining -= offspring_counts[sp.id]
            if remaining <= 0:
                break

        # Ensure we hit pop_size.
        if remaining > 0:
            offspring_counts[active[0].id] += remaining

        # Breed within each species using rank-based selection.
        for sp in active:
            count = offspring_counts.get(sp.id, 0)
            if count <= 0:
                continue

            members = sp.members
            # Use rank-based fitness for selection within species.
            member_raw_fitnesses = [selection_fitness[i] for i in members]
            member_ranks = _rank_values(member_raw_fitnesses)

            # Sort members by rank (descending).
            ranked = sorted(
                zip(members, member_ranks), key=lambda x: x[1], reverse=True
            )
            ranked_indices = [r[0] for r in ranked]
            ranked_rank_values = [r[1] for r in ranked]

            # Elitism.
            elite_count = min(cfg.elitism, count, len(ranked_indices))
            for i in range(elite_count):
                new_genomes.append(self.genomes[ranked_indices[i]].copy())

            # Fill remaining with offspring.
            for _ in range(count - elite_count):
                if self.rng.random() < cfg.p_crossover and len(ranked_indices) >= 2:
                    # Tournament select two parents using rank-based fitness.
                    p1 = self._tournament_select(ranked_indices, ranked_rank_values)
                    p2 = self._tournament_select(ranked_indices, ranked_rank_values)
                    child = self._crossover(self.genomes[p1], self.genomes[p2])
                else:
                    parent_idx = self._tournament_select(
                        ranked_indices, ranked_rank_values
                    )
                    child = self.genomes[parent_idx].copy()

                self._apply_mutations(child)
                new_genomes.append(child)

        # Truncate/expand to exact pop_size.
        if len(new_genomes) > cfg.pop_size:
            new_genomes = new_genomes[: cfg.pop_size]
        elif len(new_genomes) < cfg.pop_size:
            # Fill with copies of best genome.
            while len(new_genomes) < cfg.pop_size:
                new_genomes.append(new_genomes[0].copy())

        return new_genomes

    def _tournament_select(
        self,
        ranked_indices: list[int],
        rank_values: list[float],
        tournament_size: int = 3,
    ) -> int:
        """Tournament selection using rank-based fitness values.

        Picks best of K random candidates by their rank value.
        """
        k = min(tournament_size, len(ranked_indices))
        positions = list(self.rng.choice(len(ranked_indices), size=k, replace=False))
        best_pos = max(positions, key=lambda p: rank_values[p])
        return ranked_indices[best_pos]

    def _crossover(self, parent_a: Genome, parent_b: Genome) -> Genome:
        """Crossover two genomes: child inherits genes from both parents.

        For matching innovation numbers, randomly pick from either parent.
        Disjoint/excess genes come from the fitter parent.
        """
        # Determine fitter parent (by comparing fitnesses if available).
        a_idx = None
        b_idx = None
        for i, g in enumerate(self.genomes):
            if g is parent_a:
                a_idx = i
            if g is parent_b:
                b_idx = i

        fa = self.fitnesses[a_idx] if a_idx is not None else 0.0
        fb = self.fitnesses[b_idx] if b_idx is not None else 0.0

        if fb > fa:
            parent_a, parent_b = parent_b, parent_a  # a is now fitter

        child_node_genes = []
        child_conn_genes = []

        # Node genes: union, with matching IDs from random parent.
        all_node_ids = set(parent_a.node_genes.keys()) | set(parent_b.node_genes.keys())
        for nid in all_node_ids:
            ng_a = parent_a.node_genes.get(nid)
            ng_b = parent_b.node_genes.get(nid)
            if ng_a is not None and ng_b is not None:
                chosen = ng_a if self.rng.random() < 0.5 else ng_b
            elif ng_a is not None:
                chosen = ng_a
            else:
                chosen = ng_b
            child_node_genes.append(chosen)

        # Connection genes.
        innov_a = set(parent_a.conn_genes.keys())
        innov_b = set(parent_b.conn_genes.keys())
        shared = innov_a & innov_b
        max_innov_a = max(innov_a) if innov_a else 0
        max_innov_b = max(innov_b) if innov_b else 0

        for i in shared:
            chosen = (
                parent_a.conn_genes[i]
                if self.rng.random() < 0.5
                else parent_b.conn_genes[i]
            )
            child_conn_genes.append(chosen)

        # Disjoint from fitter parent.
        for i in innov_a - innov_b:
            child_conn_genes.append(parent_a.conn_genes[i])

        child = Genome(
            node_genes=child_node_genes,
            conn_genes=child_conn_genes,
            next_innovation=max(parent_a.next_innovation, parent_b.next_innovation),
        )

        return child
