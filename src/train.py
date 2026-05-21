"""Main evolution loop for training WANNs to play Sueca.

Adaptive curriculum training:
  Phase 0: random opponents — easy warmup (advance when median_delta > 0.5)
  Phase 1: heuristic opponents — learn to beat rules (advance when median_delta > 0.0)
  Phase 2: mixed HoF + heuristic (advance when HoF has ≥ 5 entries with positive fitness)
  Phase 3: pure self-play with HoF

Delta-fitness: genome is compared against a baseline (RandomBot) on the same
deal to eliminate deal-luck variance (Common Random Numbers / AIVAT-style).

Weight sweep: evaluation uses W ∈ {0.5, 1.0, 2.0} for true weight-agnostic fitness.
"""

from __future__ import annotations

import csv
import os
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.baselines.heuristic_bot import HeuristicBot
from src.baselines.random_bot import RandomBot
from src.engine.duplicate_loop import Bot, generate_deals
from src.oracle.fitness import (
    DEFAULT_WEIGHT_SWEEP,
    evaluate_genome,
    evaluate_population,
    oracle_tax_penalty,
)
from src.oracle.hall_of_fame import HallOfFame
from src.wann.genome import Genome
from src.wann.population import PopConfig, Population


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Population
    pop_size: int = 100
    generations: int = 200
    elitism: int = 2

    # Evaluation
    n_deals: int = 16
    curriculum_gens: int = 60
    sweep_weights: list[float] = field(default_factory=lambda: list(DEFAULT_WEIGHT_SWEEP))

    # Species
    compatibility_threshold: float = 3.0
    stagnation_limit: int = 15
    c_excess: float = 1.0
    c_disjoint: float = 1.0
    c_mismatch: float = 0.4

    # Mutation probabilities
    p_add_node: float = 0.03
    p_add_conn: float = 0.15
    p_toggle_conn: float = 0.05
    p_flip_sign: float = 0.10
    p_change_act: float = 0.02
    p_change_agg: float = 0.02
    p_crossover: float = 0.25

    # Seed genomes
    seed_fraction: float = 0.10

    # Multi-objective ranking
    pareto_complexity_prob: float = 0.80

    # Adaptive curriculum thresholds
    phase0_threshold: float = 0.5   # median delta to advance from Phase 0 → 1
    phase1_threshold: float = 0.0   # median delta to advance from Phase 1 → 2
    phase2_hof_min: int = 5         # HoF entries with positive fitness to advance Phase 2 → 3
    min_gens_per_phase: int = 10    # minimum generations in each phase
    adaptive_window: int = 5        # number of recent gens to average for phase transitions

    # Hall of Fame
    hof_size: int = 20

    # Output
    checkpoint_dir: str = "checkpoints"
    stats_file: str = "training_stats.csv"
    seed: int = 42

    def to_pop_config(self) -> PopConfig:
        return PopConfig(
            pop_size=self.pop_size,
            elitism=self.elitism,
            compatibility_threshold=self.compatibility_threshold,
            stagnation_limit=self.stagnation_limit,
            c_excess=self.c_excess,
            c_disjoint=self.c_disjoint,
            c_mismatch=self.c_mismatch,
            p_add_node=self.p_add_node,
            p_add_conn=self.p_add_conn,
            p_toggle_conn=self.p_toggle_conn,
            p_flip_sign=self.p_flip_sign,
            p_change_act=self.p_change_act,
            p_change_agg=self.p_change_agg,
            p_crossover=self.p_crossover,
            seed_fraction=self.seed_fraction,
            pareto_complexity_prob=self.pareto_complexity_prob,
        )


def _determine_phase(
    gen: int,
    current_phase: int,
    recent_deltas: deque[float],
    hof: HallOfFame,
    config: TrainConfig,
    gens_in_current_phase: int,
) -> int:
    """Determine the curriculum phase based on population performance.

    Phase transitions are gated by performance thresholds, not fixed generation counts.
    """
    if gens_in_current_phase < config.min_gens_per_phase:
        return current_phase

    if len(recent_deltas) < config.adaptive_window:
        return current_phase

    avg_delta = sum(recent_deltas) / len(recent_deltas)

    if current_phase == 0:
        if avg_delta > config.phase0_threshold:
            return 1
    elif current_phase == 1:
        if avg_delta > config.phase1_threshold:
            return 2
    elif current_phase == 2:
        positive_hof = sum(1 for e in hof.entries if e.fitness > 0)
        if positive_hof >= config.phase2_hof_min:
            return 3

    return current_phase


def _get_opponent_bots(
    phase: int,
    hof: HallOfFame,
    rng: np.random.Generator,
    config: TrainConfig,
) -> list[Bot]:
    """Determine opponent bots based on curriculum phase.

    Returns [partner, opp1, opp2].

    Phase 0: random opponents — easy warmup
    Phase 1: heuristic opponents — learn to beat rules
    Phase 2: mixed HoF + heuristic — competitive
    Phase 3: self-play with HoF
    """
    partner: Bot
    opp1: Bot
    opp2: Bot

    if phase == 0:
        # Phase 0: random bots — easy warmup.
        partner = RandomBot()
        opp1 = RandomBot()
        opp2 = RandomBot()
    elif phase == 1 or len(hof) == 0:
        # Phase 1: all heuristic.
        partner = HeuristicBot()
        opp1 = HeuristicBot()
        opp2 = HeuristicBot()
    elif phase == 2:
        # Phase 2: mixed.
        if rng.random() < 0.5 and len(hof) > 0:
            sampled = hof.sample(rng, n=1)
            partner = _genome_to_bot(sampled[0], config.sweep_weights)
        else:
            partner = HeuristicBot()

        if rng.random() < 0.3 and len(hof) > 0:
            sampled = hof.sample(rng, n=1)
            opp1 = _genome_to_bot(sampled[0], config.sweep_weights)
        else:
            opp1 = HeuristicBot()

        opp2 = HeuristicBot()
    else:
        # Phase 3: self-play with HoF.
        if len(hof) > 0:
            sampled = hof.sample(rng, n=3)
            partner = _genome_to_bot(sampled[0], config.sweep_weights)
            if len(sampled) >= 3:
                opp1 = _genome_to_bot(sampled[1], config.sweep_weights)
                opp2 = _genome_to_bot(sampled[2], config.sweep_weights)
            elif len(sampled) >= 2:
                opp1 = _genome_to_bot(sampled[1], config.sweep_weights)
                opp2 = HeuristicBot()
            else:
                opp1 = HeuristicBot()
                opp2 = HeuristicBot()
        else:
            partner = HeuristicBot()
            opp1 = HeuristicBot()
            opp2 = HeuristicBot()

    return [partner, opp1, opp2]


def _genome_to_bot(genome: Genome, weights: list[float] | None = None) -> Bot:
    """Convert a genome to a playable bot."""
    from src.oracle.fitness import WannBotSweep
    return WannBotSweep(genome, weights=weights)


def train(config: TrainConfig | None = None) -> tuple[Population, HallOfFame]:
    """Run the full training loop.

    Returns:
        (final_population, hall_of_fame).
    """
    config = config or TrainConfig()
    rng = np.random.default_rng(config.seed)

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Initialize.
    pop_config = config.to_pop_config()
    pop = Population(pop_config, seed=config.seed)
    hof = HallOfFame(max_size=config.hof_size)
    baseline_bot = RandomBot()

    # Adaptive curriculum state.
    current_phase = 0
    gens_in_phase = 0
    recent_median_deltas: deque[float] = deque(maxlen=config.adaptive_window)

    # CSV stats.
    stats_fh = open(config.stats_file, "w", newline="")
    writer = csv.writer(stats_fh)
    writer.writerow([
        "generation",
        "phase",
        "best_fitness",
        "avg_fitness",
        "median_fitness",
        "best_delta",
        "median_delta",
        "global_best_fitness",
        "n_species",
        "n_connections_best",
        "n_hidden_best",
        "oracle_tax",
        "elapsed_sec",
    ])

    print(f"{'Gen':>4s} {'Ph':>2s} {'Best':>8s} {'Avg':>8s} {'Med':>8s} "
          f"{'Δbest':>8s} {'Δmed':>8s} "
          f"{'Species':>8s} {'Conns':>6s} {'Hidden':>6s} {'Tax':>6s} {'Time':>8s}")
    print("-" * 100)

    for gen in range(config.generations):
        t0 = time.time()

        # Generate deals (re-seeded per generation).
        deals = generate_deals(gen, n_deals=config.n_deals, base_seed=config.seed * 1000)

        # Get opponent bots for this phase.
        opp_bots = _get_opponent_bots(current_phase, hof, rng, config)

        # Evaluate all genomes with delta-fitness scoring.
        fitnesses, deltas, total_illegal = evaluate_population(
            pop.genomes,
            deals,
            opp_bots,
            baseline_bot,
            generation=gen,
            curriculum_gens=config.curriculum_gens,
            base_seed=config.seed + gen * 1000,
            sweep_weights=config.sweep_weights,
            parallel=True,
        )

        pop.tell_fitnesses(fitnesses)

        # Track best.
        best_idx = int(np.argmax(fitnesses))
        best_fit = fitnesses[best_idx]
        avg_fit = float(np.mean(fitnesses))
        median_fit = float(np.median(fitnesses))
        best_delta = deltas[best_idx]
        median_delta = float(np.median(deltas))

        # Add to Hall of Fame.
        hof.add(pop.genomes[best_idx], best_fit, gen)

        # Adaptive curriculum.
        recent_median_deltas.append(median_delta)
        gens_in_phase += 1
        new_phase = _determine_phase(
            gen, current_phase, recent_median_deltas, hof, config, gens_in_phase
        )
        if new_phase != current_phase:
            print(f"  >>> Phase transition: {current_phase} → {new_phase} at gen {gen}")
            current_phase = new_phase
            gens_in_phase = 0
            recent_median_deltas.clear()

        # Stats.
        elapsed = time.time() - t0
        n_species = len([s for s in pop.species_list if s.members])
        best_genome = pop.genomes[best_idx]
        n_conns = best_genome.num_enabled()
        n_hidden = len(best_genome.hidden_ids)
        tax = oracle_tax_penalty(gen, config.curriculum_gens)

        writer.writerow([
            gen, current_phase, best_fit, avg_fit, median_fit,
            best_delta, median_delta,
            pop.global_best_fitness, n_species, n_conns, n_hidden,
            tax, elapsed,
        ])
        stats_fh.flush()

        print(
            f"{gen:4d} {current_phase:2d} {best_fit:8.4f} {avg_fit:8.4f} {median_fit:8.4f} "
            f"{best_delta:8.4f} {median_delta:8.4f} "
            f"{n_species:8d} {n_conns:6d} {n_hidden:6d} {tax:6.1f} {elapsed:7.1f}s"
        )

        # Speciate and breed next generation.
        _ = pop.speciate_and_evolve()

        # Checkpoint.
        if (gen + 1) % 10 == 0 or gen == config.generations - 1:
            hof_path = os.path.join(config.checkpoint_dir, f"hof_gen{gen + 1:04d}.npz")
            hof.save(hof_path)

            if pop.global_best_genome is not None:
                best_path = os.path.join(
                    config.checkpoint_dir, f"best_genome_gen{gen + 1:04d}.npz"
                )
                # Save best genome.
                _save_genome(pop.global_best_genome, best_path)

    stats_fh.close()

    # Final save.
    hof.save(os.path.join(config.checkpoint_dir, "hof_final.npz"))
    if pop.global_best_genome is not None:
        _save_genome(
            pop.global_best_genome,
            os.path.join(config.checkpoint_dir, "best_genome_final.npz"),
        )

    print(f"\nTraining complete. Best fitness: {pop.global_best_fitness:.4f}")
    return pop, hof


def _save_genome(genome: Genome, filepath: str) -> None:
    """Save a single genome to .npz."""
    node_ids = []
    node_types = []
    node_acts = []
    node_aggs = []
    for nid, ng in genome.node_genes.items():
        node_ids.append(ng.id)
        node_types.append(ng.node_type)
        node_acts.append(ng.activation_fn)
        node_aggs.append(ng.aggregation_fn)

    conn_innovs = []
    conn_srcs = []
    conn_dsts = []
    conn_signs = []
    conn_enabled = []
    for c in genome.conn_genes.values():
        conn_innovs.append(c.innovation)
        conn_srcs.append(c.src)
        conn_dsts.append(c.dst)
        conn_signs.append(c.sign)
        conn_enabled.append(1 if c.enabled else 0)

    np.savez(
        filepath,
        next_innovation=genome.next_innovation,
        node_ids=np.array(node_ids, dtype=np.int32),
        node_types=np.array(node_types, dtype=np.int32),
        node_acts=np.array(node_acts, dtype=np.int32),
        node_aggs=np.array(node_aggs, dtype=np.int32),
        conn_innovs=np.array(conn_innovs, dtype=np.int32),
        conn_srcs=np.array(conn_srcs, dtype=np.int32),
        conn_dsts=np.array(conn_dsts, dtype=np.int32),
        conn_signs=np.array(conn_signs, dtype=np.int32),
        conn_enabled=np.array(conn_enabled, dtype=np.int32),
    )


def load_genome(filepath: str) -> Genome:
    """Load a genome from .npz."""
    data = np.load(filepath, allow_pickle=False)
    from src.wann.genome import NodeGene, ConnGene

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

    conn_innovs = data.get("conn_innovs", np.array([], dtype=np.int32))
    conn_srcs = data.get("conn_srcs", np.array([], dtype=np.int32))
    conn_dsts = data.get("conn_dsts", np.array([], dtype=np.int32))
    conn_signs = data.get("conn_signs", np.array([], dtype=np.int32))
    conn_enabled = data.get("conn_enabled", np.array([], dtype=np.int32))

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

    next_innovation = int(data.get("next_innovation", 0))
    return Genome(node_genes=node_genes, conn_genes=conn_genes, next_innovation=next_innovation)


if __name__ == "__main__":
    config = TrainConfig(
        pop_size=100,
        generations=200,
        n_deals=16,
        curriculum_gens=60,
    )
    train(config)
