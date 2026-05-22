use crate::config::Config;
use crate::genome::{ConnGene, Genome, BIAS_ID, INPUT_START, OUTPUT_START};
use crate::mutations::{apply_mutations, mutate_add_connection, InnovationRegistry};
use crate::species::{speciate, Species};
use rand::Rng;
use std::collections::{HashMap, HashSet};

// Seed strategies matching Python SEED_STRATEGIES
// Each strategy is a tuple (name, list of (src, dst, sign))
const SEED_STRATEGIES: &[(&str, &[(usize, usize, i8)])] = &[
    ("aggressive", &[(BIAS_ID, OUTPUT_START + 2, 1)]),
    ("take_cheaply", &[(BIAS_ID, OUTPUT_START + 1, 1)]),
    (
        "partner_aware",
        &[
            (INPUT_START + 7, OUTPUT_START + 0, 1),
            (INPUT_START + 7, OUTPUT_START + 2, -1),
        ],
    ),
    (
        "trump_cutter",
        &[
            (INPUT_START + 0, OUTPUT_START + 4, -1),
            (INPUT_START + 1, OUTPUT_START + 4, 1),
        ],
    ),
    ("feeder", &[(INPUT_START + 7, OUTPUT_START + 3, 1)]),
    ("lead_attacker", &[(INPUT_START + 5, OUTPUT_START + 2, 1)]),
    ("last_taker", &[(INPUT_START + 6, OUTPUT_START + 1, 1)]),
    (
        "combined_basic",
        &[
            (INPUT_START + 7, OUTPUT_START + 0, 1),
            (INPUT_START + 7, OUTPUT_START + 2, -1),
            (INPUT_START + 5, OUTPUT_START + 2, 1),
        ],
    ),
    (
        "late_trump_aggressor",
        &[
            (INPUT_START + 19, OUTPUT_START + 4, -1),
            (INPUT_START + 1, OUTPUT_START + 4, 1),
            (INPUT_START + 19, OUTPUT_START + 2, -1),
        ],
    ),
    (
        "score_aware",
        &[
            (INPUT_START + 20, OUTPUT_START + 0, 1),
            (INPUT_START + 20, OUTPUT_START + 2, -1),
        ],
    ),
    (
        "trick_point_taker",
        &[
            (INPUT_START + 8, OUTPUT_START + 1, 1),
            (INPUT_START + 7, OUTPUT_START + 1, -1),
        ],
    ),
    (
        "void_exploiter",
        &[
            (INPUT_START + 12, OUTPUT_START + 2, 1),
            (INPUT_START + 5, OUTPUT_START + 2, 1),
        ],
    ),
    (
        "full_strategic",
        &[
            (INPUT_START + 7, OUTPUT_START + 0, 1),
            (INPUT_START + 7, OUTPUT_START + 2, -1),
            (INPUT_START + 5, OUTPUT_START + 2, 1),
            (INPUT_START + 6, OUTPUT_START + 1, 1),
            (INPUT_START + 20, OUTPUT_START + 2, -1),
            (INPUT_START + 1, OUTPUT_START + 4, 1),
            (INPUT_START + 0, OUTPUT_START + 4, -1),
        ],
    ),
];

fn create_seed_genome(strategy: &[(usize, usize, i8)]) -> Genome {
    let mut g = Genome::initial();
    for &(src, dst, sign) in strategy {
        let inno = g.next_innovation;
        g.add_connection(ConnGene::make(inno, src, dst, sign, true));
    }
    g
}

pub fn rank_values(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }

    // Argsort ascending
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ranks = vec![0.0; n];
    for (rank_pos, &idx) in indices.iter().enumerate() {
        ranks[idx] = rank_pos as f64;
    }

    // Handle ties: average the ranks of tied values.
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (values[indices[i]] - values[indices[j]]).abs() < 1e-9 {
            j += 1;
        }
        if j - i > 1 {
            let sum_ranks: f64 = (i..j).map(|r| r as f64).sum();
            let mean_rank = sum_ranks / (j - i) as f64;
            for k in i..j {
                ranks[indices[k]] = mean_rank;
            }
        }
        i = j;
    }

    // Normalize to [0, 1]
    if n > 1 {
        for r in ranks.iter_mut() {
            *r /= (n - 1) as f64;
        }
    } else {
        ranks[0] = 1.0;
    }

    ranks
}

pub fn pareto_rank(fitnesses: &[f64], complexities: &[usize]) -> Vec<f64> {
    let n = fitnesses.len();
    if n == 0 {
        return Vec::new();
    }

    let max_complexity = complexities.iter().max().copied().unwrap_or(1);
    let simplicities: Vec<f64> = complexities
        .iter()
        .map(|&c| (max_complexity - c) as f64)
        .collect();

    let mut domination_count = vec![0; n];
    let mut dominated_by = vec![Vec::new(); n];

    for i in 0..n {
        let f_i = fitnesses[i];
        let s_i = simplicities[i];
        for j in (i + 1)..n {
            let f_j = fitnesses[j];
            let s_j = simplicities[j];

            let i_dom_j = f_i >= f_j && s_i >= s_j && (f_i > f_j || s_i > s_j);
            let j_dom_i = f_j >= f_i && s_j >= s_i && (f_j > f_i || s_j > s_i);

            if i_dom_j {
                dominated_by[i].push(j);
                domination_count[j] += 1;
            } else if j_dom_i {
                dominated_by[j].push(i);
                domination_count[i] += 1;
            }
        }
    }

    let mut levels = vec![0; n];
    let mut current_front = Vec::new();
    for i in 0..n {
        if domination_count[i] == 0 {
            current_front.push(i);
        }
    }
    let mut level = 0;

    while !current_front.is_empty() {
        for &i in &current_front {
            levels[i] = level;
        }
        let mut next_front = Vec::new();
        for &i in &current_front {
            for &j in &dominated_by[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        current_front = next_front;
        level += 1;
    }

    let mut min_fit = fitnesses[0];
    let mut max_fit = fitnesses[0];
    for &f in fitnesses {
        if f < min_fit {
            min_fit = f;
        }
        if f > max_fit {
            max_fit = f;
        }
    }
    let fit_range = max_fit - min_fit;
    let perf_scores: Vec<f64> = fitnesses
        .iter()
        .map(|&f| {
            if fit_range > 0.0 {
                (f - min_fit) / fit_range
            } else {
                1.0
            }
        })
        .collect();

    let max_level = levels.iter().max().copied().unwrap_or(0);
    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        let base_score = (max_level - levels[i]) as f64;
        scores.push(base_score + 0.5 * perf_scores[i]);
    }

    let mut min_score = scores[0];
    let mut max_score = scores[0];
    for &s in &scores {
        if s < min_score {
            min_score = s;
        }
        if s > max_score {
            max_score = s;
        }
    }
    let score_range = max_score - min_score;
    let final_scores: Vec<f64> = scores
        .iter()
        .map(|&s| {
            if score_range > 0.0 {
                (s - min_score) / score_range
            } else {
                1.0
            }
        })
        .collect();

    final_scores
}

pub fn crossover<R: Rng>(
    parent_a: &Genome,
    parent_b: &Genome,
    fitness_a: f64,
    fitness_b: f64,
    rng: &mut R,
) -> Genome {
    let (fitter_parent, other_parent) = if fitness_b > fitness_a {
        (parent_b, parent_a)
    } else {
        (parent_a, parent_b)
    };

    let mut child_node_genes = Vec::new();
    let mut child_conn_genes = Vec::new();

    let ids_a = parent_a.node_ids();
    let ids_b = parent_b.node_ids();
    let all_node_ids: HashSet<usize> = ids_a.union(&ids_b).copied().collect();

    for nid in all_node_ids {
        let ng_a = parent_a.node_genes.get(&nid);
        let ng_b = parent_b.node_genes.get(&nid);
        let chosen = match (ng_a, ng_b) {
            (Some(a), Some(b)) => {
                if rng.gen_bool(0.5) {
                    a.clone()
                } else {
                    b.clone()
                }
            }
            (Some(a), None) => a.clone(),
            (None, Some(b)) => b.clone(),
            (None, None) => unreachable!(),
        };
        child_node_genes.push(chosen);
    }

    let innovs_a: HashSet<usize> = parent_a.conn_genes.keys().copied().collect();
    let innovs_b: HashSet<usize> = parent_b.conn_genes.keys().copied().collect();
    let shared: HashSet<usize> = innovs_a.intersection(&innovs_b).copied().collect();

    for i in shared {
        let chosen = if rng.gen_bool(0.5) {
            parent_a.conn_genes[&i].clone()
        } else {
            parent_b.conn_genes[&i].clone()
        };
        child_conn_genes.push(chosen);
    }

    let fitter_innovs = if fitness_b > fitness_a {
        &innovs_b
    } else {
        &innovs_a
    };
    let other_innovs = if fitness_b > fitness_a {
        &innovs_a
    } else {
        &innovs_b
    };

    for &i in fitter_innovs.difference(other_innovs) {
        child_conn_genes.push(fitter_parent.conn_genes[&i].clone());
    }

    Genome::new(
        Some(child_node_genes),
        Some(child_conn_genes),
        fitter_parent
            .next_innovation
            .max(other_parent.next_innovation),
    )
}

pub struct Population {
    pub config: Config,
    pub genomes: Vec<Genome>,
    pub fitnesses: Vec<f64>,
    pub species_list: Vec<Species>,
    pub generation: usize,
    pub next_species_id: usize,
    pub global_best_fitness: f64,
    pub global_best_genome: Option<Genome>,
}

impl Population {
    pub fn new<R: Rng>(config: Config, rng: &mut R, registry: &mut InnovationRegistry) -> Self {
        let mut genomes = Vec::new();
        let mut fitnesses = Vec::new();

        let pop_size = config.population.pop_size;
        let seed_fraction = config.population.seed_fraction;
        let n_seeds = ((pop_size as f64 * seed_fraction) as usize).max(1);

        // --- Seed genomes ---
        for i in 0..n_seeds {
            let strategy_idx = i % SEED_STRATEGIES.len();
            let strategy_conns = SEED_STRATEGIES[strategy_idx].1;
            let mut g = create_seed_genome(strategy_conns);

            // Add 1-2 random mutations
            let n_extra = rng.gen_range(0..2);
            for _ in 0..n_extra {
                apply_mutations(
                    &mut g,
                    registry,
                    rng,
                    config.mutation.p_add_node,
                    config.mutation.p_add_conn,
                    config.mutation.p_toggle_conn,
                    config.mutation.p_flip_sign,
                    config.mutation.p_change_act,
                    config.mutation.p_change_agg,
                );
            }
            genomes.push(g);
            fitnesses.push(0.0);
        }

        // --- Random-link genomes ---
        let base = Genome::initial();
        let remaining_count = pop_size - n_seeds;
        for i in 0..remaining_count {
            let mut g = base.copy();

            // Every genome gets 2-3 connections
            let n_conns = rng.gen_range(2..4);
            for _ in 0..n_conns {
                mutate_add_connection(&mut g, registry, rng);
            }

            // Top 30% get extra mutations including add_node
            if i < (remaining_count as f64 * 0.3) as usize {
                let n_mutations = rng.gen_range(2..5);
                for _ in 0..n_mutations {
                    apply_mutations(
                        &mut g,
                        registry,
                        rng,
                        config.mutation.p_add_node,
                        config.mutation.p_add_conn,
                        config.mutation.p_toggle_conn,
                        config.mutation.p_flip_sign,
                        config.mutation.p_change_act,
                        config.mutation.p_change_agg,
                    );
                }
            }

            genomes.push(g);
            fitnesses.push(0.0);
        }

        Self {
            config,
            genomes,
            fitnesses,
            species_list: Vec::new(),
            generation: 0,
            next_species_id: 0,
            global_best_fitness: f64::NEG_INFINITY,
            global_best_genome: None,
        }
    }

    pub fn tell_fitnesses(&mut self, fitnesses: &[f64]) {
        assert_eq!(fitnesses.len(), self.genomes.len());
        self.fitnesses = fitnesses.to_vec();

        // Track global best
        let mut best_idx = 0;
        let mut max_fit = fitnesses[0];
        for (i, &f) in fitnesses.iter().enumerate() {
            if f > max_fit {
                max_fit = f;
                best_idx = i;
            }
        }

        if max_fit > self.global_best_fitness {
            self.global_best_fitness = max_fit;
            self.global_best_genome = Some(self.genomes[best_idx].copy());
        }
    }

    pub fn speciate_and_evolve<R: Rng>(
        &mut self,
        current_phase: usize,
        bulking_gens: usize,
        rng: &mut R,
        registry: &mut InnovationRegistry,
    ) {
        // 1. Run Speciation
        self.next_species_id = speciate(
            &self.genomes,
            &mut self.species_list,
            self.config.species.compatibility_threshold,
            self.next_species_id,
            self.generation,
            self.config.species.c_excess,
            self.config.species.c_disjoint,
            self.config.species.c_mismatch,
        );

        // 2. Update stagnation
        for sp in self.species_list.iter_mut() {
            if sp.members.is_empty() {
                sp.increment_stagnation();
                continue;
            }
            let best_f = sp
                .members
                .iter()
                .map(|&idx| self.fitnesses[idx])
                .fold(f64::NEG_INFINITY, f64::max);
            let improved = sp.update_best(best_f);
            if !improved {
                sp.increment_stagnation();
            }
        }

        // Remove stagnated species unless it is the last active one
        let active_count = self
            .species_list
            .iter()
            .filter(|sp| !sp.members.is_empty())
            .count();
        if active_count > 1 {
            let limit = self.config.species.stagnation_limit;
            self.species_list
                .retain(|sp| sp.members.is_empty() || sp.generations_no_improvement < limit);
        }

        // 3. Breed next generation
        let new_genomes = self.breed_next_generation(current_phase, bulking_gens, rng, registry);
        self.genomes = new_genomes;
        self.fitnesses = vec![0.0; self.genomes.len()];
        self.generation += 1;
    }

    fn breed_next_generation<R: Rng>(
        &self,
        current_phase: usize,
        bulking_gens: usize,
        rng: &mut R,
        registry: &mut InnovationRegistry,
    ) -> Vec<Genome> {
        let mut active_species: Vec<&Species> = self
            .species_list
            .iter()
            .filter(|sp| !sp.members.is_empty())
            .collect();

        if active_species.is_empty() {
            return self.genomes.iter().map(|g| g.copy()).collect();
        }

        // Sort species by best fitness member descending
        active_species.sort_by(|a, b| {
            let best_a = a
                .members
                .iter()
                .map(|&idx| self.fitnesses[idx])
                .fold(f64::NEG_INFINITY, f64::max);
            let best_b = b
                .members
                .iter()
                .map(|&idx| self.fitnesses[idx])
                .fold(f64::NEG_INFINITY, f64::max);
            best_b.partial_cmp(&best_a).unwrap()
        });

        // Compute multi-objective ranking
        let is_bulking = current_phase == 0 && self.generation < bulking_gens;
        let selection_fitness = if is_bulking {
            rank_values(&self.fitnesses)
        } else {
            let complexities: Vec<usize> = self.genomes.iter().map(|g| g.num_enabled()).collect();
            if rng.gen_bool(self.config.population.pareto_complexity_prob) {
                pareto_rank(&self.fitnesses, &complexities)
            } else {
                rank_values(&self.fitnesses)
            }
        };

        // Compute offspring counts proportional to best member's selection fitness
        let mut total_fitness = 0.0;
        let mut species_best = Vec::new();
        for sp in &active_species {
            let best_f = sp
                .members
                .iter()
                .map(|&idx| selection_fitness[idx])
                .fold(0.0, f64::max);
            let adj_f = best_f + 0.1; // offset
            total_fitness += adj_f * sp.members.len() as f64;
            species_best.push(adj_f);
        }

        let mut offspring_counts = HashMap::new();
        let mut remaining = self.config.population.pop_size;
        let min_size = self.config.species.min_species_size;

        for (i, sp) in active_species.iter().enumerate() {
            if i == active_species.len() - 1 {
                offspring_counts.insert(sp.id, remaining);
            } else {
                let adj_f = species_best[i];
                let share = (self.config.population.pop_size as f64
                    * (adj_f * sp.members.len() as f64)
                    / total_fitness) as usize;

                let count = if remaining <= min_size {
                    0
                } else {
                    let max_allowed = remaining - min_size;
                    if min_size > max_allowed {
                        std::cmp::min(share, max_allowed)
                    } else {
                        share.clamp(min_size, max_allowed)
                    }
                };

                offspring_counts.insert(sp.id, count);
                remaining -= count;
            }
        }

        let mut new_genomes = Vec::with_capacity(self.config.population.pop_size);

        for sp in &active_species {
            let count = offspring_counts.get(&sp.id).copied().unwrap_or(0);
            if count == 0 {
                continue;
            }

            let members = &sp.members;
            let member_raw_fitnesses: Vec<f64> =
                members.iter().map(|&idx| selection_fitness[idx]).collect();
            let member_ranks = rank_values(&member_raw_fitnesses);

            // Sort members by rank descending
            let mut ranked: Vec<(usize, f64)> = members
                .iter()
                .copied()
                .zip(member_ranks.iter().copied())
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let ranked_indices: Vec<usize> = ranked.iter().map(|x| x.0).collect();
            let ranked_ranks: Vec<f64> = ranked.iter().map(|x| x.1).collect();

            // Elitism
            let elite_count = self
                .config
                .population
                .elitism
                .min(count)
                .min(ranked_indices.len());
            for idx in 0..elite_count {
                new_genomes.push(self.genomes[ranked_indices[idx]].copy());
            }

            // Offspring
            let breeding_count = count - elite_count;
            for _ in 0..breeding_count {
                let mut child = if rng.gen_bool(self.config.mutation.p_crossover)
                    && ranked_indices.len() >= 2
                {
                    let p1 = tournament_select(&ranked_indices, &ranked_ranks, 3, rng);
                    let p2 = tournament_select(&ranked_indices, &ranked_ranks, 3, rng);
                    crossover(
                        &self.genomes[p1],
                        &self.genomes[p2],
                        self.fitnesses[p1],
                        self.fitnesses[p2],
                        rng,
                    )
                } else {
                    let parent = tournament_select(&ranked_indices, &ranked_ranks, 3, rng);
                    self.genomes[parent].copy()
                };

                apply_mutations(
                    &mut child,
                    registry,
                    rng,
                    self.config.mutation.p_add_node,
                    self.config.mutation.p_add_conn,
                    self.config.mutation.p_toggle_conn,
                    self.config.mutation.p_flip_sign,
                    self.config.mutation.p_change_act,
                    self.config.mutation.p_change_agg,
                );
                new_genomes.push(child);
            }
        }

        // Adjust to exact population size
        let pop_size = self.config.population.pop_size;
        if new_genomes.len() > pop_size {
            new_genomes.truncate(pop_size);
        } else {
            while new_genomes.len() < pop_size {
                let best_avail = new_genomes
                    .first()
                    .map(|g| g.copy())
                    .unwrap_or_else(|| Genome::initial());
                new_genomes.push(best_avail);
            }
        }

        new_genomes
    }
}

fn tournament_select<R: Rng>(
    ranked_indices: &[usize],
    rank_values: &[f64],
    k: usize,
    rng: &mut R,
) -> usize {
    let size = k.min(ranked_indices.len());
    let mut best_idx = rng.gen_range(0..ranked_indices.len());
    let mut best_val = rank_values[best_idx];

    for _ in 1..size {
        let idx = rng.gen_range(0..ranked_indices.len());
        if rank_values[idx] > best_val {
            best_val = rank_values[idx];
            best_idx = idx;
        }
    }

    ranked_indices[best_idx]
}
