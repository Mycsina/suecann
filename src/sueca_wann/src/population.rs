use crate::config::Config;
use crate::genome::Genome;
use crate::mutations::{apply_mutations, mutate_add_connection, InnovationRegistry};
use crate::species::{speciate, Species};
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

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

pub fn pareto_rank(fitnesses: &[f64], complexities: &[f64]) -> Vec<f64> {
    let n = fitnesses.len();
    if n == 0 {
        return Vec::new();
    }

    let max_complexity = complexities.iter().fold(0.0f64, |a, &b| a.max(b));
    let simplicities: Vec<f64> = complexities.iter().map(|&c| max_complexity - c).collect();

    // Parallel domination detection — each i computes its own results independently
    let results: Vec<(usize, Vec<usize>, usize)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let f_i = fitnesses[i];
            let s_i = simplicities[i];
            let mut dominated = Vec::new();
            let mut dom_count = 0;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let f_j = fitnesses[j];
                let s_j = simplicities[j];
                if f_i >= f_j && s_i >= s_j && (f_i > f_j || s_i > s_j) {
                    dominated.push(j);
                }
                if f_j >= f_i && s_j >= s_i && (f_j > f_i || s_j > s_i) {
                    dom_count += 1;
                }
            }
            (i, dominated, dom_count)
        })
        .collect();

    let mut domination_count = vec![0; n];
    let mut dominated_by = vec![Vec::new(); n];
    for (i, dominated, count) in results {
        domination_count[i] = count;
        dominated_by[i] = dominated;
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
    let self_is_fitter = fitness_a > fitness_b;
    let fitter = if self_is_fitter { parent_a } else { parent_b };
    let lesser = if self_is_fitter { parent_b } else { parent_a };
    fitter.crossover_with(lesser, true, rng)
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
    pub generations_since_improvement: usize,
}

impl Population {
    pub fn new<R: Rng>(config: Config, rng: &mut R, registry: &Mutex<InnovationRegistry>) -> Self {
        let mut genomes = Vec::new();
        let mut fitnesses = Vec::new();

        let pop_size = config.population.pop_size;

        // --- Random-link genomes ---
        let base = Genome::initial();
        let remaining_count = pop_size;
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
            generations_since_improvement: 0,
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
            self.generations_since_improvement = 0;
        } else {
            self.generations_since_improvement += 1;
        }
    }

    pub fn speciate_and_evolve<R: Rng>(
        &mut self,
        current_phase: usize,
        bulking_gens: usize,
        rng: &mut R,
        registry: &Mutex<InnovationRegistry>,
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

        // 2. Update stagnation (parallel — each species independent)
        let fitnesses = &self.fitnesses;
        self.species_list.par_iter_mut().for_each(|sp| {
            if sp.members.is_empty() {
                sp.increment_stagnation();
                return;
            }
            let best_f = sp
                .members
                .iter()
                .map(|&idx| fitnesses[idx])
                .fold(f64::NEG_INFINITY, f64::max);
            let improved = sp.update_best(best_f);
            if !improved {
                sp.increment_stagnation();
            }
        });

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
        registry: &Mutex<InnovationRegistry>,
    ) -> Vec<Genome> {
        let mut active_species: Vec<&Species> = self
            .species_list
            .iter()
            .filter(|sp| !sp.members.is_empty())
            .collect();

        if active_species.is_empty() {
            return self.genomes.iter().map(|g| g.copy()).collect();
        }

        // Precompute best fitness per species to avoid re-scanning during sort
        let mut best_by_id: std::collections::HashMap<usize, f64> =
            std::collections::HashMap::new();
        for sp in active_species.iter() {
            let best = sp
                .members
                .iter()
                .map(|&idx| self.fitnesses[idx])
                .fold(f64::NEG_INFINITY, f64::max);
            best_by_id.insert(sp.id, best);
        }
        active_species.sort_by(|a, b| {
            let best_a = best_by_id.get(&a.id).copied().unwrap_or(f64::NEG_INFINITY);
            let best_b = best_by_id.get(&b.id).copied().unwrap_or(f64::NEG_INFINITY);
            best_b.partial_cmp(&best_a).unwrap()
        });

        // Compute multi-objective ranking
        let is_bulking = current_phase == 0 && self.generation < bulking_gens;
        let selection_fitness = if is_bulking {
            rank_values(&self.fitnesses)
        } else {
            let complexities: Vec<f64> = self
                .genomes
                .par_iter()
                .map(|g| g.calculate_complexity())
                .collect();
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

            // Elitism (small count, keep serial)
            let elite_count = self
                .config
                .population
                .elitism
                .min(count)
                .min(ranked_indices.len());
            for idx in 0..elite_count {
                new_genomes.push(self.genomes[ranked_indices[idx]].copy());
            }

            // Offspring — parallelized
            let breeding_count = count - elite_count;
            if breeding_count == 0 {
                continue;
            }

            let genomes = &self.genomes;
            let fitnesses = &self.fitnesses;
            let p_crossover = self.config.mutation.p_crossover;
            let p_add_node = self.config.mutation.p_add_node;
            let p_add_conn = self.config.mutation.p_add_conn;
            let p_toggle_conn = self.config.mutation.p_toggle_conn;
            let p_flip_sign = self.config.mutation.p_flip_sign;
            let p_change_act = self.config.mutation.p_change_act;
            let p_change_agg = self.config.mutation.p_change_agg;
            // Pre-generate seeds to avoid sharing rng across threads
            let seeds: Vec<u64> = (0..breeding_count).map(|_| rng.gen::<u64>()).collect();

            let offspring: Vec<Genome> = seeds
                .into_par_iter()
                .map_init(
                    || Pcg64::seed_from_u64(42), // placeholder, each seed overrides
                    |local_rng, seed| {
                        *local_rng = Pcg64::seed_from_u64(seed);
                        let mut child =
                            if local_rng.gen_bool(p_crossover) && ranked_indices.len() >= 2 {
                                let p1 =
                                    tournament_select(&ranked_indices, &ranked_ranks, 3, local_rng);
                                let p2 =
                                    tournament_select(&ranked_indices, &ranked_ranks, 3, local_rng);
                                crossover(
                                    &genomes[p1],
                                    &genomes[p2],
                                    fitnesses[p1],
                                    fitnesses[p2],
                                    local_rng,
                                )
                            } else {
                                let parent =
                                    tournament_select(&ranked_indices, &ranked_ranks, 3, local_rng);
                                genomes[parent].copy()
                            };

                        apply_mutations(
                            &mut child,
                            registry,
                            local_rng,
                            p_add_node,
                            p_add_conn,
                            p_toggle_conn,
                            p_flip_sign,
                            p_change_act,
                            p_change_agg,
                        );
                        child
                    },
                )
                .collect();
            new_genomes.extend(offspring);
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
                    .unwrap_or_else(Genome::initial);
                new_genomes.push(best_avail);
            }
        }

        new_genomes
    }

    pub fn reseed_from_champion(
        &mut self,
        global_best_genome: &Genome,
        fraction: f64,
        registry: &Mutex<InnovationRegistry>,
        rng: &mut rand_pcg::Pcg64,
    ) {
        let c1 = self.config.species.c_excess;
        let c2 = self.config.species.c_disjoint;
        let c3 = self.config.species.c_mismatch;

        // Find the species representing the champion (closest representative)
        let mut best_sp_idx = None;
        let mut min_dist = f64::INFINITY;

        for (i, sp) in self.species_list.iter().enumerate() {
            if sp.members.is_empty() {
                continue;
            }
            let dist = crate::species::compatibility_distance(
                global_best_genome,
                &sp.representative,
                c1,
                c2,
                c3,
            );
            if dist < min_dist {
                min_dist = dist;
                best_sp_idx = Some(i);
            }
        }

        let sp_idx = match best_sp_idx {
            Some(idx) => idx,
            None => return,
        };

        let mut members = self.species_list[sp_idx].members.clone();
        if members.is_empty() {
            return;
        }

        // Sort by fitness ascending (weakest first)
        let fitnesses = &self.fitnesses;
        members.sort_by(|&a, &b| fitnesses[a].partial_cmp(&fitnesses[b]).unwrap());

        let count = (((members.len() as f64) * fraction) as usize).max(1);

        for i in 0..count.min(members.len()) {
            let idx = members[i];
            let mut clone = global_best_genome.copy();

            // Slightly mutate the clone
            let p_add_node = 0.05;
            let p_add_conn = 0.10;
            let p_toggle_conn = 0.05;
            let p_flip_sign = 0.05;
            let p_change_act = 0.05;
            let p_change_agg = 0.05;

            crate::mutations::apply_mutations(
                &mut clone,
                registry,
                rng,
                p_add_node,
                p_add_conn,
                p_toggle_conn,
                p_flip_sign,
                p_change_act,
                p_change_agg,
            );

            self.genomes[idx] = clone;
            self.fitnesses[idx] = 0.0;
        }
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
