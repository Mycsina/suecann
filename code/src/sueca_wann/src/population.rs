use crate::config::Config;
use crate::dataset::ExpertDataset;
use crate::genome::Genome;
use crate::mutations::{apply_mutations, InnovationRegistry, MutationOutcome, TabuVetoList};
use crate::species::{speciate, Species};
use crate::wann_network::RustWannNetwork;

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
    pub fn new<R: Rng>(
        config: Config,
        _rng: &mut R,
        _registry: &Mutex<InnovationRegistry>,
    ) -> Self {
        let mut genomes = Vec::new();
        let mut fitnesses = Vec::new();

        let pop_size = config.population.pop_size;

        // --- Zero-connection genomes for PFS-NEAT ---
        let base = Genome::initial();
        let remaining_count = pop_size;
        for _ in 0..remaining_count {
            genomes.push(base.copy());
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
        tabu_list: &TabuVetoList,
        phase0_eval_data: Option<(&ExpertDataset, &[f64])>,
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

        // Enforce species cap to prevent proliferation
        crate::species::enforce_species_cap(
            &mut self.species_list,
            self.config.species.max_species,
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
        let new_genomes = self.breed_next_generation(
            current_phase,
            bulking_gens,
            tabu_list,
            phase0_eval_data,
            rng,
            registry,
        );
        self.genomes = new_genomes;
        self.fitnesses = vec![0.0; self.genomes.len()];
        self.generation += 1;
    }




    fn breed_next_generation<R: Rng>(
        &self,
        current_phase: usize,
        bulking_gens: usize,
        tabu_list: &TabuVetoList,
        phase0_eval_data: Option<(&ExpertDataset, &[f64])>,
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

                        let parent_copy = child.copy();
                        let outcome = apply_mutations(
                            &mut child,
                            registry,
                            tabu_list,
                            local_rng,
                            p_add_node,
                            p_add_conn,
                            p_toggle_conn,
                            p_flip_sign,
                            p_change_act,
                            p_change_agg,
                        );

                        // PFS-NEAT: only validate structural mutations (topology changes).
                        // Parameter tweaks (sign flip, act/agg change) are inherently
                        // incremental and rarely cause catastrophic accuracy loss.
                        if outcome == MutationOutcome::Structural && current_phase == 0 {
                            if let Some((dataset, sweep_weights)) = phase0_eval_data {
                                let parent_net = parent_copy.to_rust_wann();
                                let child_net = child.to_rust_wann();
                                let pfs_k = self.config.curriculum.pfs_sample_size;

                                // Pre-allocate scratchpad once for both parent and child
                                let max_nodes = parent_net.num_nodes.max(child_net.num_nodes);
                                let mut pfs_scratchpad = vec![0.0f64; max_nodes * sweep_weights.len()];

                                // Adaptive 2-stage: quick K=25 check first.
                                let quick_k = 25.min(pfs_k);
                                let parent_quick = evaluate_single_phase0_sample_scratchpad(
                                    &parent_net, dataset, sweep_weights, quick_k,
                                    &mut pfs_scratchpad,
                                );
                                let child_quick = evaluate_single_phase0_sample_scratchpad(
                                    &child_net, dataset, sweep_weights, quick_k,
                                    &mut pfs_scratchpad,
                                );

                                let diff = child_quick - parent_quick;
                                let child_acc = if diff < -0.05 {
                                    child_quick
                                } else if diff.abs() <= 0.02 && pfs_k > quick_k {
                                    evaluate_single_phase0_sample_scratchpad(
                                        &child_net, dataset, sweep_weights, pfs_k,
                                        &mut pfs_scratchpad,
                                    )
                                } else {
                                    child_quick
                                };

                                let parent_acc = if diff.abs() <= 0.02 && pfs_k > quick_k {
                                    evaluate_single_phase0_sample_scratchpad(
                                        &parent_net, dataset, sweep_weights, pfs_k,
                                        &mut pfs_scratchpad,
                                    )
                                } else {
                                    parent_quick
                                };

                                if child_acc < parent_acc {
                                    for c in &child.conn_genes {
                                        if c.enabled && !parent_copy.has_connection(c.src, c.dst) {
                                            tabu_list.add_tabu(c.src, c.dst);
                                        }
                                    }
                                    child = parent_copy;
                                }
                            }
                        }
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
        tabu_list: &TabuVetoList,
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
                tabu_list,
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

pub fn evaluate_single_phase0_sample(
    network: &RustWannNetwork,
    dataset: &ExpertDataset,
    sweep_weights: &[f64],
    sample_size: usize,
) -> f64 {
    let mut scratchpad = vec![0.0f64; network.num_nodes * sweep_weights.len()];
    evaluate_single_phase0_sample_scratchpad(
        network, dataset, sweep_weights, sample_size, &mut scratchpad,
    )
}

/// Same as `evaluate_single_phase0_sample` but reuses a caller-provided
/// scratchpad to avoid allocation in hot loops (PFS-NEAT breeding).
#[inline(always)]
pub fn evaluate_single_phase0_sample_scratchpad(
    network: &RustWannNetwork,
    dataset: &ExpertDataset,
    sweep_weights: &[f64],
    sample_size: usize,
    scratchpad: &mut [f64],
) -> f64 {
    use crate::genome::{INPUT_COUNT, OUTPUT_COUNT, OUTPUT_START};

    let n_states = dataset.num_states.min(sample_size);
    if n_states == 0 {
        return 0.0;
    }

    // Ensure scratchpad is large enough for weight-batched forward pass
    let n_weights = sweep_weights.len();
    debug_assert!(scratchpad.len() >= network.num_nodes * n_weights);
    let mut correct: u32 = 0;

    for idx in 0..n_states {
        // Zero-copy: read directly from flat array into stack array
        let base = idx * INPUT_COUNT;
        let mut inputs = [0.0f64; INPUT_COUNT];
        inputs.copy_from_slice(&dataset.states[base..base + INPUT_COUNT]);

        network.forward_batched(&inputs, sweep_weights, scratchpad);

        let mut total_outputs = [0.0f64; OUTPUT_COUNT];
        for w in 0..n_weights {
            total_outputs[0] += scratchpad[(OUTPUT_START + 0) * n_weights + w];
            total_outputs[1] += scratchpad[(OUTPUT_START + 1) * n_weights + w];
            total_outputs[2] += scratchpad[(OUTPUT_START + 2) * n_weights + w];
        }

        // Unrolled argmax (3 intents)
        let v0 = total_outputs[0];
        let v1 = total_outputs[1];
        let v2 = total_outputs[2];

        let best_intent = if v0 >= v1 && v0 >= v2 {
            0
        } else if v1 >= v2 {
            1
        } else {
            2
        };

        if dataset.soft_intents[idx * OUTPUT_COUNT + best_intent] > 0.0 {
            correct += 1;
        }
    }

    correct as f64 / n_states as f64
}

pub fn speciate_and_evolve_joint(
    lead_pop: &mut Population,
    follow_pop: &mut Population,
    lead_tabu: &TabuVetoList,
    follow_tabu: &TabuVetoList,
    lead_registry: &Mutex<InnovationRegistry>,
    follow_registry: &Mutex<InnovationRegistry>,
    rng: &mut rand_pcg::Pcg64,
) {
    // 1. Run Speciation on lead_pop (as the master partition)
    lead_pop.next_species_id = crate::species::speciate(
        &lead_pop.genomes,
        &mut lead_pop.species_list,
        lead_pop.config.species.compatibility_threshold,
        lead_pop.next_species_id,
        lead_pop.generation,
        lead_pop.config.species.c_excess,
        lead_pop.config.species.c_disjoint,
        lead_pop.config.species.c_mismatch,
    );

    // Enforce species cap
    crate::species::enforce_species_cap(
        &mut lead_pop.species_list,
        lead_pop.config.species.max_species,
        lead_pop.config.species.c_excess,
        lead_pop.config.species.c_disjoint,
        lead_pop.config.species.c_mismatch,
    );

    // Copy species list to follow_pop to keep species/members partition aligned
    follow_pop.species_list = lead_pop.species_list.clone();
    follow_pop.next_species_id = lead_pop.next_species_id;

    // 2. Update stagnation for the species (using the shared joint fitnesses)
    let fitnesses = &lead_pop.fitnesses;
    lead_pop.species_list.par_iter_mut().for_each(|sp| {
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
    let active_count = lead_pop
        .species_list
        .iter()
        .filter(|sp| !sp.members.is_empty())
        .count();
    if active_count > 1 {
        let limit = lead_pop.config.species.stagnation_limit;
        lead_pop.species_list
            .retain(|sp| sp.members.is_empty() || sp.generations_no_improvement < limit);
    }
    follow_pop.species_list = lead_pop.species_list.clone();

    // 3. Breed next generation jointly to preserve alignment
    let (new_lead_genomes, new_follow_genomes) = breed_next_generation_joint(
        lead_pop,
        follow_pop,
        lead_tabu,
        follow_tabu,
        lead_registry,
        follow_registry,
        rng,
    );

    lead_pop.genomes = new_lead_genomes;
    lead_pop.fitnesses = vec![0.0; lead_pop.genomes.len()];
    lead_pop.generation += 1;

    follow_pop.genomes = new_follow_genomes;
    follow_pop.fitnesses = vec![0.0; follow_pop.genomes.len()];
    follow_pop.generation += 1;
}

pub fn breed_next_generation_joint(
    lead_pop: &Population,
    follow_pop: &Population,
    lead_tabu: &TabuVetoList,
    follow_tabu: &TabuVetoList,
    lead_registry: &Mutex<InnovationRegistry>,
    follow_registry: &Mutex<InnovationRegistry>,
    rng: &mut rand_pcg::Pcg64,
) -> (Vec<Genome>, Vec<Genome>) {
    let mut active_species: Vec<&Species> = lead_pop
        .species_list
        .iter()
        .filter(|sp| !sp.members.is_empty())
        .collect();

    if active_species.is_empty() {
        return (
            lead_pop.genomes.iter().map(|g| g.copy()).collect(),
            follow_pop.genomes.iter().map(|g| g.copy()).collect(),
        );
    }

    // Precompute best fitness per species to avoid re-scanning during sort
    let mut best_by_id: std::collections::HashMap<usize, f64> =
        std::collections::HashMap::new();
    for sp in active_species.iter() {
        let best = sp
            .members
            .iter()
            .map(|&idx| lead_pop.fitnesses[idx])
            .fold(f64::NEG_INFINITY, f64::max);
        best_by_id.insert(sp.id, best);
    }
    active_species.sort_by(|a, b| {
        let best_a = best_by_id.get(&a.id).copied().unwrap_or(f64::NEG_INFINITY);
        let best_b = best_by_id.get(&b.id).copied().unwrap_or(f64::NEG_INFINITY);
        best_b.partial_cmp(&best_a).unwrap()
    });

    // Compute joint complexity to rank the teams
    let complexities_lead: Vec<f64> = lead_pop
        .genomes
        .par_iter()
        .map(|g| g.calculate_complexity())
        .collect();
    let complexities_follow: Vec<f64> = follow_pop
        .genomes
        .par_iter()
        .map(|g| g.calculate_complexity())
        .collect();
    let complexities: Vec<f64> = complexities_lead
        .iter()
        .zip(complexities_follow.iter())
        .map(|(&l, &f)| l + f)
        .collect();

    let selection_fitness = if rng.gen_bool(lead_pop.config.population.pareto_complexity_prob) {
        pareto_rank(&lead_pop.fitnesses, &complexities)
    } else {
        rank_values(&lead_pop.fitnesses)
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
    let mut remaining = lead_pop.config.population.pop_size;
    let min_size = lead_pop.config.species.min_species_size;

    for (i, sp) in active_species.iter().enumerate() {
        if i == active_species.len() - 1 {
            offspring_counts.insert(sp.id, remaining);
        } else {
            let adj_f = species_best[i];
            let share = (lead_pop.config.population.pop_size as f64
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

    let mut new_lead_genomes = Vec::with_capacity(lead_pop.config.population.pop_size);
    let mut new_follow_genomes = Vec::with_capacity(follow_pop.config.population.pop_size);

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

        // Elitism (keep aligned)
        let elite_count = lead_pop
            .config
            .population
            .elitism
            .min(count)
            .min(ranked_indices.len());
        for idx in 0..elite_count {
            new_lead_genomes.push(lead_pop.genomes[ranked_indices[idx]].copy());
            new_follow_genomes.push(follow_pop.genomes[ranked_indices[idx]].copy());
        }

        // Offspring — parallelized
        let breeding_count = count - elite_count;
        if breeding_count == 0 {
            continue;
        }

        let p_crossover = lead_pop.config.mutation.p_crossover;
        let p_add_node = lead_pop.config.mutation.p_add_node;
        let p_add_conn = lead_pop.config.mutation.p_add_conn;
        let p_toggle_conn = lead_pop.config.mutation.p_toggle_conn;
        let p_flip_sign = lead_pop.config.mutation.p_flip_sign;
        let p_change_act = lead_pop.config.mutation.p_change_act;
        let p_change_agg = lead_pop.config.mutation.p_change_agg;

        let seeds: Vec<u64> = (0..breeding_count).map(|_| rng.gen::<u64>()).collect();

        let offspring: Vec<(Genome, Genome)> = seeds
            .into_par_iter()
            .map_init(
                || Pcg64::seed_from_u64(42),
                |local_rng, seed| {
                    *local_rng = Pcg64::seed_from_u64(seed);
                    let (mut child_lead, mut child_follow) =
                        if local_rng.gen_bool(p_crossover) && ranked_indices.len() >= 2 {
                            let p1 =
                                tournament_select(&ranked_indices, &ranked_ranks, 3, local_rng);
                            let p2 =
                                tournament_select(&ranked_indices, &ranked_ranks, 3, local_rng);
                            let c_lead = crossover(
                                &lead_pop.genomes[p1],
                                &lead_pop.genomes[p2],
                                lead_pop.fitnesses[p1],
                                lead_pop.fitnesses[p2],
                                local_rng,
                            );
                            let c_follow = crossover(
                                &follow_pop.genomes[p1],
                                &follow_pop.genomes[p2],
                                follow_pop.fitnesses[p1],
                                follow_pop.fitnesses[p2],
                                local_rng,
                            );
                            (c_lead, c_follow)
                        } else {
                            let parent =
                                tournament_select(&ranked_indices, &ranked_ranks, 3, local_rng);
                            (lead_pop.genomes[parent].copy(), follow_pop.genomes[parent].copy())
                        };

                    apply_mutations(
                        &mut child_lead,
                        lead_registry,
                        lead_tabu,
                        local_rng,
                        p_add_node,
                        p_add_conn,
                        p_toggle_conn,
                        p_flip_sign,
                        p_change_act,
                        p_change_agg,
                    );

                    apply_mutations(
                        &mut child_follow,
                        follow_registry,
                        follow_tabu,
                        local_rng,
                        p_add_node,
                        p_add_conn,
                        p_toggle_conn,
                        p_flip_sign,
                        p_change_act,
                        p_change_agg,
                    );

                    (child_lead, child_follow)
                },
            )
            .collect();

        for (l, f) in offspring {
            new_lead_genomes.push(l);
            new_follow_genomes.push(f);
        }
    }

    // Adjust to exact population size (keeping alignment)
    let pop_size = lead_pop.config.population.pop_size;
    if new_lead_genomes.len() > pop_size {
        new_lead_genomes.truncate(pop_size);
        new_follow_genomes.truncate(pop_size);
    } else {
        while new_lead_genomes.len() < pop_size {
            let best_lead = new_lead_genomes
                .first()
                .map(|g| g.copy())
                .unwrap_or_else(Genome::initial);
            let best_follow = new_follow_genomes
                .first()
                .map(|g| g.copy())
                .unwrap_or_else(Genome::initial);
            new_lead_genomes.push(best_lead);
            new_follow_genomes.push(best_follow);
        }
    }

    (new_lead_genomes, new_follow_genomes)
}

