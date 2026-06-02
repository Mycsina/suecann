use crate::checkpoint::{TrainingState, BrainTrainingState};
use crate::config::Config;
use crate::dataset::{load_expert_dataset, ExpertDataset};
use crate::genome::{JsonGenome, JsonGenomeJoint, FIRST_HIDDEN_ID, INPUT_COUNT, OUTPUT_COUNT, OUTPUT_START};
use crate::hall_of_fame::{HallOfFame, JsonHallOfFame, JsonHallOfFameJoint};
use crate::mutations::{InnovationRegistry, TabuVetoList};
use crate::population::Population;

use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use rayon::prelude::*;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

pub fn generate_deals_rust(
    gen: usize,
    n_deals: usize,
    base_seed: u64,
) -> Vec<crate::evaluator::EvaluatorDeal> {
    let seed = base_seed + gen as u64;
    let mut deals = Vec::new();

    for i in 0..n_deals {
        let deal_seed = seed * 1000 + i as u64;
        let mut rng = Pcg64::seed_from_u64(deal_seed);

        let mut deck: Vec<u8> = (0..40).collect();
        deck.shuffle(&mut rng);

        let mut hands = [0u64; 4];
        for player in 0..4 {
            for card_idx in 0..10 {
                let card = deck[player * 10 + card_idx];
                hands[player] |= 1u64 << card;
            }
        }

        let trump = rng.gen_range(0..4) as u8;
        deals.push(crate::evaluator::EvaluatorDeal {
            hands,
            trump,
            seed: deal_seed,
        });
    }

    deals
}

// ---------------------------------------------------------------------------
// Phase 0: Supervised classification accuracy
// ---------------------------------------------------------------------------
pub fn evaluate_phase0(
    genomes: &[crate::wann_network::RustWannNetwork],
    dataset: &ExpertDataset,
    sweep_weights: &[f64],
) -> Vec<f64> {
    let max_nodes = genomes
        .iter()
        .map(|g| g.num_nodes)
        .max()
        .unwrap_or(FIRST_HIDDEN_ID);

    genomes
        .into_par_iter()
        .map(|candidate| {
            let mut scratchpad = vec![0.0f64; max_nodes];
            let mut correct = 0.0;

            for idx in 0..dataset.num_states {
                let mut inputs = [0.0f64; INPUT_COUNT];
                for i in 0..INPUT_COUNT {
                    inputs[i] = dataset.states[idx * INPUT_COUNT + i];
                }

                let mut total_outputs = [0.0f64; OUTPUT_COUNT];
                for &w in sweep_weights {
                    candidate.forward(&inputs, w, &mut scratchpad);
                    for i in 0..OUTPUT_COUNT {
                        total_outputs[i] += scratchpad[OUTPUT_START + i];
                    }
                }

                let mut best_intent = 0;
                let mut max_val = f64::NEG_INFINITY;
                for i in 0..OUTPUT_COUNT {
                    let val = if i == 3 {
                        total_outputs[i] - 0.25 * (sweep_weights.len() as f64)
                    } else {
                        total_outputs[i]
                    };
                    if val > max_val {
                        max_val = val;
                        best_intent = i;
                    }
                }

                correct += dataset.soft_intents[idx * 4 + best_intent] as f64;
            }

            correct / dataset.num_states as f64
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Phase 1: Self-play with HOF opponents
// ---------------------------------------------------------------------------
#[allow(clippy::too_many_arguments)]
pub fn evaluate_phase1(
    genomes: &[crate::wann_network::RustWannNetwork],
    is_lead: bool,
    deals: &[crate::evaluator::EvaluatorDeal],
    reference_brain: &crate::wann_network::RustWannNetwork,
    hof_lead_networks: &[crate::wann_network::RustWannNetwork],
    hof_follow_networks: &[crate::wann_network::RustWannNetwork],
    partner_bot_type: i32,
    opp1_bot_type: i32,
    opp2_bot_type: i32,
    baseline_bot_type: i32,
    sweep_weights: &[f64],
    base_seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<crate::evaluator::WannBehavior>) {
    let max_nodes = genomes
        .iter()
        .map(|g| g.num_nodes)
        .chain(std::iter::once(reference_brain.num_nodes))
        .chain(hof_lead_networks.iter().map(|g| g.num_nodes))
        .chain(hof_follow_networks.iter().map(|g| g.num_nodes))
        .max()
        .unwrap_or(FIRST_HIDDEN_ID);

    let results: Vec<(f64, crate::evaluator::WannBehavior)> = genomes
        .into_par_iter()
        .enumerate()
        .map(|(i, candidate)| {
            let mut scratchpad = vec![0.0f64; max_nodes];
            let (candidate_lead, candidate_follow) = if is_lead {
                (candidate, reference_brain)
            } else {
                (reference_brain, candidate)
            };
            crate::evaluator::evaluate_genome_delta(
                candidate_lead,
                candidate_follow,
                baseline_bot_type,
                partner_bot_type,
                opp1_bot_type,
                opp2_bot_type,
                hof_lead_networks,
                hof_follow_networks,
                sweep_weights,
                deals,
                base_seed + (i as u64),
                &mut scratchpad,
            )
        })
        .collect();

    let mut fitnesses = Vec::with_capacity(genomes.len());
    let mut deltas = Vec::with_capacity(genomes.len());
    let mut behaviors = Vec::with_capacity(genomes.len());

    for (delta, behavior) in results {
        fitnesses.push(delta);
        deltas.push(delta);
        behaviors.push(behavior);
    }

    (fitnesses, deltas, behaviors)
}

// ---------------------------------------------------------------------------
// Per-generation evaluation dispatchers
// ---------------------------------------------------------------------------
fn run_phase1_generation(
    genomes: &[crate::wann_network::RustWannNetwork],
    is_lead: bool,
    hof_lead: &HallOfFame,
    hof_follow: &HallOfFame,
    map_elites_lead: &crate::map_elites::MapElitesArchive,
    map_elites_follow: &crate::map_elites::MapElitesArchive,
    reference_brain: &crate::wann_network::RustWannNetwork,
    config: &Config,
    gen: usize,
    rng: &mut Pcg64,
) -> (Vec<f64>, Vec<f64>, Vec<crate::evaluator::WannBehavior>) {
    let deals = generate_deals_rust(
        gen,
        config.evaluation.n_deals,
        config.evaluation.seed * 1000,
    );

    let mut hof_lead_nets = Vec::new();
    let mut hof_follow_nets = Vec::new();

    let sample_seat_bot = |rng: &mut Pcg64,
                           h_lead: &HallOfFame,
                           h_follow: &HallOfFame,
                           me_lead: &crate::map_elites::MapElitesArchive,
                           me_follow: &crate::map_elites::MapElitesArchive,
                           hg_lead: &mut Vec<crate::wann_network::RustWannNetwork>,
                           hg_follow: &mut Vec<crate::wann_network::RustWannNetwork>|
     -> i32 {
        let elite_prob = ((gen as f64 - 200.0) / 400.0).clamp(0.0, 0.40);

        if rng.gen_bool(elite_prob) {
            2 // EliteHeuristicBot
        } else if rng.gen_bool(0.5) {
            let use_map_elites = rng.gen_bool(0.5);
            let lead_genome = if use_map_elites {
                me_lead.sample_random(rng)
            } else {
                None
            }.or_else(|| h_lead.sample(rng, 1).first().cloned());

            let follow_genome = if use_map_elites {
                me_follow.sample_random(rng)
            } else {
                None
            }.or_else(|| h_follow.sample(rng, 1).first().cloned());

            if let (Some(g_lead), Some(g_follow)) = (lead_genome, follow_genome) {
                let bot_type = 10 + hg_lead.len() as i32; // WANNs start at 10
                hg_lead.push(g_lead.to_rust_wann());
                hg_follow.push(g_follow.to_rust_wann());
                bot_type
            } else {
                1 // OldHeuristicBot
            }
        } else {
            1 // Old HeuristicBot (Baseline Sanity)
        }
    };

    let mut sample_rng = rng.clone();
    let partner_bot_type: i32 = sample_seat_bot(
        &mut sample_rng,
        hof_lead,
        hof_follow,
        map_elites_lead,
        map_elites_follow,
        &mut hof_lead_nets,
        &mut hof_follow_nets,
    );
    let opp1_bot_type: i32 = sample_seat_bot(
        &mut sample_rng,
        hof_lead,
        hof_follow,
        map_elites_lead,
        map_elites_follow,
        &mut hof_lead_nets,
        &mut hof_follow_nets,
    );
    let opp2_bot_type: i32 = sample_seat_bot(
        &mut sample_rng,
        hof_lead,
        hof_follow,
        map_elites_lead,
        map_elites_follow,
        &mut hof_lead_nets,
        &mut hof_follow_nets,
    );
    *rng = sample_rng;

    evaluate_phase1(
        genomes,
        is_lead,
        &deals,
        reference_brain,
        &hof_lead_nets,
        &hof_follow_nets,
        partner_bot_type,
        opp1_bot_type,
        opp2_bot_type,
        2, // baseline = HeuristicBot
        &config.evaluation.sweep_weights,
        config.evaluation.seed + gen as u64 * 1000,
    )
}

fn split_dataset(dataset: &ExpertDataset) -> (ExpertDataset, ExpertDataset) {
    let mut lead_states = Vec::new();
    let mut lead_soft_intents = Vec::new();
    let mut lead_legal_masks = Vec::new();

    let mut follow_states = Vec::new();
    let mut follow_soft_intents = Vec::new();
    let mut follow_legal_masks = Vec::new();

    for idx in 0..dataset.num_states {
        let state_offset = idx * INPUT_COUNT;
        let is_leading = (dataset.states[state_offset + crate::constants::BeliefFeature::AmILeading as usize] - 1.0).abs() < 1e-9;
        
        let state_slice = &dataset.states[state_offset..state_offset + INPUT_COUNT];
        let intent_slice = &dataset.soft_intents[idx * 4..idx * 4 + 4];
        let legal_mask = dataset.legal_masks[idx];

        if is_leading {
            lead_states.extend_from_slice(state_slice);
            lead_soft_intents.extend_from_slice(intent_slice);
            lead_legal_masks.push(legal_mask);
        } else {
            follow_states.extend_from_slice(state_slice);
            follow_soft_intents.extend_from_slice(intent_slice);
            follow_legal_masks.push(legal_mask);
        }
    }

    let lead_dataset = ExpertDataset {
        num_states: lead_legal_masks.len(),
        states: lead_states,
        soft_intents: lead_soft_intents,
        legal_masks: lead_legal_masks,
    };

    let follow_dataset = ExpertDataset {
        num_states: follow_legal_masks.len(),
        states: follow_states,
        soft_intents: follow_soft_intents,
        legal_masks: follow_legal_masks,
    };

    (lead_dataset, follow_dataset)
}

// ---------------------------------------------------------------------------
// Main Training Loop
// ---------------------------------------------------------------------------
pub fn train(config: Config, resume: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Pcg64::seed_from_u64(config.evaluation.seed);

    // Determine run directory: if resuming, use existing dir; otherwise create dated folder
    let run_dir = if resume {
        config.output.checkpoint_dir.clone()
    } else {
        let today = chrono::Local::now().format("%Y-%m-%d").to_string();
        let base = Path::new(&config.output.checkpoint_dir);
        let mut n = 1;
        let dir = loop {
            let candidate = base.join(format!("{}-{}", today, n));
            if !candidate.exists() {
                break candidate;
            }
            n += 1;
        };
        dir.to_string_lossy().to_string()
    };
    let mut config = config;
    config.output.checkpoint_dir = run_dir.clone();
    config.output.stats_file = format!("{}/training_stats.csv", run_dir);

    fs::create_dir_all(&run_dir)?;
    fs::create_dir_all(Path::new(&run_dir).join("genomes"))?;
    println!("Run directory: {}", run_dir);

    let state_checkpoint_path = Path::new(&run_dir).join("training_state.bin");

    let mut lead_pop: Population;
    let mut lead_hof: HallOfFame;
    let mut lead_map_elites: crate::map_elites::MapElitesArchive;
    let lead_registry: Mutex<InnovationRegistry>;
    let lead_tabu = TabuVetoList::new(1000);

    let mut follow_pop: Population;
    let mut follow_hof: HallOfFame;
    let mut follow_map_elites: crate::map_elites::MapElitesArchive;
    let follow_registry: Mutex<InnovationRegistry>;
    let follow_tabu = TabuVetoList::new(1000);

    let mut start_gen = 0;
    let mut current_phase = 0;

    if resume && state_checkpoint_path.exists() {
        println!(
            "Resuming training from checkpoint {:?}",
            state_checkpoint_path
        );
        let state = TrainingState::load_from_file(&state_checkpoint_path)?;
        start_gen = state.generation;
        current_phase = state.current_phase;

        let lead_state = state.lead;
        lead_pop = Population {
            config: config.clone(),
            genomes: lead_state.genomes,
            fitnesses: vec![0.0; config.population.pop_size],
            species_list: lead_state.species,
            generation: state.generation,
            next_species_id: lead_state.next_species_id,
            global_best_fitness: lead_state.global_best_fitness,
            global_best_genome: lead_state.global_best_genome,
            generations_since_improvement: lead_state.generations_since_improvement,
        };
        lead_hof = lead_state.hof;
        lead_map_elites = lead_state.map_elites;
        lead_registry = Mutex::new(InnovationRegistry::new(lead_state.next_innovation));

        let follow_state = state.follow;
        follow_pop = Population {
            config: config.clone(),
            genomes: follow_state.genomes,
            fitnesses: vec![0.0; config.population.pop_size],
            species_list: follow_state.species,
            generation: state.generation,
            next_species_id: follow_state.next_species_id,
            global_best_fitness: follow_state.global_best_fitness,
            global_best_genome: follow_state.global_best_genome,
            generations_since_improvement: follow_state.generations_since_improvement,
        };
        follow_hof = follow_state.hof;
        follow_map_elites = follow_state.map_elites;
        follow_registry = Mutex::new(InnovationRegistry::new(follow_state.next_innovation));
    } else {
        lead_registry = Mutex::new(InnovationRegistry::new(0));
        lead_pop = Population::new(config.clone(), &mut rng, &lead_registry);
        lead_hof = HallOfFame::new(config.hall_of_fame.hof_size);
        lead_map_elites = crate::map_elites::MapElitesArchive::new();

        follow_registry = Mutex::new(InnovationRegistry::new(0));
        follow_pop = Population::new(config.clone(), &mut rng, &follow_registry);
        follow_hof = HallOfFame::new(config.hall_of_fame.hof_size);
        follow_map_elites = crate::map_elites::MapElitesArchive::new();
    }

    // Load dataset for Phase 0
    let dataset = load_expert_dataset(&config.curriculum.phase0_dataset)?;
    let (lead_dataset, follow_dataset) = split_dataset(&dataset);
    println!(
        "  >>> Split dataset: {} Lead states, {} Follow states.",
        lead_dataset.num_states, follow_dataset.num_states
    );

    // CSV Stats
    let stats_path = Path::new(&config.output.stats_file);
    let mut csv_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(!resume || !stats_path.exists())
        .open(stats_path)?;

    if !resume || csv_file.metadata()?.len() == 0 {
        writeln!(
            csv_file,
            "generation,phase,lead_best_fitness,lead_avg_fitness,follow_best_fitness,follow_avg_fitness,lead_n_species,follow_n_species,lead_n_connections_best,follow_n_connections_best,elapsed_sec"
        )?;
    }

    println!(
        "{:>4} {:>2} {:>8} {:>8} {:>8} {:>8} {:>6} {:>6} {:>6} {:>6} {:>8}",
        "Gen",
        "Ph",
        "L-Best",
        "L-Avg",
        "F-Best",
        "F-Avg",
        "L-Spec",
        "F-Spec",
        "L-Conn",
        "F-Conn",
        "Time"
    );
    println!("{}", "-".repeat(95));

    for gen in start_gen..config.population.generations {
        let t0 = Instant::now();

        // Phase selection
        let new_phase = if gen < config.curriculum.phase0_gens {
            0
        } else {
            1
        };
        if new_phase != current_phase {
            println!(
                "  >>> Phase transition: {} -> {} at gen {}",
                current_phase, new_phase, gen
            );
            current_phase = new_phase;
            if current_phase == 1 {
                // Pre-populate Lead HOF
                println!("  >>> Pre-populating Phase 1 HOF with top unique Phase 0 genomes...");
                lead_hof.clear();
                let mut lead_indices: Vec<usize> = (0..lead_pop.genomes.len()).collect();
                lead_indices.sort_by(|&a, &b| lead_pop.fitnesses[b].partial_cmp(&lead_pop.fitnesses[a]).unwrap());

                let mut unique_lead = Vec::new();
                let c1 = config.species.c_excess;
                let c2 = config.species.c_disjoint;
                let c3 = config.species.c_mismatch;

                for &idx in &lead_indices {
                    let candidate = &lead_pop.genomes[idx];
                    let is_unique = unique_lead.iter().all(|g| {
                        crate::species::compatibility_distance(candidate, g, c1, c2, c3) > 0.05
                    });
                    if is_unique {
                        unique_lead.push(candidate.copy());
                        if unique_lead.len() >= 5 {
                            break;
                        }
                    }
                }
                for &idx in &lead_indices {
                    if unique_lead.len() >= 5 {
                        break;
                    }
                    let already_added = unique_lead.iter().any(|g| {
                        crate::species::compatibility_distance(&lead_pop.genomes[idx], g, c1, c2, c3) < 1e-9
                    });
                    if !already_added {
                        unique_lead.push(lead_pop.genomes[idx].copy());
                    }
                }

                // Pre-populate Follow HOF
                follow_hof.clear();
                let mut follow_indices: Vec<usize> = (0..follow_pop.genomes.len()).collect();
                follow_indices.sort_by(|&a, &b| follow_pop.fitnesses[b].partial_cmp(&follow_pop.fitnesses[a]).unwrap());

                let mut unique_follow = Vec::new();
                for &idx in &follow_indices {
                    let candidate = &follow_pop.genomes[idx];
                    let is_unique = unique_follow.iter().all(|g| {
                        crate::species::compatibility_distance(candidate, g, c1, c2, c3) > 0.05
                    });
                    if is_unique {
                        unique_follow.push(candidate.copy());
                        if unique_follow.len() >= 5 {
                            break;
                        }
                    }
                }
                for &idx in &follow_indices {
                    if unique_follow.len() >= 5 {
                        break;
                    }
                    let already_added = unique_follow.iter().any(|g| {
                        crate::species::compatibility_distance(&follow_pop.genomes[idx], g, c1, c2, c3) < 1e-9
                    });
                    if !already_added {
                        unique_follow.push(follow_pop.genomes[idx].copy());
                    }
                }

                // Evaluate pre-populated Lead and Follow genomes
                let deals = generate_deals_rust(
                    gen,
                    config.evaluation.n_deals,
                    config.evaluation.seed * 1000,
                );

                let lead_networks: Vec<crate::wann_network::RustWannNetwork> =
                    unique_lead.iter().map(|g| g.to_rust_wann()).collect();
                let follow_networks: Vec<crate::wann_network::RustWannNetwork> =
                    unique_follow.iter().map(|g| g.to_rust_wann()).collect();

                let dummy_lead_ref = lead_networks[0].clone();
                let dummy_follow_ref = follow_networks[0].clone();

                let (lead_fitnesses, _, _) = evaluate_phase1(
                    &lead_networks,
                    true,
                    &deals,
                    &dummy_follow_ref,
                    &[],
                    &[],
                    2,   // partner = HeuristicBot
                    2,   // opp1 = HeuristicBot
                    2,   // opp2 = HeuristicBot
                    2,   // baseline = HeuristicBot
                    &config.evaluation.sweep_weights,
                    config.evaluation.seed + gen as u64 * 1000,
                );

                for (genome, fitness) in unique_lead.into_iter().zip(lead_fitnesses.into_iter()) {
                    lead_hof.add(&genome, fitness, gen);
                }

                let (follow_fitnesses, _, _) = evaluate_phase1(
                    &follow_networks,
                    false,
                    &deals,
                    &dummy_lead_ref,
                    &[],
                    &[],
                    2,   // partner = HeuristicBot
                    2,   // opp1 = HeuristicBot
                    2,   // opp2 = HeuristicBot
                    2,   // baseline = HeuristicBot
                    &config.evaluation.sweep_weights,
                    config.evaluation.seed + gen as u64 * 1000,
                );

                for (genome, fitness) in unique_follow.into_iter().zip(follow_fitnesses.into_iter()) {
                    follow_hof.add(&genome, fitness, gen);
                }

                println!(
                    "  >>> Pre-populated HOF with {} Lead and {} Follow unique genomes from Phase 0",
                    lead_hof.entries.len(),
                    follow_hof.entries.len()
                );

                lead_pop.global_best_fitness = f64::NEG_INFINITY;
                lead_pop.global_best_genome = None;
                lead_pop.generations_since_improvement = 0;

                follow_pop.global_best_fitness = f64::NEG_INFINITY;
                follow_pop.global_best_genome = None;
                follow_pop.generations_since_improvement = 0;
            }
        }

        let lead_rust_genomes: Vec<crate::wann_network::RustWannNetwork> =
            lead_pop.genomes.par_iter().map(|g| g.to_rust_wann()).collect();
        let follow_rust_genomes: Vec<crate::wann_network::RustWannNetwork> =
            follow_pop.genomes.par_iter().map(|g| g.to_rust_wann()).collect();

        let lead_champion = lead_pop.global_best_genome.as_ref().cloned()
            .or_else(|| lead_hof.best().map(|e| e.genome.copy()))
            .unwrap_or_else(|| lead_pop.genomes[0].copy())
            .to_rust_wann();

        let follow_champion = follow_pop.global_best_genome.as_ref().cloned()
            .or_else(|| follow_hof.best().map(|e| e.genome.copy()))
            .unwrap_or_else(|| follow_pop.genomes[0].copy())
            .to_rust_wann();

        let (lead_fitnesses, _lead_deltas, lead_behaviors) = if current_phase == 0 {
            let accs = evaluate_phase0(&lead_rust_genomes, &lead_dataset, &config.evaluation.sweep_weights);
            (accs.clone(), accs, Vec::new())
        } else {
            run_phase1_generation(
                &lead_rust_genomes,
                true,
                &lead_hof,
                &follow_hof,
                &lead_map_elites,
                &follow_map_elites,
                &follow_champion,
                &config,
                gen,
                &mut rng,
            )
        };

        let (follow_fitnesses, _follow_deltas, follow_behaviors) = if current_phase == 0 {
            let accs = evaluate_phase0(&follow_rust_genomes, &follow_dataset, &config.evaluation.sweep_weights);
            (accs.clone(), accs, Vec::new())
        } else {
            run_phase1_generation(
                &follow_rust_genomes,
                false,
                &lead_hof,
                &follow_hof,
                &lead_map_elites,
                &follow_map_elites,
                &lead_champion,
                &config,
                gen,
                &mut rng,
            )
        };

        lead_pop.tell_fitnesses(&lead_fitnesses);
        follow_pop.tell_fitnesses(&follow_fitnesses);

        // Add each genome's behavior to MAP-Elites if in Phase 1
        if current_phase == 1 {
            for (i, behavior) in lead_behaviors.iter().enumerate() {
                let intent_pref =
                    (behavior.intent_counts[1] as f64) / (behavior.total_actions.max(1) as f64);
                let aggression = ((behavior.total_lead_points as f64)
                    / (behavior.count_leads.max(1) as f64 * 10.0))
                    .clamp(0.0, 1.0);
                lead_map_elites.add(&lead_pop.genomes[i], lead_fitnesses[i], gen, intent_pref, aggression);
            }
            for (i, behavior) in follow_behaviors.iter().enumerate() {
                let intent_pref =
                    (behavior.intent_counts[1] as f64) / (behavior.total_actions.max(1) as f64);
                let aggression = ((behavior.total_lead_points as f64)
                    / (behavior.count_leads.max(1) as f64 * 10.0))
                    .clamp(0.0, 1.0);
                follow_map_elites.add(&follow_pop.genomes[i], follow_fitnesses[i], gen, intent_pref, aggression);
            }
        }

        // Find best candidate
        let mut lead_best_idx = 0;
        let mut lead_max_fit = lead_fitnesses[0];
        for (i, &f) in lead_fitnesses.iter().enumerate() {
            if f > lead_max_fit {
                lead_max_fit = f;
                lead_best_idx = i;
            }
        }
        let lead_best_fit = lead_fitnesses[lead_best_idx];
        let lead_avg_fit = lead_fitnesses.iter().sum::<f64>() / lead_fitnesses.len() as f64;

        let mut follow_best_idx = 0;
        let mut follow_max_fit = follow_fitnesses[0];
        for (i, &f) in follow_fitnesses.iter().enumerate() {
            if f > follow_max_fit {
                follow_max_fit = f;
                follow_best_idx = i;
            }
        }
        let follow_best_fit = follow_fitnesses[follow_best_idx];
        let follow_avg_fit = follow_fitnesses.iter().sum::<f64>() / follow_fitnesses.len() as f64;

        // Add to Hall of Fame
        lead_hof.add(&lead_pop.genomes[lead_best_idx], lead_best_fit, gen);
        follow_hof.add(&follow_pop.genomes[follow_best_idx], follow_best_fit, gen);

        let elapsed = t0.elapsed().as_secs_f64();
        let lead_species = lead_pop
            .species_list
            .iter()
            .filter(|s| !s.members.is_empty())
            .count();
        let follow_species = follow_pop
            .species_list
            .iter()
            .filter(|s| !s.members.is_empty())
            .count();

        writeln!(
            csv_file,
            "{},{},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{:.2}",
            gen,
            current_phase,
            lead_best_fit,
            lead_avg_fit,
            follow_best_fit,
            follow_avg_fit,
            lead_species,
            follow_species,
            lead_pop.genomes[lead_best_idx].num_enabled(),
            follow_pop.genomes[follow_best_idx].num_enabled(),
            elapsed
        )?;
        csv_file.flush()?;

        println!(
            "{:4} {:2} {:8.4} {:8.4} {:8.4} {:8.4} {:6} {:6} {:6} {:6} {:7.1}s",
            gen,
            current_phase,
            lead_best_fit,
            lead_avg_fit,
            follow_best_fit,
            follow_avg_fit,
            lead_species,
            follow_species,
            lead_pop.genomes[lead_best_idx].num_enabled(),
            follow_pop.genomes[follow_best_idx].num_enabled(),
            elapsed
        );

        // Breed next generation
        if current_phase == 1 && lead_pop.generations_since_improvement > 20 {
            if let Some(gb_clone) = lead_pop.global_best_genome.as_ref().map(|g| g.copy()) {
                println!(
                    "  >>> Lead: No improvement for {} generations. Reseeding species with mutations of the global best champion...",
                    lead_pop.generations_since_improvement
                );
                lead_pop.reseed_from_champion(&gb_clone, 0.10, &lead_tabu, &lead_registry, &mut rng);
                lead_pop.generations_since_improvement = 0;
            }
        }
        if current_phase == 1 && follow_pop.generations_since_improvement > 20 {
            if let Some(gb_clone) = follow_pop.global_best_genome.as_ref().map(|g| g.copy()) {
                println!(
                    "  >>> Follow: No improvement for {} generations. Reseeding species with mutations of the global best champion...",
                    follow_pop.generations_since_improvement
                );
                follow_pop.reseed_from_champion(&gb_clone, 0.10, &follow_tabu, &follow_registry, &mut rng);
                follow_pop.generations_since_improvement = 0;
            }
        }

        let lead_eval_data = if current_phase == 0 {
            Some((&lead_dataset, config.evaluation.sweep_weights.as_slice()))
        } else {
            None
        };
        let follow_eval_data = if current_phase == 0 {
            Some((&follow_dataset, config.evaluation.sweep_weights.as_slice()))
        } else {
            None
        };

        lead_pop.speciate_and_evolve(
            current_phase,
            config.curriculum.bulking_gens,
            &lead_tabu,
            lead_eval_data,
            &mut rng,
            &lead_registry,
        );

        follow_pop.speciate_and_evolve(
            current_phase,
            config.curriculum.bulking_gens,
            &follow_tabu,
            follow_eval_data,
            &mut rng,
            &follow_registry,
        );

        // Checkpointing
        if (gen + 1) % 10 == 0 || gen == config.population.generations - 1 {
            // Save HOF (Joint)
            let hof_path = Path::new(&config.output.checkpoint_dir)
                .join("genomes")
                .join(format!("hof_gen{:04}.json", gen + 1));
            let json_lead_hof = JsonHallOfFame::from_hof(&lead_hof);
            let json_follow_hof = JsonHallOfFame::from_hof(&follow_hof);
            let joint_hof = JsonHallOfFameJoint {
                lead: json_lead_hof,
                follow: json_follow_hof,
            };
            let hof_file = fs::File::create(hof_path)?;
            serde_json::to_writer_pretty(hof_file, &joint_hof)?;

            // Save global best genome
            let lead_best = lead_pop.global_best_genome.as_ref().map(JsonGenome::from_genome);
            let follow_best = follow_pop.global_best_genome.as_ref().map(JsonGenome::from_genome);
            let joint_best = JsonGenomeJoint {
                lead: lead_best,
                follow: follow_best,
            };
            let best_path = Path::new(&config.output.checkpoint_dir)
                .join("genomes")
                .join(format!("best_genome_gen{:04}.json", gen + 1));
            let best_file = fs::File::create(best_path)?;
            serde_json::to_writer_pretty(best_file, &joint_best)?;

            // Save generation best genome
            let lead_gen_best = Some(JsonGenome::from_genome(&lead_pop.genomes[lead_best_idx]));
            let follow_gen_best = Some(JsonGenome::from_genome(&follow_pop.genomes[follow_best_idx]));
            let joint_gen_best = JsonGenomeJoint {
                lead: lead_gen_best,
                follow: follow_gen_best,
            };
            let gen_best_path = Path::new(&config.output.checkpoint_dir)
                .join("genomes")
                .join(format!("gen_best_genome_gen{:04}.json", gen + 1));
            let gen_best_file = fs::File::create(gen_best_path)?;
            serde_json::to_writer_pretty(gen_best_file, &joint_gen_best)?;

            // Save stateful training checkpoint
            let state = TrainingState {
                generation: gen + 1,
                current_phase,
                lead: BrainTrainingState {
                    genomes: lead_pop.genomes.clone(),
                    species: lead_pop.species_list.clone(),
                    hof: lead_hof.clone(),
                    map_elites: lead_map_elites.clone(),
                    next_species_id: lead_pop.next_species_id,
                    global_best_fitness: lead_pop.global_best_fitness,
                    global_best_genome: lead_pop.global_best_genome.clone(),
                    generations_since_improvement: lead_pop.generations_since_improvement,
                    next_innovation: lead_registry.lock().unwrap().next_innovation,
                },
                follow: BrainTrainingState {
                    genomes: follow_pop.genomes.clone(),
                    species: follow_pop.species_list.clone(),
                    hof: follow_hof.clone(),
                    map_elites: follow_map_elites.clone(),
                    next_species_id: follow_pop.next_species_id,
                    global_best_fitness: follow_pop.global_best_fitness,
                    global_best_genome: follow_pop.global_best_genome.clone(),
                    generations_since_improvement: follow_pop.generations_since_improvement,
                    next_innovation: follow_registry.lock().unwrap().next_innovation,
                },
            };
            state.save_to_file(&state_checkpoint_path)?;
        }
    }

    // Save final files
    let hof_path = Path::new(&config.output.checkpoint_dir)
        .join("genomes")
        .join("hof_final.json");
    let json_lead_hof = JsonHallOfFame::from_hof(&lead_hof);
    let json_follow_hof = JsonHallOfFame::from_hof(&follow_hof);
    let joint_hof = JsonHallOfFameJoint {
        lead: json_lead_hof,
        follow: json_follow_hof,
    };
    let hof_file = fs::File::create(hof_path)?;
    serde_json::to_writer_pretty(hof_file, &joint_hof)?;

    let lead_best = lead_pop.global_best_genome.as_ref().map(JsonGenome::from_genome);
    let follow_best = follow_pop.global_best_genome.as_ref().map(JsonGenome::from_genome);
    let joint_best = JsonGenomeJoint {
        lead: lead_best,
        follow: follow_best,
    };
    let best_path = Path::new(&config.output.checkpoint_dir)
        .join("genomes")
        .join("best_genome_final.json");
    let best_file = fs::File::create(best_path)?;
    serde_json::to_writer_pretty(best_file, &joint_best)?;

    Ok(())
}
