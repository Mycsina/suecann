use crate::checkpoint::TrainingState;
use crate::config::Config;
use crate::dataset::{load_expert_dataset, ExpertDataset};
use crate::genome::{JsonGenome, FIRST_HIDDEN_ID, INPUT_COUNT, OUTPUT_COUNT, OUTPUT_START};
use crate::hall_of_fame::{HallOfFame, JsonHallOfFame};
use crate::mutations::InnovationRegistry;
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
) -> Vec<sueca_solver::evaluator::EvaluatorDeal> {
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
        deals.push(sueca_solver::evaluator::EvaluatorDeal {
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
    genomes: &[sueca_solver::wann::RustWannNetwork],
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
            let mut correct = 0;

            for idx in 0..dataset.num_states {
                let mut inputs = [0.0f64; INPUT_COUNT];
                for i in 0..INPUT_COUNT {
                    inputs[i] = dataset.states[idx * INPUT_COUNT + i];
                }
                let target_intent = dataset.intents[idx] as usize;
                let mask = dataset.legal_masks[idx];

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
                    let is_legal = (mask & (1 << i)) != 0;
                    let val = if i == 3 {
                        total_outputs[i] - 0.25 * (sweep_weights.len() as f64)
                    } else {
                        total_outputs[i]
                    };
                    if is_legal && val > max_val {
                        max_val = val;
                        best_intent = i;
                    }
                }

                if best_intent == target_intent {
                    correct += 1;
                }
            }

            correct as f64 / dataset.num_states as f64
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Phase 1: Self-play with HOF opponents
// ---------------------------------------------------------------------------
pub fn evaluate_phase1(
    genomes: &[sueca_solver::wann::RustWannNetwork],
    deals: &[sueca_solver::evaluator::EvaluatorDeal],
    hof_networks: &[sueca_solver::wann::RustWannNetwork],
    partner_bot_type: i32,
    opp1_bot_type: i32,
    opp2_bot_type: i32,
    baseline_bot_type: i32,
    sweep_weights: &[f64],
    base_seed: u64,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<sueca_solver::evaluator::WannBehavior>,
) {
    let max_nodes = genomes
        .iter()
        .map(|g| g.num_nodes)
        .chain(hof_networks.iter().map(|g| g.num_nodes))
        .max()
        .unwrap_or(FIRST_HIDDEN_ID);

    let results: Vec<(f64, sueca_solver::evaluator::WannBehavior)> = genomes
        .into_par_iter()
        .enumerate()
        .map(|(i, candidate)| {
            let mut scratchpad = vec![0.0f64; max_nodes];
            sueca_solver::evaluator::evaluate_genome_delta(
                candidate,
                baseline_bot_type,
                partner_bot_type,
                opp1_bot_type,
                opp2_bot_type,
                hof_networks,
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
fn run_phase0_generation(
    genomes: &[sueca_solver::wann::RustWannNetwork],
    dataset: &ExpertDataset,
    sweep_weights: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let accs = evaluate_phase0(genomes, dataset, sweep_weights);
    (accs.clone(), accs)
}

fn run_phase1_generation(
    genomes: &[sueca_solver::wann::RustWannNetwork],
    hof: &HallOfFame,
    map_elites: &crate::map_elites::MapElitesArchive,
    config: &Config,
    gen: usize,
    rng: &mut Pcg64,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<sueca_solver::evaluator::WannBehavior>,
) {
    let deals = generate_deals_rust(
        gen,
        config.evaluation.n_deals,
        config.evaluation.seed * 1000,
    );

    let mut hof_genomes = Vec::new();
    let partner_bot_type: i32;
    let opp1_bot_type: i32;
    let opp2_bot_type: i32;

    let sample_seat_bot = |rng: &mut Pcg64,
                           h: &HallOfFame,
                           me: &crate::map_elites::MapElitesArchive,
                           hg: &mut Vec<sueca_solver::wann::RustWannNetwork>|
     -> i32 {
        if rng.gen_bool(0.5) {
            let use_map_elites = rng.gen_bool(0.5);
            let sampled = if use_map_elites {
                me.sample_random(rng)
            } else {
                None
            };
            let genome = sampled.or_else(|| h.sample(rng, 1).first().cloned());

            if let Some(g) = genome {
                let bot_type = 2 + hg.len() as i32;
                hg.push(g.to_rust_wann());
                bot_type
            } else {
                1
            }
        } else {
            1
        }
    };

    let mut sample_rng = rng.clone();
    partner_bot_type = sample_seat_bot(&mut sample_rng, hof, map_elites, &mut hof_genomes);
    opp1_bot_type = sample_seat_bot(&mut sample_rng, hof, map_elites, &mut hof_genomes);
    opp2_bot_type = sample_seat_bot(&mut sample_rng, hof, map_elites, &mut hof_genomes);
    *rng = sample_rng;

    evaluate_phase1(
        genomes,
        &deals,
        &hof_genomes,
        partner_bot_type,
        opp1_bot_type,
        opp2_bot_type,
        1, // baseline = HeuristicBot
        &config.evaluation.sweep_weights,
        config.evaluation.seed + gen as u64 * 1000,
    )
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

    let mut pop: Population;
    let mut hof: HallOfFame;
    let mut map_elites: crate::map_elites::MapElitesArchive;
    let registry: Mutex<InnovationRegistry>;
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

        pop = Population {
            config: config.clone(),
            genomes: state.genomes,
            fitnesses: vec![0.0; config.population.pop_size],
            species_list: state.species,
            generation: state.generation,
            next_species_id: state.next_species_id,
            global_best_fitness: state.global_best_fitness,
            global_best_genome: state.global_best_genome,
            generations_since_improvement: state.generations_since_improvement,
        };
        hof = state.hof;
        map_elites = state.map_elites;
        registry = Mutex::new(InnovationRegistry::new(state.next_innovation));
    } else {
        registry = Mutex::new(InnovationRegistry::new(0));
        pop = Population::new(config.clone(), &mut rng, &registry);
        hof = HallOfFame::new(config.hall_of_fame.hof_size);
        map_elites = crate::map_elites::MapElitesArchive::new();
    }

    // Load dataset for Phase 0
    let dataset = load_expert_dataset(&config.curriculum.phase0_dataset)?;

    // CSV Stats
    let stats_path = Path::new(&config.output.stats_file);
    let mut csv_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(!resume || !stats_path.exists())
        .append(resume && stats_path.exists())
        .open(stats_path)?;

    if !resume || csv_file.metadata()?.len() == 0 {
        writeln!(
            csv_file,
            "generation,phase,best_fitness,avg_fitness,median_fitness,best_delta,median_delta,global_best_fitness,n_species,n_connections_best,n_hidden_best,elapsed_sec"
        )?;
    }

    println!(
        "{:>4} {:>2} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6} {:>6} {:>6} {:>8}",
        "Gen",
        "Ph",
        "Best",
        "Avg",
        "Med",
        "D-Best",
        "D-Med",
        "G-Best",
        "Spec",
        "Conn",
        "Hidd",
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
                println!("  >>> Pre-populating Phase 1 HOF with top unique Phase 0 genomes...");
                hof.clear();

                // Get sorted indices of genomes by Phase 0 accuracy
                let mut pop_indices: Vec<usize> = (0..pop.genomes.len()).collect();
                pop_indices
                    .sort_by(|&a, &b| pop.fitnesses[b].partial_cmp(&pop.fitnesses[a]).unwrap());

                let mut unique_genomes = Vec::new();
                let c1 = config.species.c_excess;
                let c2 = config.species.c_disjoint;
                let c3 = config.species.c_mismatch;

                for &idx in &pop_indices {
                    let candidate = &pop.genomes[idx];
                    let is_unique = unique_genomes.iter().all(|g| {
                        crate::species::compatibility_distance(candidate, g, c1, c2, c3) > 0.05
                    });
                    if is_unique {
                        unique_genomes.push(candidate.copy());
                        if unique_genomes.len() >= 5 {
                            break;
                        }
                    }
                }

                // If not enough unique, fill with non-identical
                for &idx in &pop_indices {
                    if unique_genomes.len() >= 5 {
                        break;
                    }
                    let already_added = unique_genomes.iter().any(|g| {
                        crate::species::compatibility_distance(&pop.genomes[idx], g, c1, c2, c3)
                            < 1e-9
                    });
                    if !already_added {
                        unique_genomes.push(pop.genomes[idx].copy());
                    }
                }

                // Evaluate them with HeuristicBot matches to register baseline fitness
                let deals = generate_deals_rust(
                    gen,
                    config.evaluation.n_deals,
                    config.evaluation.seed * 1000,
                );

                let unique_networks: Vec<sueca_solver::wann::RustWannNetwork> =
                    unique_genomes.iter().map(|g| g.to_rust_wann()).collect();

                let (fitnesses, _, _) = evaluate_phase1(
                    &unique_networks,
                    &deals,
                    &[], // no HOF opponents during pre-population
                    1,   // partner = HeuristicBot
                    1,   // opp1 = HeuristicBot
                    1,   // opp2 = HeuristicBot
                    1,   // baseline = HeuristicBot
                    &config.evaluation.sweep_weights,
                    config.evaluation.seed + gen as u64 * 1000,
                );

                for (genome, fitness) in unique_genomes.into_iter().zip(fitnesses.into_iter()) {
                    hof.add(&genome, fitness, gen);
                }

                println!(
                    "  >>> Pre-populated HOF with {} unique genomes from Phase 0",
                    hof.entries.len()
                );

                pop.global_best_fitness = f64::NEG_INFINITY;
                pop.global_best_genome = None;
                pop.generations_since_improvement = 0;
            }
        }

        let rust_genomes: Vec<sueca_solver::wann::RustWannNetwork> =
            pop.genomes.par_iter().map(|g| g.to_rust_wann()).collect();

        let (fitnesses, deltas, behaviors) = if current_phase == 0 {
            let (fit, dt) =
                run_phase0_generation(&rust_genomes, &dataset, &config.evaluation.sweep_weights);
            (fit, dt, Vec::new())
        } else {
            run_phase1_generation(&rust_genomes, &hof, &map_elites, &config, gen, &mut rng)
        };

        pop.tell_fitnesses(&fitnesses);

        // Add each genome's behavior to MAP-Elites if in Phase 1
        if current_phase == 1 {
            for (i, behavior) in behaviors.iter().enumerate() {
                let intent_pref =
                    (behavior.intent_counts[1] as f64) / (behavior.total_actions.max(1) as f64);
                let aggression = ((behavior.total_lead_points as f64)
                    / (behavior.count_leads.max(1) as f64 * 10.0))
                    .clamp(0.0, 1.0);
                map_elites.add(&pop.genomes[i], fitnesses[i], gen, intent_pref, aggression);
            }
        }

        // Find best candidate
        let mut best_idx = 0;
        let mut max_fit = fitnesses[0];
        for (i, &f) in fitnesses.iter().enumerate() {
            if f > max_fit {
                max_fit = f;
                best_idx = i;
            }
        }

        let best_fit = fitnesses[best_idx];
        let avg_fit = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
        let mut sorted_fitnesses = fitnesses.clone();
        sorted_fitnesses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_fit = sorted_fitnesses[sorted_fitnesses.len() / 2];

        let best_delta = deltas[best_idx];
        let mut sorted_deltas = deltas.clone();
        sorted_deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_delta = sorted_deltas[sorted_deltas.len() / 2];

        // Add to Hall of Fame
        hof.add(&pop.genomes[best_idx], best_fit, gen);

        let elapsed = t0.elapsed().as_secs_f64();
        let n_species = pop
            .species_list
            .iter()
            .filter(|s| !s.members.is_empty())
            .count();
        let best_genome = &pop.genomes[best_idx];
        let n_conns = best_genome.num_enabled();
        let n_hidden = best_genome.hidden_ids().len();

        writeln!(
            csv_file,
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{:.2}",
            gen,
            current_phase,
            best_fit,
            avg_fit,
            median_fit,
            best_delta,
            median_delta,
            pop.global_best_fitness,
            n_species,
            n_conns,
            n_hidden,
            elapsed
        )?;
        csv_file.flush()?;

        println!(
            "{:4} {:2} {:8.4} {:8.4} {:8.4} {:8.4} {:8.4} {:8.4} {:6} {:6} {:6} {:7.1}s",
            gen,
            current_phase,
            best_fit,
            avg_fit,
            median_fit,
            best_delta,
            median_delta,
            pop.global_best_fitness,
            n_species,
            n_conns,
            n_hidden,
            elapsed
        );

        // Breed next generation
        if current_phase == 1 && pop.generations_since_improvement > 20 {
            if let Some(gb_clone) = pop.global_best_genome.as_ref().map(|g| g.copy()) {
                println!(
                    "  >>> No improvement for {} generations. Reseeding species with mutations of the global best champion...",
                    pop.generations_since_improvement
                );
                pop.reseed_from_champion(&gb_clone, 0.10, &registry, &mut rng);
                pop.generations_since_improvement = 0;
            }
        }

        pop.speciate_and_evolve(
            current_phase,
            config.curriculum.bulking_gens,
            &mut rng,
            &registry,
        );

        // Checkpointing
        if (gen + 1) % 10 == 0 || gen == config.population.generations - 1 {
            // Save HOF
            let hof_path = Path::new(&config.output.checkpoint_dir)
                .join("genomes")
                .join(format!("hof_gen{:04}.json", gen + 1));
            let json_hof = JsonHallOfFame::from_hof(&hof);
            let hof_file = fs::File::create(hof_path)?;
            serde_json::to_writer_pretty(hof_file, &json_hof)?;

            // Save global best genome
            if let Some(ref gb) = pop.global_best_genome {
                let best_path = Path::new(&config.output.checkpoint_dir)
                    .join("genomes")
                    .join(format!("best_genome_gen{:04}.json", gen + 1));
                let json_gb = JsonGenome::from_genome(gb);
                let gb_file = fs::File::create(best_path)?;
                serde_json::to_writer_pretty(gb_file, &json_gb)?;
            }

            // Save generation best genome
            let gen_best_path = Path::new(&config.output.checkpoint_dir)
                .join("genomes")
                .join(format!("gen_best_genome_gen{:04}.json", gen + 1));
            let json_gen_best = JsonGenome::from_genome(&pop.genomes[best_idx]);
            let gen_best_file = fs::File::create(gen_best_path)?;
            serde_json::to_writer_pretty(gen_best_file, &json_gen_best)?;

            // Save stateful training checkpoint
            let state = TrainingState {
                generation: gen + 1,
                next_species_id: pop.next_species_id,
                global_best_fitness: pop.global_best_fitness,
                global_best_genome: pop.global_best_genome.clone(),
                genomes: pop.genomes.clone(),
                species: pop.species_list.clone(),
                hof: hof.clone(),
                next_innovation: registry.lock().unwrap().next_innovation,
                current_phase,
                generations_since_improvement: pop.generations_since_improvement,
                map_elites: map_elites.clone(),
            };
            state.save_to_file(&state_checkpoint_path)?;
        }
    }

    // Save final files
    let hof_path = Path::new(&config.output.checkpoint_dir).join("genomes").join("hof_final.json");
    let json_hof = JsonHallOfFame::from_hof(&hof);
    let hof_file = fs::File::create(hof_path)?;
    serde_json::to_writer_pretty(hof_file, &json_hof)?;

    if let Some(ref gb) = pop.global_best_genome {
        let best_path = Path::new(&config.output.checkpoint_dir).join("genomes").join("best_genome_final.json");
        let json_gb = JsonGenome::from_genome(gb);
        let gb_file = fs::File::create(best_path)?;
        serde_json::to_writer_pretty(gb_file, &json_gb)?;
    }

    Ok(())
}
