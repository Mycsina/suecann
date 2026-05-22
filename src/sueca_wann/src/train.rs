use crate::checkpoint::TrainingState;
use crate::config::Config;
use crate::dataset::{load_expert_dataset, ExpertDataset};
use crate::genome::JsonGenome;
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
use std::time::Instant;

pub fn oracle_tax_penalty(generation: usize, phase0_gens: usize, curriculum_gens: usize) -> f64 {
    if curriculum_gens == 0 {
        return -3.0;
    }
    let active_gens = generation.saturating_sub(phase0_gens);
    let frac = (active_gens as f64 / curriculum_gens as f64).min(1.0);
    -0.25 + frac * (-3.0 - (-0.25))
}

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
    let max_nodes = genomes.iter().map(|g| g.num_nodes).max().unwrap_or(27);

    genomes
        .into_par_iter()
        .map(|candidate| {
            let mut scratchpad = vec![0.0f64; max_nodes];
            let mut correct = 0;

            for idx in 0..dataset.num_states {
                let mut inputs = [0.0f64; 21];
                for i in 0..21 {
                    inputs[i] = dataset.states[idx * 21 + i];
                }
                let target_intent = dataset.intents[idx] as usize;
                let mask = dataset.legal_masks[idx];

                let mut total_outputs = [0.0f64; 5];
                for &w in sweep_weights {
                    candidate.forward(&inputs, w, &mut scratchpad);
                    for i in 0..5 {
                        total_outputs[i] += scratchpad[22 + i];
                    }
                }

                let mut best_intent = 0;
                let mut max_val = f64::NEG_INFINITY;
                for i in 0..5 {
                    let is_legal = (mask & (1 << i)) != 0;
                    if is_legal && total_outputs[i] > max_val {
                        max_val = total_outputs[i];
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
    generation: usize,
    curriculum_gens: usize,
    phase0_gens: usize,
) -> (Vec<f64>, Vec<f64>, usize) {
    let max_nodes = genomes
        .iter()
        .map(|g| g.num_nodes)
        .chain(hof_networks.iter().map(|g| g.num_nodes))
        .max()
        .unwrap_or(27);

    let results: Vec<(f64, usize)> = genomes
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

    let tax_per = oracle_tax_penalty(generation, phase0_gens, curriculum_gens);
    let n_games = deals.len() * 4;

    let mut fitnesses = Vec::with_capacity(genomes.len());
    let mut deltas = Vec::with_capacity(genomes.len());
    let mut total_illegal = 0;

    for (delta, illegal_count) in results {
        total_illegal += illegal_count;
        let illegal_rate = if n_games > 0 {
            illegal_count as f64 / n_games as f64
        } else {
            0.0
        };
        let fit = delta + tax_per * illegal_rate;
        fitnesses.push(fit);
        deltas.push(delta);
    }

    (fitnesses, deltas, total_illegal)
}

// ---------------------------------------------------------------------------
// Phase 0 → 1 HOF transfer: re-evaluate HOF genomes under Phase 1 fitness
// ---------------------------------------------------------------------------
fn transfer_hof_to_phase1(hof: &mut HallOfFame, config: &Config, gen: usize, _rng: &mut Pcg64) {
    if hof.entries.is_empty() {
        return;
    }

    let deals = generate_deals_rust(
        gen,
        config.evaluation.n_deals,
        config.evaluation.seed * 1000,
    );

    let hof_networks: Vec<sueca_solver::wann::RustWannNetwork> = hof
        .entries
        .iter()
        .map(|e| e.genome.to_rust_wann())
        .collect();

    let (fitnesses, _, _) = evaluate_phase1(
        &hof_networks,
        &deals,
        &[], // no HOF opponents during transfer
        1,   // partner = HeuristicBot
        1,   // opp1 = HeuristicBot
        1,   // opp2 = HeuristicBot
        1,   // baseline = HeuristicBot
        &config.evaluation.sweep_weights,
        config.evaluation.seed + gen as u64 * 1000,
        gen,
        config.evaluation.curriculum_gens,
        config.curriculum.phase0_gens,
    );

    // Update HOF entries with Phase 1 fitness
    for (entry, &fitness) in hof.entries.iter_mut().zip(fitnesses.iter()) {
        entry.fitness = fitness;
    }

    // Re-sort by new fitness
    hof.entries
        .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
}

// ---------------------------------------------------------------------------
// Per-generation evaluation dispatchers
// ---------------------------------------------------------------------------
fn run_phase0_generation(
    genomes: &[sueca_solver::wann::RustWannNetwork],
    dataset: &ExpertDataset,
    sweep_weights: &[f64],
) -> (Vec<f64>, Vec<f64>, usize) {
    let accs = evaluate_phase0(genomes, dataset, sweep_weights);
    (accs.clone(), accs, 0)
}

fn run_phase1_generation(
    genomes: &[sueca_solver::wann::RustWannNetwork],
    hof: &HallOfFame,
    config: &Config,
    gen: usize,
    rng: &mut Pcg64,
) -> (Vec<f64>, Vec<f64>, usize) {
    let deals = generate_deals_rust(
        gen,
        config.evaluation.n_deals,
        config.evaluation.seed * 1000,
    );

    let mut hof_genomes = Vec::new();
    let partner_bot_type: i32;
    let opp1_bot_type: i32;
    let opp2_bot_type: i32;

    if hof.entries.is_empty() {
        partner_bot_type = 1;
        opp1_bot_type = 1;
        opp2_bot_type = 1;
    } else {
        let sampled = hof.sample(rng, 2);
        hof_genomes.push(sampled[0].to_rust_wann());
        partner_bot_type = 2;

        if sampled.len() >= 2 {
            hof_genomes.push(sampled[1].to_rust_wann());
            opp1_bot_type = 3;
        } else {
            opp1_bot_type = 1;
        }
        opp2_bot_type = 1;
    }

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
        gen,
        config.evaluation.curriculum_gens,
        config.curriculum.phase0_gens,
    )
}

// ---------------------------------------------------------------------------
// Main Training Loop
// ---------------------------------------------------------------------------
pub fn train(config: Config, resume: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Pcg64::seed_from_u64(config.evaluation.seed);

    fs::create_dir_all(&config.output.checkpoint_dir)?;

    let state_checkpoint_path = Path::new(&config.output.checkpoint_dir).join("training_state.bin");

    let mut pop: Population;
    let mut hof: HallOfFame;
    let mut registry: InnovationRegistry;
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
        };
        hof = state.hof;
        registry = InnovationRegistry::new(state.next_innovation);
    } else {
        registry = InnovationRegistry::new(0);
        pop = Population::new(config.clone(), &mut rng, &mut registry);
        hof = HallOfFame::new(config.hall_of_fame.hof_size);
    }

    // Load dataset for Phase 0
    let dataset = load_expert_dataset("expert_states_w50_d2.npz")?;

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
            "generation,phase,best_fitness,avg_fitness,median_fitness,best_delta,median_delta,global_best_fitness,n_species,n_connections_best,n_hidden_best,oracle_tax,elapsed_sec"
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
                // Transfer HOF to Phase 1 fitness metric instead of clearing
                println!(
                    "  >>> Re-evaluating {} HOF entries under Phase 1 fitness...",
                    hof.entries.len()
                );
                transfer_hof_to_phase1(&mut hof, &config, gen, &mut rng);
                pop.global_best_fitness = f64::NEG_INFINITY;
                pop.global_best_genome = None;
            }
        }

        let rust_genomes: Vec<sueca_solver::wann::RustWannNetwork> =
            pop.genomes.iter().map(|g| g.to_rust_wann()).collect();

        let (fitnesses, deltas, _total_illegal) = if current_phase == 0 {
            run_phase0_generation(&rust_genomes, &dataset, &config.evaluation.sweep_weights)
        } else {
            run_phase1_generation(&rust_genomes, &hof, &config, gen, &mut rng)
        };

        pop.tell_fitnesses(&fitnesses);

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

        let tax = oracle_tax_penalty(
            gen,
            config.curriculum.phase0_gens,
            config.evaluation.curriculum_gens,
        );

        writeln!(
            csv_file,
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{:.2},{:.2}",
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
            tax,
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
        pop.speciate_and_evolve(
            current_phase,
            config.curriculum.bulking_gens,
            &mut rng,
            &mut registry,
        );

        // Checkpointing
        if (gen + 1) % 10 == 0 || gen == config.population.generations - 1 {
            // Save HOF
            let hof_path = Path::new(&config.output.checkpoint_dir)
                .join(format!("hof_gen{:04}.json", gen + 1));
            let json_hof = JsonHallOfFame::from_hof(&hof);
            let hof_file = fs::File::create(hof_path)?;
            serde_json::to_writer_pretty(hof_file, &json_hof)?;

            // Save global best genome
            if let Some(ref gb) = pop.global_best_genome {
                let best_path = Path::new(&config.output.checkpoint_dir)
                    .join(format!("best_genome_gen{:04}.json", gen + 1));
                let json_gb = JsonGenome::from_genome(gb);
                let gb_file = fs::File::create(best_path)?;
                serde_json::to_writer_pretty(gb_file, &json_gb)?;
            }

            // Save generation best genome
            let gen_best_path = Path::new(&config.output.checkpoint_dir)
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
                next_innovation: registry.next_innovation,
                current_phase,
            };
            state.save_to_file(&state_checkpoint_path)?;
        }
    }

    // Save final files
    let hof_path = Path::new(&config.output.checkpoint_dir).join("hof_final.json");
    let json_hof = JsonHallOfFame::from_hof(&hof);
    let hof_file = fs::File::create(hof_path)?;
    serde_json::to_writer_pretty(hof_file, &json_hof)?;

    if let Some(ref gb) = pop.global_best_genome {
        let best_path = Path::new(&config.output.checkpoint_dir).join("best_genome_final.json");
        let json_gb = JsonGenome::from_genome(gb);
        let gb_file = fs::File::create(best_path)?;
        serde_json::to_writer_pretty(gb_file, &json_gb)?;
    }

    Ok(())
}
