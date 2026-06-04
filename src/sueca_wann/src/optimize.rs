use crate::wann_network::RustWannNetwork;
use crate::evaluator::{play_game_sim, rotate_hands, SimulatorBot, WannBehavior, EvaluatorDeal};
use crate::train::generate_deals_rust;

use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug)]
pub struct OptimizedWeightsReport {
    pub best_fitness: f64,
    pub lead_weights: Vec<f64>,
    pub follow_weights: Vec<f64>,
}

pub fn run_weight_optimization(
    genome_path: &str,
    n_deals: usize,
    generations: usize,
    seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting Differential Evolution weight optimization...");
    println!("  Genome: {}", genome_path);
    println!("  Deals: {}", n_deals);
    println!("  Generations: {}", generations);
    println!("  Seed: {}", seed);

    // 1. Load genome
    let (lead_opt, follow_opt) = crate::compile_rules::load_genome(genome_path)?;
    let lead_genome = lead_opt.ok_or("No Lead Brain genome found")?;
    let follow_genome = follow_opt.ok_or("No Follow Brain genome found")?;

    let lead_net = lead_genome.to_rust_wann();
    let follow_net = follow_genome.to_rust_wann();

    // 2. Count enabled connections
    let lead_conns_len = lead_genome.conn_genes.iter().filter(|c| c.enabled).count();
    let follow_conns_len = follow_genome.conn_genes.iter().filter(|c| c.enabled).count();
    let total_dim = lead_conns_len + follow_conns_len;

    println!("Lead Brain enabled connections: {}", lead_conns_len);
    println!("Follow Brain enabled connections: {}", follow_conns_len);
    println!("Total optimization dimension: {}", total_dim);

    if total_dim == 0 {
        return Err("No enabled connections to optimize".into());
    }

    // 3. Generate duplicate deals for evaluation
    let deals = generate_deals_rust(0, n_deals, seed * 1000);
    println!("Generated {} duplicate evaluation deals.", n_deals);

    // 4. Precompute HeuristicBot baseline scores using duplicate deals
    println!("Precomputing HeuristicBot baseline scores...");
    let max_nodes = lead_net.num_nodes.max(follow_net.num_nodes);
    let baseline_scores: Vec<f64> = (0..deals.len() * 4)
        .into_par_iter()
        .map(|idx| {
            let deal_idx = idx / 4;
            let rot = idx % 4;
            let deal = &deals[deal_idx];

            let partner = SimulatorBot::Heuristic;
            let opp1 = SimulatorBot::Heuristic;
            let opp2 = SimulatorBot::Heuristic;
            let baseline = SimulatorBot::Heuristic;

            let rotated_hands = rotate_hands(&deal.hands, rot);
            let adj_first = rot as u8;
            let evaluation_seed = (seed * 1000) ^ ((deal_idx as u64) << 32) ^ (rot as u64);

            let bots_baseline = [
                baseline,
                opp1,
                partner,
                opp2,
            ];

            let mut dummy_behavior = WannBehavior::default();
            let mut scratchpad = vec![0.0f64; max_nodes];
            let result_baseline = play_game_sim(
                rotated_hands,
                deal.trump,
                adj_first,
                &bots_baseline,
                evaluation_seed,
                &mut scratchpad,
                &mut dummy_behavior,
            );
            result_baseline.team_02_score as f64
        })
        .collect();

    // 5. Differential Evolution parameters
    let pop_size = crate::constants::DE_POP_SIZE;
    let f_scaling = crate::constants::DE_F_SCALING;
    let cr_crossover = crate::constants::DE_CR_CROSSOVER;
    let weight_min = crate::constants::DE_WEIGHT_MIN;
    let weight_max = crate::constants::DE_WEIGHT_MAX;

    let mut rng = Pcg64::seed_from_u64(seed);

    // Initialize population:
    // Candidate 0: exact baseline of 1.0 (equivalent to the evolved uniform W = 1.0 genome)
    // Other candidates: perturbed around 1.0 within clamping limits
    let mut population: Vec<Vec<f64>> = Vec::with_capacity(pop_size);
    population.push(vec![1.0; total_dim]);
    for _ in 1..pop_size {
        let vec: Vec<f64> = (0..total_dim)
            .map(|_| {
                let noise = rng.gen_range(-0.5..=0.5);
                (1.0_f64 + noise).clamp(weight_min, weight_max)
            })
            .collect();
        population.push(vec);
    }

    // Evaluate initial population
    println!("Evaluating initial population...");
    let mut fitnesses: Vec<f64> = population
        .par_iter()
        .map(|candidate| {
            evaluate_candidate_weights(
                &lead_net,
                &follow_net,
                lead_conns_len,
                follow_conns_len,
                candidate,
                &deals,
                &baseline_scores,
                seed * 1000,
            )
        })
        .collect();

    let mut best_fitness = f64::NEG_INFINITY;
    let mut best_candidate = population[0].clone();
    for i in 0..pop_size {
        if fitnesses[i] > best_fitness {
            best_fitness = fitnesses[i];
            best_candidate = population[i].clone();
        }
    }

    println!("Initial Best Fitness (Delta vs HeuristicBot): {:.4}", best_fitness);

    // 6. Optimization Loop
    for gen in 1..=generations {
        let mut next_population = population.clone();
        let mut next_fitnesses = fitnesses.clone();

        // Evaluate mutations/crossovers in parallel
        let results: Vec<(Vec<f64>, f64)> = (0..pop_size)
            .into_par_iter()
            .map_init(
                || Pcg64::seed_from_u64(seed + gen as u64 * 999),
                |local_rng, i| {
                    // DE/best/1/bin mutant generation:
                    // Find two distinct random indices different from i
                    let mut r1 = local_rng.gen_range(0..pop_size);
                    while r1 == i {
                        r1 = local_rng.gen_range(0..pop_size);
                    }
                    let mut r2 = local_rng.gen_range(0..pop_size);
                    while r2 == i || r2 == r1 {
                        r2 = local_rng.gen_range(0..pop_size);
                    }

                    // Find best candidate index in the current population
                    let mut current_best_idx = 0;
                    let mut current_best_fit = fitnesses[0];
                    for idx in 1..pop_size {
                        if fitnesses[idx] > current_best_fit {
                            current_best_fit = fitnesses[idx];
                            current_best_idx = idx;
                        }
                    }

                    let best_vec = &population[current_best_idx];
                    let x_r1 = &population[r1];
                    let x_r2 = &population[r2];

                    let mut trial = vec![0.0; total_dim];
                    let rand_j = local_rng.gen_range(0..total_dim);

                    for j in 0..total_dim {
                        if local_rng.gen::<f64>() < cr_crossover || j == rand_j {
                            // Mutate
                            let val = best_vec[j] + f_scaling * (x_r1[j] - x_r2[j]);
                            // Strictly clamp to search boundary to prevent threshold saturation
                            trial[j] = val.clamp(weight_min, weight_max);
                        } else {
                            trial[j] = population[i][j];
                        }
                    }

                    let fit_trial = evaluate_candidate_weights(
                        &lead_net,
                        &follow_net,
                        lead_conns_len,
                        follow_conns_len,
                        &trial,
                        &deals,
                        &baseline_scores,
                        seed * 1000,
                    );

                    if fit_trial >= fitnesses[i] {
                        (trial, fit_trial)
                    } else {
                        (population[i].clone(), fitnesses[i])
                    }
                },
            )
            .collect();

        for i in 0..pop_size {
            next_population[i] = results[i].0.clone();
            next_fitnesses[i] = results[i].1;

            if next_fitnesses[i] > best_fitness {
                best_fitness = next_fitnesses[i];
                best_candidate = next_population[i].clone();
            }
        }

        population = next_population;
        fitnesses = next_fitnesses;

        let avg_fit: f64 = fitnesses.iter().sum::<f64>() / (pop_size as f64);
        println!(
            "Gen {:2}/{} | Best: {:.4} | Avg: {:.4}",
            gen, generations, best_fitness, avg_fit
        );
    }

    // 7. Save report
    let (best_lead_weights, best_follow_weights) = best_candidate.split_at(lead_conns_len);
    let report = OptimizedWeightsReport {
        best_fitness,
        lead_weights: best_lead_weights.to_vec(),
        follow_weights: best_follow_weights.to_vec(),
    };

    let parent_dir = Path::new(genome_path).parent().unwrap_or(Path::new("."));
    let output_path = parent_dir.join("optimized_weights.json");
    let file = File::create(&output_path)?;
    serde_json::to_writer_pretty(file, &report)?;

    println!(
        "Optimization complete! Best Fitness: {:.4}. Saved weights to {}",
        best_fitness,
        output_path.display()
    );

    Ok(())
}

fn evaluate_candidate_weights(
    lead_net: &RustWannNetwork,
    follow_net: &RustWannNetwork,
    lead_conns_len: usize,
    _follow_conns_len: usize,
    candidate: &[f64],
    deals: &[EvaluatorDeal],
    baseline_scores: &[f64],
    base_seed: u64,
) -> f64 {
    let (lead_weights, follow_weights) = candidate.split_at(lead_conns_len);
    let max_nodes = lead_net.num_nodes.max(follow_net.num_nodes);

    let candidate_bot = SimulatorBot::WannWeighted {
        lead_brain: lead_net,
        follow_brain: follow_net,
        lead_weights,
        follow_weights,
    };

    let partner = SimulatorBot::Heuristic;
    let opp1 = SimulatorBot::Heuristic;
    let opp2 = SimulatorBot::Heuristic;

    let mut total_score_candidate = 0.0;
    let mut total_score_baseline = 0.0;
    let first_player = 0;

    for (d_idx, deal) in deals.iter().enumerate() {
        for rot in 0..4 {
            let rotated_hands = rotate_hands(&deal.hands, rot);
            let adj_first = (first_player + rot as u8) % 4;
            let evaluation_seed = base_seed ^ ((d_idx as u64) << 32) ^ (rot as u64);

            let bots_candidate = [
                candidate_bot.clone(),
                opp1.clone(),
                partner.clone(),
                opp2.clone(),
            ];

            let mut dummy_behavior = WannBehavior::default();
            let mut scratchpad = vec![0.0f64; max_nodes];
            let result_candidate = play_game_sim(
                rotated_hands,
                deal.trump,
                adj_first,
                &bots_candidate,
                evaluation_seed,
                &mut scratchpad,
                &mut dummy_behavior,
            );

            total_score_candidate += result_candidate.team_02_score as f64;
            total_score_baseline += baseline_scores[d_idx * 4 + rot];
        }
    }

    let num_games = (deals.len() * 4) as f64;
    (total_score_candidate - total_score_baseline) / num_games
}
