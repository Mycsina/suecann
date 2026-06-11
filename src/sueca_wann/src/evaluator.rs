use crate::wann_network::RustWannNetwork;

use sueca_solver::belief::encode_belief_state;
use sueca_solver::heuristic::{
    resolve_intent, select_card_heuristic, select_card_heuristic_old, select_card_random,
};
use sueca_solver::rng::LcgRng;
use sueca_solver::simulator::SuecaSimulatorGame;

#[derive(Debug, Clone, Default)]
pub struct WannBehavior {
    pub intent_counts: [usize; 3],
    pub total_lead_points: usize,
    pub count_leads: usize,
    pub total_actions: usize,
}

// ---------------------------------------------------------------------------
// Bot Strategy Types for Game Simulation
// ---------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub enum SimulatorBot<'a> {
    Random,
    OldHeuristic,
    Heuristic,
    Wann {
        lead_brain: &'a RustWannNetwork,
        follow_brain: &'a RustWannNetwork,
        weights: &'a [f64],
    },
    WannWeighted {
        lead_brain: &'a RustWannNetwork,
        follow_brain: &'a RustWannNetwork,
        lead_weights: &'a [f64],
        follow_weights: &'a [f64],
    },
    Pimc {
        n_worlds: usize,
        search_depth: u8,
    },
}

impl<'a> SimulatorBot<'a> {
    pub fn select_card(
        &self,
        game: &SuecaSimulatorGame,
        seat: u8,
        rng: &mut LcgRng,
        scratchpad: &mut [f64],
        mut behavior: Option<&mut WannBehavior>,
    ) -> u8 {
        match self {
            SimulatorBot::Random => {
                let legal = game.state.legal_moves();
                select_card_random(legal, rng)
            }
            SimulatorBot::OldHeuristic => select_card_heuristic_old(game, seat),
            SimulatorBot::Heuristic => select_card_heuristic(game, seat),
            SimulatorBot::Wann {
                lead_brain,
                follow_brain,
                weights,
            } => {
                let belief = encode_belief_state(game, seat);

                // Dynamically route to Lead or Follow brain per card-play slice
                let network =
                    if (belief[crate::constants::BeliefFeature::AmILeading as usize] - 1.0).abs()
                        < 1e-9
                    {
                        lead_brain
                    } else {
                        follow_brain
                    };

                // Weight-batched forward: single CSR traversal for all sweep weights
                let n_weights = weights.len();
                network.forward_batched(&belief, weights, scratchpad);

                let mut sum_outputs = [0.0f64; sueca_solver::constants::OUTPUT_COUNT];
                for w in 0..n_weights {
                    sum_outputs[0] +=
                        scratchpad[(sueca_solver::constants::OUTPUT_START + 0) * n_weights + w];
                    sum_outputs[1] +=
                        scratchpad[(sueca_solver::constants::OUTPUT_START + 1) * n_weights + w];
                    sum_outputs[2] +=
                        scratchpad[(sueca_solver::constants::OUTPUT_START + 2) * n_weights + w];
                }

                let mut mean_outputs = [0.0f64; sueca_solver::constants::OUTPUT_COUNT];
                for i in 0..sueca_solver::constants::OUTPUT_COUNT {
                    mean_outputs[i] = sum_outputs[i] / (n_weights as f64);
                }

                let adjusted_outputs = mean_outputs;

                let mut max_val = adjusted_outputs[0];
                for i in 1..sueca_solver::constants::OUTPUT_COUNT {
                    if adjusted_outputs[i] > max_val {
                        max_val = adjusted_outputs[i];
                    }
                }

                let mut best_intents = [0usize; sueca_solver::constants::OUTPUT_COUNT];
                let mut best_count = 0;
                for i in 0..sueca_solver::constants::OUTPUT_COUNT {
                    if (adjusted_outputs[i] - max_val).abs() < 1e-9 {
                        best_intents[best_count] = i;
                        best_count += 1;
                    }
                }

                let chosen_intent = if best_count == 1 {
                    best_intents[0]
                } else {
                    best_intents[rng.gen_range(0..best_count)]
                };

                let card = resolve_intent(chosen_intent, game, seat);

                if let Some(ref mut beh) = behavior {
                    if chosen_intent < sueca_solver::constants::OUTPUT_COUNT {
                        beh.intent_counts[chosen_intent] += 1;
                    }
                    if game.current_trick_len == 0 {
                        beh.total_lead_points +=
                            sueca_solver::engine::CARD_POINTS[card as usize] as usize;
                        beh.count_leads += 1;
                    }
                    beh.total_actions += 1;
                }

                card
            }
            SimulatorBot::WannWeighted {
                lead_brain,
                follow_brain,
                lead_weights,
                follow_weights,
            } => {
                let belief = encode_belief_state(game, seat);

                // Dynamically route to Lead or Follow brain per card-play slice
                let is_leading = (belief[crate::constants::BeliefFeature::AmILeading as usize] - 1.0).abs() < 1e-9;
                let (network, brain_weights) = if is_leading {
                    (lead_brain, lead_weights)
                } else {
                    (follow_brain, follow_weights)
                };

                // Forward weighted pass
                network.forward_weighted(&belief, brain_weights, scratchpad);

                let adjusted_outputs = [
                    scratchpad[sueca_solver::constants::OUTPUT_START],
                    scratchpad[sueca_solver::constants::OUTPUT_START + 1],
                    scratchpad[sueca_solver::constants::OUTPUT_START + 2],
                ];

                let mut max_val = adjusted_outputs[0];
                for i in 1..sueca_solver::constants::OUTPUT_COUNT {
                    if adjusted_outputs[i] > max_val {
                        max_val = adjusted_outputs[i];
                    }
                }

                let mut best_intents = [0usize; sueca_solver::constants::OUTPUT_COUNT];
                let mut best_count = 0;
                for i in 0..sueca_solver::constants::OUTPUT_COUNT {
                    if (adjusted_outputs[i] - max_val).abs() < 1e-9 {
                        best_intents[best_count] = i;
                        best_count += 1;
                    }
                }

                let chosen_intent = if best_count == 1 {
                    best_intents[0]
                } else {
                    best_intents[rng.gen_range(0..best_count)]
                };

                let card = resolve_intent(chosen_intent, game, seat);

                if let Some(ref mut beh) = behavior {
                    if chosen_intent < sueca_solver::constants::OUTPUT_COUNT {
                        beh.intent_counts[chosen_intent] += 1;
                    }
                    if game.current_trick_len == 0 {
                        beh.total_lead_points +=
                            sueca_solver::engine::CARD_POINTS[card as usize] as usize;
                        beh.count_leads += 1;
                    }
                    beh.total_actions += 1;
                }

                card
            }
            SimulatorBot::Pimc {
                n_worlds,
                search_depth,
            } => {
                let legal = game.state.legal_moves();
                if legal.count_ones() == 1 {
                    return legal.trailing_zeros() as u8;
                }

                let mut current_hands = 0u64;
                for h in &game.state.hands {
                    current_hands |= h;
                }
                let played_cards_mask = (!current_hands) & 0x000000FFFFFFFFFFu64;

                let mut target_sizes = [0u8; 4];
                for s in 0..4 {
                    target_sizes[s] = game.state.hands[s].count_ones() as u8;
                }

                let led_suit = if game.current_trick_len > 0 {
                    sueca_solver::engine::CARD_SUIT[game.current_trick[0] as usize]
                } else {
                    4
                };

                let current_trick_cards = &game.current_trick[..game.current_trick_len];

                let evs = sueca_solver::pimc::solve_pimc(
                    seat,
                    game.state.hands[seat as usize],
                    played_cards_mask,
                    game.voids,
                    target_sizes,
                    game.state.trump,
                    led_suit,
                    current_trick_cards,
                    game.state.current_player,
                    game.state.current_trick_winner,
                    game.state.current_trick_best_card,
                    [game.state.team_02_score, game.state.team_13_score],
                    game.state.trick_number,
                    *n_worlds,
                    *search_depth,
                    rng.next_u64(),
                );

                let mut best_card = legal.trailing_zeros() as u8;
                let mut max_ev = f64::NEG_INFINITY;
                for r in evs {
                    if r.ev > max_ev {
                        max_ev = r.ev;
                        best_card = r.card;
                    }
                }
                best_card
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Seat Rotation Helpers
// ---------------------------------------------------------------------------
#[inline(always)]
pub fn rotate_hands(hands: &[u64; 4], rotation: usize) -> [u64; 4] {
    let mut rot_hands = [0u64; 4];
    for i in 0..4 {
        rot_hands[i] = hands[(i + 4 - rotation) % 4];
    }
    rot_hands
}

#[inline(always)]
fn rotate_first_player(first_player: u8, rotation: usize) -> u8 {
    (first_player + rotation as u8) % 4
}

// ---------------------------------------------------------------------------
// Game Simulation Runner
// ---------------------------------------------------------------------------
#[allow(dead_code)]
pub struct GameResultSim {
    pub team_02_score: u8,
    pub team_13_score: u8,
    pub team_02_game_points: u8,
    pub team_13_game_points: u8,
}

fn compute_game_points(team_02: u8, team_13: u8) -> (u8, u8) {
    let tier = |pts: u8| -> u8 {
        if pts == 120 {
            4
        } else if pts >= 91 {
            2
        } else if pts >= 61 {
            1
        } else {
            0
        }
    };
    (tier(team_02), tier(team_13))
}

pub fn play_game_sim(
    hands: [u64; 4],
    trump: u8,
    first_player: u8,
    bots: &[SimulatorBot; 4],
    seed: u64,
    scratchpad: &mut [f64],
    behavior: &mut WannBehavior,
) -> GameResultSim {
    let mut rng = LcgRng::new(seed);
    let mut game = SuecaSimulatorGame::new(hands, trump, first_player);

    while game.state.trick_number < 10 {
        let seat = game.state.current_player;
        let card = if seat == 0 {
            bots[seat as usize].select_card(&game, seat, &mut rng, scratchpad, Some(behavior))
        } else {
            bots[seat as usize].select_card(&game, seat, &mut rng, scratchpad, None)
        };
        game.play_card(card);
    }

    let team_02 = game.state.team_02_score;
    let team_13 = game.state.team_13_score;
    let (gp_02, gp_13) = compute_game_points(team_02, team_13);

    GameResultSim {
        team_02_score: team_02,
        team_13_score: team_13,
        team_02_game_points: gp_02,
        team_13_game_points: gp_13,
    }
}

// ---------------------------------------------------------------------------
// Evaluator API & Parallel Execution
// ---------------------------------------------------------------------------
pub struct EvaluatorDeal {
    pub hands: [u64; 4],
    pub trump: u8,
    pub seed: u64,
}

pub fn get_bot_from_type<'a>(
    bot_type: i32,
    hof_lead_networks: &'a [RustWannNetwork],
    hof_follow_networks: &'a [RustWannNetwork],
    sweep_weights: &'a [f64],
) -> SimulatorBot<'a> {
    match bot_type {
        -1 => SimulatorBot::Pimc {
            n_worlds: 5,
            search_depth: 1,
        },
        0 => SimulatorBot::Random,
        1 => SimulatorBot::OldHeuristic,
        2 => SimulatorBot::Heuristic,
        3 => SimulatorBot::Heuristic,
        t if t >= 10 => {
            let idx = (t - 10) as usize;
            SimulatorBot::Wann {
                lead_brain: &hof_lead_networks[idx],
                follow_brain: &hof_follow_networks[idx],
                weights: sweep_weights,
            }
        }
        _ => SimulatorBot::Random,
    }
}

/// Evaluate a candidate genome vs a baseline bot on the same deals with CRN.
/// Returns (average_delta, behavior_metrics) — fitness is computed in Python.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_genome_delta(
    candidate_lead: &RustWannNetwork,
    candidate_follow: &RustWannNetwork,
    partner_bot_type: i32,
    opp1_bot_type: i32,
    opp2_bot_type: i32,
    hof_lead_networks: &[RustWannNetwork],
    hof_follow_networks: &[RustWannNetwork],
    sweep_weights: &[f64],
    deals: &[EvaluatorDeal],
    base_seed: u64,
    baseline_scores: &[f64],
    scratchpad: &mut [f64],
) -> (f64, WannBehavior) {
    let mut total_score_candidate = 0.0;
    let mut total_score_baseline = 0.0;
    let mut accum_behavior = WannBehavior::default();

    let first_player = 0;

    let partner = get_bot_from_type(partner_bot_type, hof_lead_networks, hof_follow_networks, sweep_weights);
    let opp1 = get_bot_from_type(opp1_bot_type, hof_lead_networks, hof_follow_networks, sweep_weights);
    let opp2 = get_bot_from_type(opp2_bot_type, hof_lead_networks, hof_follow_networks, sweep_weights);

    let candidate_bot = SimulatorBot::Wann {
        lead_brain: candidate_lead,
        follow_brain: candidate_follow,
        weights: sweep_weights,
    };

    for (d_idx, deal) in deals.iter().enumerate() {
        for rot in 0..4 {
            let rotated_hands = rotate_hands(&deal.hands, rot);
            let adj_first = rotate_first_player(first_player, rot);
            let evaluation_seed = base_seed ^ ((d_idx as u64) << 32) ^ (rot as u64);

            // --- Play with candidate WANN ---
            let bots_candidate = [
                candidate_bot.clone(),
                opp1.clone(),
                partner.clone(),
                opp2.clone(),
            ];

            let mut game_behavior = WannBehavior::default();
            let result_candidate = play_game_sim(
                rotated_hands,
                deal.trump,
                adj_first,
                &bots_candidate,
                evaluation_seed,
                scratchpad,
                &mut game_behavior,
            );

            for i in 0..3 {
                accum_behavior.intent_counts[i] += game_behavior.intent_counts[i];
            }
            accum_behavior.total_lead_points += game_behavior.total_lead_points;
            accum_behavior.count_leads += game_behavior.count_leads;
            accum_behavior.total_actions += game_behavior.total_actions;

            total_score_candidate += result_candidate.team_02_score as f64;
            total_score_baseline += baseline_scores[d_idx * 4 + rot];
        }
    }

    let num_games = (deals.len() * 4) as f64;
    let avg_candidate = total_score_candidate / num_games;
    let avg_baseline = total_score_baseline / num_games;
    let avg_delta = avg_candidate - avg_baseline;

    (avg_delta, accum_behavior)
}

#[cfg(test)]
mod tests {
    use crate::genome::Genome;
    use crate::train::evaluate_phase1;
    use crate::train::generate_deals_rust;

    #[test]
    fn test_evaluate_phase1_compiles_and_runs() {
        let g_lead = Genome::initial().to_rust_wann();
        let g_follow = Genome::initial().to_rust_wann();

        let deals = generate_deals_rust(0, 2, 42);
        let sweep_weights = vec![1.0];

        let (fitnesses, deltas, behaviors) = evaluate_phase1(
            &[g_lead.clone()],
            true, // is_lead
            &deals,
            &g_follow,
            &[],
            &[],
            1, // partner_bot_type = OldHeuristic
            1, // opp1_bot_type = OldHeuristic
            1, // opp2_bot_type = OldHeuristic
            1, // baseline_bot_type = OldHeuristic
            &sweep_weights,
            123, // base_seed
        );

        assert_eq!(fitnesses.len(), 1);
        assert_eq!(deltas.len(), 1);
        assert_eq!(behaviors.len(), 1);
        assert!(deltas[0].is_finite());
    }

    #[test]
    fn test_evaluate_phase1_joint_compiles_and_runs() {
        let g_lead = Genome::initial().to_rust_wann();
        let g_follow = Genome::initial().to_rust_wann();

        let deals = generate_deals_rust(0, 2, 42);
        let sweep_weights = vec![1.0];

        let (fitnesses, behaviors) = crate::train::evaluate_phase1_joint(
            &[g_lead.clone()],
            &[g_follow.clone()],
            &deals,
            &[],
            &[],
            1, // partner_bot_type = OldHeuristic
            1, // opp1_bot_type = OldHeuristic
            1, // opp2_bot_type = OldHeuristic
            1, // baseline_bot_type = OldHeuristic
            &sweep_weights,
            123, // base_seed
        );

        assert_eq!(fitnesses.len(), 1);
        assert_eq!(behaviors.len(), 1);
        assert!(fitnesses[0].is_finite());
    }

    #[test]
    fn test_joint_breeding_alignment() {
        use crate::config::Config;
        use crate::population::{Population, speciate_and_evolve_joint};
        use crate::mutations::{InnovationRegistry, TabuVetoList};
        use std::sync::Mutex;
        use rand_pcg::Pcg64;
        use rand::SeedableRng;

        let mut rng = Pcg64::seed_from_u64(42);
        let config = Config {
            population: crate::config::PopulationConfig {
                pop_size: 10,
                generations: 3,
                elitism: 1,
                pareto_complexity_prob: 0.80,
            },
            evaluation: crate::config::EvaluationConfig {
                n_deals: 2,
                sweep_weights: vec![1.0],
                seed: 1337,
            },
            species: crate::config::SpeciesConfig {
                compatibility_threshold: 1.4,
                stagnation_limit: 40,
                c_excess: 1.0,
                c_disjoint: 1.0,
                c_mismatch: 0.5,
                min_species_size: 1,
                max_species: 20,
            },
            mutation: crate::config::MutationConfig {
                p_add_node: 0.20,
                p_add_conn: 0.35,
                p_toggle_conn: 0.05,
                p_flip_sign: 0.10,
                p_change_act: 0.25,
                p_change_agg: 0.15,
                p_crossover: 0.40,
            },
            curriculum: crate::config::CurriculumConfig {
                phase0_gens: 1,
                bulking_gens: 1,
                min_gens_per_phase: 1,
                adaptive_window: 10,
                phase0_dataset: "expert_states_w20_d2.npz".to_string(),
                pfs_sample_size: 100,
                class_balance_target: 30000,
                soft_balance_min_ratio: 0.20,
                use_class_weighting: true,
            },
            hall_of_fame: crate::config::HallOfFameConfig {
                hof_size: 5,
            },
            output: crate::config::OutputConfig {
                checkpoint_dir: "checkpoints/test_run".to_string(),
                stats_file: "checkpoints/test_run/training_stats.csv".to_string(),
            },
        };


        let lead_registry = Mutex::new(InnovationRegistry::new(0));
        let follow_registry = Mutex::new(InnovationRegistry::new(0));

        let mut lead_pop = Population::new(config.clone(), &mut rng, &lead_registry);
        let mut follow_pop = Population::new(config.clone(), &mut rng, &follow_registry);

        let lead_tabu = TabuVetoList::new(1000);
        let follow_tabu = TabuVetoList::new(1000);

        // Assign some dummy fitnesses
        for i in 0..config.population.pop_size {
            lead_pop.fitnesses[i] = i as f64 * 0.1;
            follow_pop.fitnesses[i] = i as f64 * 0.1;
        }

        // Perform joint speciation and evolution
        speciate_and_evolve_joint(
            &mut lead_pop,
            &mut follow_pop,
            &lead_tabu,
            &follow_tabu,
            &lead_registry,
            &follow_registry,
            &mut rng,
        );

        // Check that populations remained aligned and have the correct pop size
        assert_eq!(lead_pop.genomes.len(), config.population.pop_size);
        assert_eq!(follow_pop.genomes.len(), config.population.pop_size);

        // Speciate list in both pops should be identical
        assert_eq!(lead_pop.species_list.len(), follow_pop.species_list.len());
        for (sp_l, sp_f) in lead_pop.species_list.iter().zip(follow_pop.species_list.iter()) {
            assert_eq!(sp_l.id, sp_f.id);
            assert_eq!(sp_l.members, sp_f.members);
        }
    }
}
