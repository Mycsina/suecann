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
                    false,
                    None,
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

    // -----------------------------------------------------------------------
    // Resolver-ceiling diagnostic.
    //
    // Question: can the 3-intent resolver even EXPRESS Elite-grade play?
    // We measure, over realistic game states, how often Elite's chosen card is
    // reproducible by SOME intent (expressiveness coverage), and we play each
    // pure intent + a "mimic-Elite oracle" head-to-head vs the Elite team.
    //
    // If coverage < 100% and the oracle (the best a perfect intent-selector
    // could do) cannot match Elite, the ceiling is architectural — no amount
    // of WANN training closes the gap. Run with:
    //   cargo test -p sueca_wann --release diagnostic_resolver_ceiling -- --nocapture --test-threads=1
    // -----------------------------------------------------------------------
    #[derive(Clone, Copy)]
    enum DiagPolicy {
        Elite,
        Intent(usize),
        MimicOracle,
        RolloutOracle,
        /// Flat Monte-Carlo PIMC with Elite playouts (the Stage-1 teacher).
        RolloutPimc { worlds: usize },
    }

    /// Play a game-copy to completion with Elite in all four seats.
    fn diag_rollout_elite(mut game: sueca_solver::simulator::SuecaSimulatorGame) -> (u8, u8) {
        use sueca_solver::heuristic::select_card_heuristic;
        while game.state.trick_number < 10 {
            let s = game.state.current_player;
            let c = select_card_heuristic(&game, s);
            game.play_card(c);
        }
        (game.state.team_02_score, game.state.team_13_score)
    }

    fn diag_pick(p: DiagPolicy, game: &sueca_solver::simulator::SuecaSimulatorGame, seat: u8) -> u8 {
        use sueca_solver::heuristic::{resolve_intent, select_card_heuristic};
        match p {
            DiagPolicy::Elite => select_card_heuristic(game, seat),
            DiagPolicy::Intent(i) => resolve_intent(i, game, seat),
            DiagPolicy::MimicOracle => {
                let e = select_card_heuristic(game, seat);
                for i in 0..sueca_solver::constants::OUTPUT_COUNT {
                    if resolve_intent(i, game, seat) == e {
                        return e;
                    }
                }
                // No intent expresses Elite's move — fall back to EFFICIENT_WIN.
                resolve_intent(1, game, seat)
            }
            DiagPolicy::RolloutOracle => {
                // Per-decision best response: try each intent, continue the game
                // with Elite in all seats, keep the intent that maximizes this
                // ego-team's final score. ego team = seat % 2.
                let ego_team = seat % 2;
                let mut best_card = resolve_intent(0, game, seat);
                let mut best_score = -1i32;
                for i in 0..sueca_solver::constants::OUTPUT_COUNT {
                    let card = resolve_intent(i, game, seat);
                    let mut g = *game;
                    g.play_card(card);
                    let (t02, t13) = diag_rollout_elite(g);
                    let score = if ego_team == 0 { t02 as i32 } else { t13 as i32 };
                    if score > best_score {
                        best_score = score;
                        best_card = card;
                    }
                }
                best_card
            }
            DiagPolicy::RolloutPimc { worlds } => {
                // Flat MC PIMC with Elite playouts; pick the legal move with the
                // highest ego-team EV. Fixed seed -> reproducible diagnostic.
                let res = sueca_solver::pimc::solve_pimc_rollout(game, worlds, 0xC0FFEE);
                res.iter()
                    .max_by(|a, b| a.ev.partial_cmp(&b.ev).unwrap())
                    .map(|r| r.card)
                    .unwrap_or_else(|| select_card_heuristic(game, seat))
            }
        }
    }

    /// Play one deal with the given per-seat policies. Returns (team02, team13).
    fn diag_play(deal: &crate::evaluator::EvaluatorDeal, policies: [DiagPolicy; 4]) -> (u8, u8) {
        use sueca_solver::simulator::SuecaSimulatorGame;
        let mut game = SuecaSimulatorGame::new(deal.hands, deal.trump, 0);
        while game.state.trick_number < 10 {
            let seat = game.state.current_player;
            let card = diag_pick(policies[seat as usize], &game, seat);
            game.play_card(card);
        }
        (game.state.team_02_score, game.state.team_13_score)
    }

    // Dump (belief[35], rollout-optimal-intent) for DECISIVE states to CSV, so we
    // can measure (in Python) whether the belief features can predict the winning
    // intent at all. Sampled on the Elite-vs-Elite trajectory (the states faced
    // when playing the opponent we must beat).
    //   cargo test -p sueca_wann --release diagnostic_dump_intent_labels -- --nocapture --test-threads=1
    #[test]
    #[ignore = "on-demand diagnostic; run explicitly"]
    fn diagnostic_dump_intent_labels() {
        use std::io::Write;
        use rand::SeedableRng;
        use rand_pcg::Pcg64;

        // Use the REAL PIMC-labeling extraction path (no balancer). Each accepted
        // state carries the best-of-3-intent soft label. Dump (belief[35], argmax)
        // so Python RF can measure whether the belief predicts the best intent.
        let config = crate::dataset_gen::DatasetConfig {
            n_worlds: 100,
            search_depth: 4,
            target_total: 0,
            seed: 4242,
            output_path: String::new(),
            soft_balance_min_ratio: 0.0,
            diff_mode: false,
            fixed_worlds: None,
        };
        let n_deals: u64 = std::env::var("DUMP_DEALS").ok().and_then(|s| s.parse().ok()).unwrap_or(4000);
        let mut out = String::new();
        let mut counts = [0usize; 3];
        let mut n = 0usize;
        for d in 0..n_deals {
            let mut rng = Pcg64::seed_from_u64(4242u64.wrapping_mul(1000).wrapping_add(d));
            let (states, _rej) = crate::dataset_gen::generate_deal_states(&mut rng, &config, None);
            for (belief, soft, _mask) in states {
                let lab = (0..3)
                    .max_by(|&a, &b| soft[a].partial_cmp(&soft[b]).unwrap())
                    .unwrap();
                counts[lab] += 1;
                n += 1;
                for v in belief.iter() {
                    out.push_str(&format!("{:.5},", v));
                }
                out.push_str(&format!("{}\n", lab));
            }
        }
        let path = "/tmp/champ/intent_labels.csv";
        std::fs::File::create(path).unwrap().write_all(out.as_bytes()).unwrap();
        println!("Wrote {} PIMC-labeled states to {}  (label counts {:?})", n, path, counts);
    }

    // How good is Elite vs near-optimal (strong PIMC) play? This bounds what ANY
    // belief-state policy can achieve against Elite. If strong PIMC only ties
    // Elite, Elite is near-optimal and beating it is essentially impossible; if
    // PIMC crushes Elite, the headroom is real and the bottleneck is the policy.
    //   cargo test -p sueca_wann --release diagnostic_pimc_vs_elite -- --nocapture --test-threads=1
    #[test]
    #[ignore = "on-demand diagnostic; run explicitly"]
    fn diagnostic_pimc_vs_elite() {
        use crate::train::generate_deals_rust;
        let n = std::env::var("PIMC_DEALS").ok().and_then(|s| s.parse().ok()).unwrap_or(120usize);
        let deals = generate_deals_rust(0, n, 31337);
        let nw: usize = std::env::var("PIMC_WORLDS").ok().and_then(|s| s.parse().ok()).unwrap_or(80);
        let depth: u8 = std::env::var("PIMC_DEPTH").ok().and_then(|s| s.parse().ok()).unwrap_or(3);

        let mut sum_pimc = 0u64;
        let mut wins = 0usize;
        let mut sum_elite = 0u64;
        let mut elite_wins = 0usize;
        for deal in &deals {
            // Team 0&2 = strong PIMC, Team 1&3 = Elite.
            let pimc = crate::evaluator::SimulatorBot::Pimc { n_worlds: nw, search_depth: depth };
            let elite = crate::evaluator::SimulatorBot::Heuristic;
            let bots = [pimc.clone(), elite.clone(), pimc.clone(), elite.clone()];
            let mut scratch = vec![0.0f64; 8];
            let mut beh = crate::evaluator::WannBehavior::default();
            let res = crate::evaluator::play_game_sim(deal.hands, deal.trump, 0, &bots, deal.seed, &mut scratch, &mut beh);
            sum_pimc += res.team_02_score as u64;
            if res.team_02_score > 60 { wins += 1; }

            // Reference: Elite vs Elite on the same deal (positional baseline).
            let be = [elite.clone(), elite.clone(), elite.clone(), elite.clone()];
            let mut sc2 = vec![0.0f64; 8];
            let mut bh2 = crate::evaluator::WannBehavior::default();
            let r2 = crate::evaluator::play_game_sim(deal.hands, deal.trump, 0, &be, deal.seed, &mut sc2, &mut bh2);
            sum_elite += r2.team_02_score as u64;
            if r2.team_02_score > 60 { elite_wins += 1; }
        }
        println!("\n=== PIMC({},{}) vs Elite over {} deals ===", nw, depth, n);
        println!("  alpha-beta PIMC:  avg pts = {:.1}/120   win% = {:.1}%", sum_pimc as f64 / n as f64, 100.0 * wins as f64 / n as f64);
        println!("  Elite-self:       avg pts = {:.1}/120   win% = {:.1}%  (positional baseline)", sum_elite as f64 / n as f64, 100.0 * elite_wins as f64 / n as f64);

        // ── Stage-1 teacher: flat Monte-Carlo PIMC with Elite playouts ──
        // Team 0&2 = rollout-PIMC, Team 1&3 = Elite, same deals. This is GATE 1:
        // does the cheap rollout teacher beat Elite by a clear margin (>= ~55%,
        // judged against the Elite-self positional baseline above)?
        let mut sum_roll = 0u64;
        let mut roll_wins = 0usize;
        for deal in &deals {
            let (t02, _t13) = diag_play(
                deal,
                [
                    DiagPolicy::RolloutPimc { worlds: nw },
                    DiagPolicy::Elite,
                    DiagPolicy::RolloutPimc { worlds: nw },
                    DiagPolicy::Elite,
                ],
            );
            sum_roll += t02 as u64;
            if t02 > 60 { roll_wins += 1; }
        }
        println!(
            "  rollout PIMC({}):  avg pts = {:.1}/120   win% = {:.1}%  <-- GATE 1 teacher",
            nw,
            sum_roll as f64 / n as f64,
            100.0 * roll_wins as f64 / n as f64
        );
    }

    #[test]
    #[ignore = "on-demand diagnostic; run explicitly"]
    fn diagnostic_resolver_ceiling() {
        use sueca_solver::heuristic::{resolve_intent, select_card_heuristic};
        use sueca_solver::simulator::SuecaSimulatorGame;
        use crate::train::generate_deals_rust;

        let n_deals = 400;
        let deals = generate_deals_rust(0, n_deals, 7777);

        // --- Part A: static expressiveness coverage on the Elite trajectory ---
        let mut total = 0usize;
        let mut covered = 0usize;          // Elite card reachable by some intent
        let mut lead_total = 0usize;
        let mut lead_covered = 0usize;
        let mut follow_total = 0usize;
        let mut follow_covered = 0usize;
        let mut distinct_hist = [0usize; 4]; // index = # distinct intent cards (1..=3)
        let mut forced = 0usize;            // states with only one legal card

        for deal in &deals {
            let mut game = SuecaSimulatorGame::new(deal.hands, deal.trump, 0);
            while game.state.trick_number < 10 {
                let seat = game.state.current_player;
                let legal = game.state.legal_moves();
                let n_legal = legal.count_ones();
                let leading = game.current_trick_len == 0;

                let e = select_card_heuristic(&game, seat);
                let i0 = resolve_intent(0, &game, seat);
                let i1 = resolve_intent(1, &game, seat);
                let i2 = resolve_intent(2, &game, seat);

                if n_legal == 1 {
                    forced += 1;
                } else {
                    total += 1;
                    let hit = e == i0 || e == i1 || e == i2;
                    if hit { covered += 1; }
                    if leading {
                        lead_total += 1;
                        if hit { lead_covered += 1; }
                    } else {
                        follow_total += 1;
                        if hit { follow_covered += 1; }
                    }
                    // distinct intent cards
                    let mut d = 1;
                    if i1 != i0 { d += 1; }
                    if i2 != i0 && i2 != i1 { d += 1; }
                    distinct_hist[d] += 1;
                }

                game.play_card(e);
            }
        }

        let pct = |a: usize, b: usize| if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 };
        println!("\n========== RESOLVER CEILING DIAGNOSTIC ==========");
        println!("Deals: {}  | non-forced decision states: {}  (forced single-legal: {})", n_deals, total, forced);
        println!("Elite-move expressiveness coverage (some intent == Elite card):");
        println!("  overall : {:.1}%  ({}/{})", pct(covered, total), covered, total);
        println!("  leading : {:.1}%  ({}/{})", pct(lead_covered, lead_total), lead_covered, lead_total);
        println!("  following: {:.1}%  ({}/{})", pct(follow_covered, follow_total), follow_covered, follow_total);
        println!("Distinct intent cards among {{0,1,2}} (non-forced states):");
        for d in 1..=3 {
            println!("  {} distinct: {:.1}%  ({})", d, pct(distinct_hist[d], total), distinct_hist[d]);
        }

        // --- Part B: head-to-head team strength vs the Elite team ---
        // Team 0&2 = candidate policy, Team 1&3 = Elite. Report candidate avg
        // card points (out of 120) and win rate over deals (>60 = win).
        let eval_team = |pol: DiagPolicy, label: &str| {
            let mut sum = 0u64;
            let mut wins = 0usize;
            for deal in &deals {
                let (t02, _t13) = diag_play(deal, [pol, DiagPolicy::Elite, pol, DiagPolicy::Elite]);
                sum += t02 as u64;
                if t02 > 60 { wins += 1; }
            }
            let avg = sum as f64 / n_deals as f64;
            println!("  {:<22} avg pts = {:5.1} / 120   win% = {:5.1}%", label, avg, 100.0 * wins as f64 / n_deals as f64);
        };

        println!("\nHead-to-head (Team 0&2 = policy) vs Elite team (1&3), {} deals:", n_deals);
        eval_team(DiagPolicy::Elite, "Elite (sanity~50%)");
        eval_team(DiagPolicy::Intent(0), "always MAX_FORCE");
        eval_team(DiagPolicy::Intent(1), "always EFFICIENT_WIN");
        eval_team(DiagPolicy::Intent(2), "always EQUITY_BUILDER");
        eval_team(DiagPolicy::MimicOracle, "MimicElite oracle");
        eval_team(DiagPolicy::RolloutOracle, "Rollout best-of-3");

        // --- Part C/D: WANN intent-selection quality on its own trajectory ---
        // Walk each deal with ego=trained WANN (seats 0,2), Elite (1,3). At each
        // ego decision: record the WANN's intent, compute the rollout (PI) score
        // of all 3 intents, and measure agreement with the outcome-optimal intent
        // + the regret (points the WANN leaves on the table by bad selection).
        let champ_path = "/tmp/champ/best_genome_final.json";
        if let Ok((Some(lead_g), Some(follow_g))) = crate::compile_rules::load_genome(champ_path) {
            use sueca_solver::belief::encode_belief_state;
            use sueca_solver::heuristic::{resolve_intent, select_card_heuristic};
            use sueca_solver::rng::LcgRng;
            use sueca_solver::simulator::SuecaSimulatorGame;
            const OS: usize = sueca_solver::constants::OUTPUT_START;

            let lead_net = lead_g.to_rust_wann();
            let follow_net = follow_g.to_rust_wann();
            let sweep = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0];
            let nw = sweep.len();
            let max_nodes = lead_net.num_nodes.max(follow_net.num_nodes);

            let mut rng = LcgRng::new(99);
            let mut scratch = vec![0.0f64; max_nodes * nw];

            let mut champ_mix = [0usize; 3];      // WANN intent histogram
            let mut opt_mix_dec = [0usize; 3];    // optimal intent on strict-unique decisive states
            let mut ego_decisions = 0usize;
            let mut decisive = 0usize;            // intent choice changes rollout outcome
            let mut agree = 0usize;               // WANN picked an outcome-optimal intent
            let mut total_regret = 0i64;          // sum of (best - wann) rollout pts
            let mut team_pts = 0u64;
            let mut wins = 0usize;

            for deal in &deals {
                let mut game = SuecaSimulatorGame::new(deal.hands, deal.trump, 0);
                while game.state.trick_number < 10 {
                    let seat = game.state.current_player;
                    if seat % 2 == 0 {
                        ego_decisions += 1;
                        // --- WANN inference (mirrors SimulatorBot::Wann) ---
                        let belief = encode_belief_state(&game, seat);
                        let leading_flag =
                            (belief[crate::constants::BeliefFeature::AmILeading as usize] - 1.0).abs() < 1e-9;
                        let net = if leading_flag { &lead_net } else { &follow_net };
                        net.forward_batched(&belief, &sweep, &mut scratch);
                        let mut tot = [0.0f64; 3];
                        for w in 0..nw {
                            for k in 0..3 {
                                tot[k] += scratch[(OS + k) * nw + w];
                            }
                        }
                        let maxv = tot[0].max(tot[1]).max(tot[2]);
                        let mut best = [0usize; 3];
                        let mut bc = 0;
                        for k in 0..3 {
                            if (tot[k] - maxv).abs() < 1e-9 {
                                best[bc] = k;
                                bc += 1;
                            }
                        }
                        let w = if bc == 1 { best[0] } else { best[rng.gen_range(0..bc)] };
                        champ_mix[w] += 1;

                        // --- rollout (PI) score of all 3 intents ---
                        let mut sc = [-1i32; 3];
                        for i in 0..3 {
                            let card = resolve_intent(i, &game, seat);
                            let mut g = game;
                            g.play_card(card);
                            let (t02, _t13) = diag_rollout_elite(g);
                            sc[i] = t02 as i32;
                        }
                        let mx = sc[0].max(sc[1]).max(sc[2]);
                        let mn = sc[0].min(sc[1]).min(sc[2]);
                        total_regret += (mx - sc[w]) as i64;
                        if mx > mn {
                            decisive += 1;
                            if sc[w] == mx {
                                agree += 1;
                            }
                            let n_opt = (0..3).filter(|&i| sc[i] == mx).count();
                            if n_opt == 1 {
                                let oi = (0..3).find(|&i| sc[i] == mx).unwrap();
                                opt_mix_dec[oi] += 1;
                            }
                        }
                        let card = resolve_intent(w, &game, seat);
                        game.play_card(card);
                    } else {
                        let c = select_card_heuristic(&game, seat);
                        game.play_card(c);
                    }
                }
                team_pts += game.state.team_02_score as u64;
                if game.state.team_02_score > 60 {
                    wins += 1;
                }
            }

            let mixp = |a: [usize; 3]| {
                let t = a.iter().sum::<usize>().max(1);
                (
                    100.0 * a[0] as f64 / t as f64,
                    100.0 * a[1] as f64 / t as f64,
                    100.0 * a[2] as f64 / t as f64,
                )
            };
            let (c0, c1, c2) = mixp(champ_mix);
            let (d0, d1, d2) = mixp(opt_mix_dec);
            println!("  TRAINED champion       avg pts = {:5.1} / 120   win% = {:5.1}%", team_pts as f64 / n_deals as f64, 100.0 * wins as f64 / n_deals as f64);
            println!("    WANN intent mix     : MAX_FORCE {:.0}%  EFFICIENT {:.0}%  EQUITY {:.0}%", c0, c1, c2);
            println!("    OPTIMAL mix (strict-decisive only): MAX_FORCE {:.0}%  EFFICIENT {:.0}%  EQUITY {:.0}%", d0, d1, d2);
            println!("    ego decisions: {}  | decisive (intent matters): {} ({:.0}%)", ego_decisions, decisive, pct(decisive, ego_decisions));
            println!("    WANN picks an optimal intent on decisive states: {:.0}%  (chance ~ {:.0}%)", pct(agree, decisive), 100.0/3.0);
            println!("    avg PI regret from intent choice: {:.1} pts / game", total_regret as f64 / n_deals as f64);
        } else {
            println!("  (trained champion not found at {} — skipped)", champ_path);
        }
        println!("=================================================\n");
    }
}
