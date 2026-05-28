use crate::wann_network::RustWannNetwork;

use sueca_solver::belief::encode_belief_state;
use sueca_solver::heuristic::{
    resolve_intent, select_card_heuristic, select_card_heuristic_old, select_card_random,
};
use sueca_solver::rng::LcgRng;
use sueca_solver::simulator::SuecaSimulatorGame;

#[derive(Debug, Clone, Default)]
pub struct WannBehavior {
    pub intent_counts: [usize; 4],
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
        network: &'a RustWannNetwork,
        weights: &'a [f64],
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
            SimulatorBot::Wann { network, weights } => {
                let belief = encode_belief_state(game, seat);

                // Weight sweep output averaging
                let mut sum_outputs = [0.0f64; sueca_solver::constants::OUTPUT_COUNT];
                for &w in *weights {
                    network.forward(&belief, w, scratchpad);
                    for i in 0..sueca_solver::constants::OUTPUT_COUNT {
                        sum_outputs[i] += scratchpad[sueca_solver::constants::OUTPUT_START + i];
                    }
                }

                let mut mean_outputs = [0.0f64; sueca_solver::constants::OUTPUT_COUNT];
                for i in 0..sueca_solver::constants::OUTPUT_COUNT {
                    mean_outputs[i] = sum_outputs[i] / (weights.len() as f64);
                }

                // Apply structural activation baseline penalty
                let mut adjusted_outputs = mean_outputs;
                adjusted_outputs[3] -= 0.25;

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
                    if chosen_intent < 4 {
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
                for (card, ev) in evs {
                    if ev > max_ev {
                        max_ev = ev;
                        best_card = card;
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
    hof_networks: &'a [RustWannNetwork],
    sweep_weights: &'a [f64],
) -> SimulatorBot<'a> {
    match bot_type {
        -1 => SimulatorBot::Pimc {
            n_worlds: 5,
            search_depth: 1,
        },
        0 => SimulatorBot::Random,
        1 => SimulatorBot::OldHeuristic,
        3 => SimulatorBot::Heuristic,
        t if t >= 10 => {
            let idx = (t - 10) as usize;
            SimulatorBot::Wann {
                network: &hof_networks[idx],
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
    candidate: &RustWannNetwork,
    baseline_bot_type: i32,
    partner_bot_type: i32,
    opp1_bot_type: i32,
    opp2_bot_type: i32,
    hof_networks: &[RustWannNetwork],
    sweep_weights: &[f64],
    deals: &[EvaluatorDeal],
    base_seed: u64,
    scratchpad: &mut [f64],
) -> (f64, WannBehavior) {
    let mut total_score_candidate = 0.0;
    let mut total_score_baseline = 0.0;
    let mut accum_behavior = WannBehavior::default();

    let first_player = 0;

    let partner = get_bot_from_type(partner_bot_type, hof_networks, sweep_weights);
    let opp1 = get_bot_from_type(opp1_bot_type, hof_networks, sweep_weights);
    let opp2 = get_bot_from_type(opp2_bot_type, hof_networks, sweep_weights);
    let baseline = get_bot_from_type(baseline_bot_type, hof_networks, sweep_weights);

    let candidate_bot = SimulatorBot::Wann {
        network: candidate,
        weights: sweep_weights,
    };

    for deal in deals {
        for rot in 0..4 {
            let rotated_hands = rotate_hands(&deal.hands, rot);
            let adj_first = rotate_first_player(first_player, rot);
            let game_seed = base_seed + deal.seed + (rot as u64) * 10000;

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
                game_seed,
                scratchpad,
                &mut game_behavior,
            );

            for i in 0..4 {
                accum_behavior.intent_counts[i] += game_behavior.intent_counts[i];
            }
            accum_behavior.total_lead_points += game_behavior.total_lead_points;
            accum_behavior.count_leads += game_behavior.count_leads;
            accum_behavior.total_actions += game_behavior.total_actions;

            // --- Play with baseline ---
            let bots_baseline = [
                baseline.clone(),
                opp1.clone(),
                partner.clone(),
                opp2.clone(),
            ];

            let mut dummy_behavior = WannBehavior::default();
            let result_baseline = play_game_sim(
                rotated_hands,
                deal.trump,
                adj_first,
                &bots_baseline,
                game_seed,
                scratchpad,
                &mut dummy_behavior,
            );

            total_score_candidate += result_candidate.team_02_score as f64;
            total_score_baseline += result_baseline.team_02_score as f64;
        }
    }

    let num_games = (deals.len() * 4) as f64;
    let avg_candidate = total_score_candidate / num_games;
    let avg_baseline = total_score_baseline / num_games;
    let avg_delta = avg_candidate - avg_baseline;

    (avg_delta, accum_behavior)
}
