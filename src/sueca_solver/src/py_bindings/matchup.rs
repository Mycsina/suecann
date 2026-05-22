use super::wann::PyWannNetwork;
use crate::evaluator;
use crate::pimc;
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(FromPyObject)]
pub struct PyDeal {
    pub hands: Vec<Vec<u8>>,
    pub trump: u8,
    pub seed: u64,
}

impl PyDeal {
    pub fn to_rust(&self) -> evaluator::EvaluatorDeal {
        let mut hands = [0u64; 4];
        for s in 0..4 {
            for &card in &self.hands[s] {
                hands[s] |= 1u64 << card;
            }
        }
        evaluator::EvaluatorDeal {
            hands,
            trump: self.trump,
            seed: self.seed,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (
    deals,
    bot_a_type,
    bot_a_network,
    bot_b_type,
    bot_b_network,
    sweep_weights,
    base_seed
))]
#[allow(clippy::too_many_arguments)]
pub fn run_matchup_rust(
    py: Python,
    deals: Vec<PyDeal>,
    bot_a_type: i32,
    bot_a_network: Option<PyWannNetwork>,
    bot_b_type: i32,
    bot_b_network: Option<PyWannNetwork>,
    sweep_weights: Vec<f64>,
    base_seed: u64,
) -> PyResult<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)> {
    let rust_deals: Vec<evaluator::EvaluatorDeal> = deals.iter().map(|d| d.to_rust()).collect();
    let opt_network_a = bot_a_network.map(|n| n.inner);
    let opt_network_b = bot_b_network.map(|n| n.inner);

    let max_nodes = [
        opt_network_a.as_ref().map(|n| n.num_nodes).unwrap_or(0),
        opt_network_b.as_ref().map(|n| n.num_nodes).unwrap_or(0),
    ]
    .iter()
    .max()
    .copied()
    .unwrap_or(27)
    .max(27);

    let results = py.allow_threads(|| {
        rust_deals
            .into_par_iter()
            .map(|deal| {
                let mut scratchpad = vec![0.0f64; max_nodes];

                let bot_a = match bot_a_type {
                    -1 => evaluator::SimulatorBot::Pimc {
                        n_worlds: 10,
                        search_depth: 1,
                    },
                    0 => evaluator::SimulatorBot::Random,
                    1 => evaluator::SimulatorBot::Heuristic,
                    _ => {
                        if let Some(ref n) = opt_network_a {
                            evaluator::SimulatorBot::Wann {
                                network: n,
                                weights: &sweep_weights,
                            }
                        } else {
                            evaluator::SimulatorBot::Random
                        }
                    }
                };

                let bot_b = match bot_b_type {
                    -1 => evaluator::SimulatorBot::Pimc {
                        n_worlds: 10,
                        search_depth: 1,
                    },
                    0 => evaluator::SimulatorBot::Random,
                    1 => evaluator::SimulatorBot::Heuristic,
                    _ => {
                        if let Some(ref n) = opt_network_b {
                            evaluator::SimulatorBot::Wann {
                                network: n,
                                weights: &sweep_weights,
                            }
                        } else {
                            evaluator::SimulatorBot::Random
                        }
                    }
                };

                let bots1 = [bot_a.clone(), bot_b.clone(), bot_a.clone(), bot_b.clone()];
                let seed1 = base_seed + deal.seed + 1000;
                let res1 = evaluator::play_game_sim(
                    deal.hands,
                    deal.trump,
                    0,
                    &bots1,
                    seed1,
                    &mut scratchpad,
                );

                let bots2 = [bot_b.clone(), bot_a.clone(), bot_b.clone(), bot_a.clone()];
                let seed2 = base_seed + deal.seed + 2000;
                let res2 = evaluator::play_game_sim(
                    deal.hands,
                    deal.trump,
                    0,
                    &bots2,
                    seed2,
                    &mut scratchpad,
                );

                (
                    res1.team_02_score,
                    res2.team_13_score,
                    res1.team_02_game_points,
                    res2.team_13_game_points,
                    res1.team_13_game_points,
                    res2.team_02_game_points,
                )
            })
            .collect::<Vec<(u8, u8, u8, u8, u8, u8)>>()
    });

    let mut score_a1 = Vec::with_capacity(results.len());
    let mut score_a2 = Vec::with_capacity(results.len());
    let mut gpts_a1 = Vec::with_capacity(results.len());
    let mut gpts_a2 = Vec::with_capacity(results.len());
    let mut gpts_b1 = Vec::with_capacity(results.len());
    let mut gpts_b2 = Vec::with_capacity(results.len());

    for (s1, s2, g1, g2, gb1, gb2) in results {
        score_a1.push(s1);
        score_a2.push(s2);
        gpts_a1.push(g1);
        gpts_a2.push(g2);
        gpts_b1.push(gb1);
        gpts_b2.push(gb2);
    }

    Ok((score_a1, score_a2, gpts_a1, gpts_a2, gpts_b1, gpts_b2))
}

#[pyfunction]
#[pyo3(signature = (
    deals,
    bot_a_type,
    bot_a_network,
    bot_b_type,
    bot_b_network,
    sweep_weights,
    base_seed
))]
pub fn run_snr_matchup_rust(
    py: Python,
    deals: Vec<PyDeal>,
    bot_a_type: i32,
    bot_a_network: Option<PyWannNetwork>,
    bot_b_type: i32,
    bot_b_network: Option<PyWannNetwork>,
    sweep_weights: Vec<f64>,
    base_seed: u64,
) -> PyResult<(Vec<i16>, Vec<i16>)> {
    let rust_deals: Vec<evaluator::EvaluatorDeal> = deals.iter().map(|d| d.to_rust()).collect();
    let opt_network_a = bot_a_network.map(|n| n.inner);
    let opt_network_b = bot_b_network.map(|n| n.inner);

    let max_nodes = [
        opt_network_a.as_ref().map(|n| n.num_nodes).unwrap_or(0),
        opt_network_b.as_ref().map(|n| n.num_nodes).unwrap_or(0),
    ]
    .iter()
    .max()
    .copied()
    .unwrap_or(27)
    .max(27);

    let results = py.allow_threads(|| {
        rust_deals
            .into_par_iter()
            .map(|deal| {
                let mut scratchpad = vec![0.0f64; max_nodes];
                let mut deal_gp_deltas = Vec::with_capacity(4);
                let mut deal_cp_deltas = Vec::with_capacity(4);

                let bot_a = match bot_a_type {
                    -1 => evaluator::SimulatorBot::Pimc {
                        n_worlds: 10,
                        search_depth: 1,
                    },
                    0 => evaluator::SimulatorBot::Random,
                    1 => evaluator::SimulatorBot::Heuristic,
                    _ => {
                        if let Some(ref n) = opt_network_a {
                            evaluator::SimulatorBot::Wann {
                                network: n,
                                weights: &sweep_weights,
                            }
                        } else {
                            evaluator::SimulatorBot::Random
                        }
                    }
                };

                let bot_b = match bot_b_type {
                    -1 => evaluator::SimulatorBot::Pimc {
                        n_worlds: 10,
                        search_depth: 1,
                    },
                    0 => evaluator::SimulatorBot::Random,
                    1 => evaluator::SimulatorBot::Heuristic,
                    _ => {
                        if let Some(ref n) = opt_network_b {
                            evaluator::SimulatorBot::Wann {
                                network: n,
                                weights: &sweep_weights,
                            }
                        } else {
                            evaluator::SimulatorBot::Random
                        }
                    }
                };

                let random_bot = evaluator::SimulatorBot::Random;

                for rot in 0..4 {
                    let rotated_hands = evaluator::rotate_hands(&deal.hands, rot);
                    let adj_first = (rot as u8) % 4;
                    let game_seed = base_seed + deal.seed + (rot as u64) * 10000;

                    let bots_a = [
                        bot_a.clone(),
                        random_bot.clone(),
                        random_bot.clone(),
                        random_bot.clone(),
                    ];
                    let res_a = evaluator::play_game_sim(
                        rotated_hands,
                        deal.trump,
                        adj_first,
                        &bots_a,
                        game_seed,
                        &mut scratchpad,
                    );

                    let bots_b = [
                        bot_b.clone(),
                        random_bot.clone(),
                        random_bot.clone(),
                        random_bot.clone(),
                    ];
                    let res_b = evaluator::play_game_sim(
                        rotated_hands,
                        deal.trump,
                        adj_first,
                        &bots_b,
                        game_seed,
                        &mut scratchpad,
                    );

                    let gp_delta =
                        (res_a.team_02_game_points as i16) - (res_b.team_02_game_points as i16);
                    let cp_delta = (res_a.team_02_score as i16) - (res_b.team_02_score as i16);

                    deal_gp_deltas.push(gp_delta);
                    deal_cp_deltas.push(cp_delta);
                }

                (deal_gp_deltas, deal_cp_deltas)
            })
            .collect::<Vec<(Vec<i16>, Vec<i16>)>>()
    });

    let mut all_gp_deltas = Vec::with_capacity(results.len() * 4);
    let mut all_cp_deltas = Vec::with_capacity(results.len() * 4);
    for (gps, cps) in results {
        all_gp_deltas.extend(gps);
        all_cp_deltas.extend(cps);
    }

    Ok((all_gp_deltas, all_cp_deltas))
}

fn generate_deals_rust_solver(
    gen: u64,
    n_deals: usize,
    base_seed: u64,
) -> Vec<evaluator::EvaluatorDeal> {
    let seed = base_seed + gen;
    let mut deals = Vec::with_capacity(n_deals);

    for i in 0..n_deals {
        let deal_seed = seed * 1000 + i as u64;
        let mut deal_rng = evaluator::LcgRng::new(deal_seed);

        let mut deck: Vec<u8> = (0..40).collect();
        for idx in (1..40).rev() {
            let j = deal_rng.gen_range(0..idx + 1);
            deck.swap(idx, j);
        }

        let mut hands = [0u64; 4];
        for player in 0..4 {
            for card_idx in 0..10 {
                let card = deck[player * 10 + card_idx];
                hands[player] |= 1u64 << card;
            }
        }

        let trump = deal_rng.gen_range(0..4) as u8;
        deals.push(evaluator::EvaluatorDeal {
            hands,
            trump,
            seed: deal_seed,
        });
    }
    deals
}

#[pyfunction]
pub fn generate_expert_dataset_rust(
    py: Python,
    n_worlds: usize,
    search_depth: u8,
    target_count: usize,
    base_seed: u64,
) -> PyResult<(Vec<f64>, Vec<u8>, Vec<u8>)> {
    let total_needed = target_count * 5;
    let mut dataset_states = Vec::with_capacity(total_needed * 21);
    let mut dataset_intents = Vec::with_capacity(total_needed);
    let mut dataset_legal_masks = Vec::with_capacity(total_needed);

    let mut intent_counts = [0usize; 5];
    let mut total_saved = 0;

    let mut rng = evaluator::LcgRng::new(base_seed);
    let mut deal_gen = 0u64;

    py.allow_threads(|| {
        while total_saved < total_needed {
            let deals = generate_deals_rust_solver(deal_gen, 100, rng.next_u64());
            deal_gen += 1;

            for deal in deals {
                let mut game = evaluator::SuecaSimulatorGame::new(deal.hands, deal.trump, 0);

                let target_trick = rng.gen_range(2..9) as u8;

                while game.state.trick_number < target_trick && game.state.trick_number < 10 {
                    let seat = game.state.current_player;
                    let mut scratchpad = vec![0.0; 27];
                    let mut illegal_count = 0;
                    let card = evaluator::SimulatorBot::Heuristic.select_card(
                        &game,
                        seat,
                        &mut rng,
                        &mut scratchpad,
                        &mut illegal_count,
                    );
                    game.play_card(card);
                }

                if game.state.trick_number >= 10 {
                    continue;
                }

                let num_to_play = rng.gen_range(0..4);
                let mut broke = false;
                for _ in 0..num_to_play {
                    if game.state.trick_number >= 10 {
                        broke = true;
                        break;
                    }
                    let seat = game.state.current_player;
                    let mut scratchpad = vec![0.0; 27];
                    let mut illegal_count = 0;
                    let card = evaluator::SimulatorBot::Heuristic.select_card(
                        &game,
                        seat,
                        &mut rng,
                        &mut scratchpad,
                        &mut illegal_count,
                    );
                    game.play_card(card);
                }

                if broke || game.state.trick_number >= 10 {
                    continue;
                }

                let seat = game.state.current_player;
                let legal_moves = game.state.legal_moves();
                if legal_moves.count_ones() <= 1 {
                    continue;
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
                    game.current_trick[0] / 10
                } else {
                    4
                };
                let current_trick_cards = &game.current_trick[..game.current_trick_len];

                let pimc_seed = deal.seed + rng.next_u64();
                let evs = pimc::solve_pimc(
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
                    n_worlds,
                    search_depth,
                    pimc_seed,
                );

                let mut best_card = 40u8;
                let mut max_ev = f64::NEG_INFINITY;
                for (card, ev) in evs {
                    if ev > max_ev {
                        max_ev = ev;
                        best_card = card;
                    }
                }

                if best_card == 40 {
                    continue;
                }

                let mut matching_intents = Vec::new();
                for intent in 0..5 {
                    let (card_intent, was_illegal) = evaluator::resolve_intent(intent, &game, seat);
                    if !was_illegal && card_intent == best_card {
                        matching_intents.push(intent);
                    }
                }

                if matching_intents.is_empty() {
                    continue;
                }

                let mut underfilled = Vec::new();
                for &intent in &matching_intents {
                    if intent_counts[intent] < target_count {
                        underfilled.push(intent);
                    }
                }

                if underfilled.is_empty() {
                    continue;
                }

                let mut target_intent = underfilled[0];
                let mut min_count = intent_counts[target_intent];
                for &intent in &underfilled[1..] {
                    if intent_counts[intent] < min_count {
                        min_count = intent_counts[intent];
                        target_intent = intent;
                    }
                }

                let mut legal_mask = 0u8;
                for intent in 0..5 {
                    let (_, was_illegal) = evaluator::resolve_intent(intent, &game, seat);
                    if !was_illegal {
                        legal_mask |= 1 << intent;
                    }
                }

                let belief = evaluator::encode_belief_state(&game, seat);

                dataset_states.extend_from_slice(&belief);
                dataset_intents.push(target_intent as u8);
                dataset_legal_masks.push(legal_mask);

                intent_counts[target_intent] += 1;
                total_saved += 1;

                if total_saved % 1000 == 0 || total_saved == total_needed {
                    println!(
                        "Progress: {}/{} states saved. Counts: {:?}",
                        total_saved, total_needed, intent_counts
                    );
                }

                if total_saved >= total_needed {
                    break;
                }
            }
        }
    });

    Ok((dataset_states, dataset_intents, dataset_legal_masks))
}
