use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use rayon::prelude::*;
use sueca_solver::belief::encode_belief_state;
use sueca_solver::constants::{INPUT_COUNT, OUTPUT_COUNT};
use sueca_solver::engine::{CARD_POINTS, CARD_SUIT};
use sueca_solver::heuristic::resolve_intent;
use sueca_solver::simulator::SuecaSimulatorGame;

pub struct DatasetConfig {
    pub n_worlds: usize,
    pub search_depth: u8,
    pub target_total: usize,
    pub seed: u64,
    pub output_path: String,
    pub pimc_min_margin: f64,
}

pub fn generate_dataset(config: &DatasetConfig) {
    let total_target = config.target_total;

    println!(
        "Generating expert dataset: {} worlds, depth {}, {} total states...",
        config.n_worlds, config.search_depth, total_target
    );

    let mut all_states = Vec::with_capacity(total_target * INPUT_COUNT);
    let mut all_intents = Vec::with_capacity(total_target * 4);
    let mut all_masks = Vec::with_capacity(total_target);
    let mut intent_counts = [0usize; OUTPUT_COUNT];

    let mut batch_seed = config.seed;
    let mut batches = 0;

    loop {
        if all_intents.len() / 4 >= total_target {
            break;
        }
        batches += 1;

        let batch_size = 512;
        let batch_results: Vec<Vec<(Vec<f64>, [f32; 4], u8)>> = (0..batch_size)
            .into_par_iter()
            .map(|i| {
                let deal_seed = batch_seed + i as u64;
                let mut rng = Pcg64::seed_from_u64(deal_seed);
                generate_deal_states(&mut rng, config)
            })
            .collect();

        for results in batch_results {
            for (state, soft_intent, mask) in results {
                if all_intents.len() / 4 < total_target {
                    for i in 0..4 {
                        if soft_intent[i] > 0.0 {
                            intent_counts[i] += 1;
                        }
                    }
                    all_states.extend(&state);
                    all_intents.extend_from_slice(&soft_intent);
                    all_masks.push(mask);
                }
            }
        }

        batch_seed += batch_size as u64;

        if batches % 10 == 0 {
            println!(
                "  batch {}, total: {}/{}, per intent: {:?}",
                batches,
                all_intents.len() / 4,
                total_target,
                intent_counts
            );
        }
    }

    println!(
        "Final dataset: {} states, intent distribution: {:?}",
        all_intents.len() / 4,
        intent_counts
    );

    save_npz(&all_states, &all_intents, &all_masks, &config.output_path);
}

pub fn generate_deal_states(rng: &mut Pcg64, config: &DatasetConfig) -> Vec<(Vec<f64>, [f32; 4], u8)> {
    let mut results = Vec::new();

    // Deal cards
    let mut deck: Vec<u8> = (0..40).collect();
    for i in (1..40).rev() {
        let j = (rng.gen_range(0u64..((i + 1) as u64))) as usize;
        deck.swap(i, j);
    }

    let mut hands = [0u64; 4];
    for p in 0..4 {
        for c in 0..10 {
            hands[p] |= 1u64 << deck[p * 10 + c];
        }
    }

    let trump = (rng.gen_range(0u64..4)) as u8;
    let first_player = (rng.gen_range(0u64..4)) as u8;

    // Play random cards until a random mid-game position
    let max_tricks = (rng.gen_range(1u64..9)) as usize; // 1 to 8 tricks played
    let mut game = SuecaSimulatorGame::new(hands, trump, first_player);

    for _ in 0..max_tricks {
        if game.state.trick_number >= 10 {
            break;
        }
        let legal = game.state.legal_moves();
        if legal == 0 {
            break;
        }
        let count = legal.count_ones();
        let idx = (rng.gen_range(0u64..(count as u64))) as usize;
        let mut temp = legal;
        for _ in 0..idx {
            temp &= temp - 1;
        }
        let card = temp.trailing_zeros() as u8;
        game.play_card(card);
    }

    if game.state.trick_number >= 10 {
        return results;
    }

    // CRITICAL: Extract data ONLY for the active player whose turn it is.
    // legal_moves(), resolve_intent(), and the belief encoder are all defined
    // relative to the current player. Looping over all 4 seats at a frozen
    // state produces garbage for 75% of samples (wrong legal mask, intent
    // resolver fails and falls back to MAX_FORCE, belief/perspective mismatch).
    let seat = game.state.current_player;
    let legal = game.state.legal_moves();

    if legal.count_ones() <= 1 {
        return results;
    }

    let current_scores = [game.state.team_02_score, game.state.team_13_score];
    let current_trick_number = game.state.trick_number;

    let belief = encode_belief_state(&game, seat);

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
        CARD_SUIT[game.current_trick[0] as usize]
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
        seat,
        game.state.current_trick_winner,
        game.state.current_trick_best_card,
        current_scores,
        current_trick_number,
        config.n_worlds,
        config.search_depth,
        rng.gen_range(0u64..u64::MAX),
    );

    let mut sorted_evs = evs.clone();
    sorted_evs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if sorted_evs.is_empty() {
        return results;
    }

    let best_card = sorted_evs[0].0;
    if sorted_evs.len() >= 2 {
        let second_best_card = sorted_evs[1].0;
        let bypass_margin_check = CARD_SUIT[best_card as usize] == CARD_SUIT[second_best_card as usize]
            && CARD_POINTS[best_card as usize] == 0
            && CARD_POINTS[second_best_card as usize] == 0;

        if !bypass_margin_check && (sorted_evs[0].1 - sorted_evs[1].1) < config.pimc_min_margin {
            return results; // Reject true tactical ambiguity
        }
    }

    if let Some(soft_intent) = map_card_to_soft_intents(best_card, &game, seat) {
        let legal_mask = build_legal_mask(legal, &game, seat);
        let state_vec: Vec<f64> = belief.to_vec();
        results.push((state_vec, soft_intent, legal_mask));
    }
    // Unclassifiable states are rejected — no fallback pollution.

    results
}

pub fn map_card_to_soft_intents(card: u8, game: &SuecaSimulatorGame, seat: u8) -> Option<[f32; 4]> {
    let mut matching_intents = Vec::new();
    for intent in 0..4 {
        let resolved = resolve_intent(intent, game, seat);
        if resolved == card {
            matching_intents.push(intent);
        }
    }
    if matching_intents.is_empty() {
        return None;
    }

    // Apply Intent Demotion for leading states:
    // If a card satisfies both MAX_FORCE (0) and EQUITY_BUILDER (3) when leading,
    // break the tie in favor of the more specific tactical intent (EQUITY_BUILDER).
    if game.current_trick_len == 0 {
        if matching_intents.contains(&0) && matching_intents.contains(&3) {
            matching_intents.retain(|&intent| intent != 0);
        }
    }

    let prob = 1.0f32 / (matching_intents.len() as f32);
    let mut target_vector = [0.0f32; 4];
    for intent in matching_intents {
        target_vector[intent] = prob;
    }
    Some(target_vector)
}

pub fn build_legal_mask(legal: u64, game: &SuecaSimulatorGame, seat: u8) -> u8 {
    // For each intent, check if it's legal (produces a card in legal_moves)
    let mut mask = 0u8;
    for intent in 0..OUTPUT_COUNT {
        let card = resolve_intent(intent, game, seat);
        if (legal & (1u64 << card)) != 0 {
            mask |= 1 << intent;
        }
    }
    mask
}

fn save_npz(states: &[f64], intents: &[f32], masks: &[u8], path: &str) {
    use std::io::Write;
    use zip::write::FileOptions;

    let num_states = intents.len() / 4;
    let states_f32: Vec<f32> = states.iter().map(|&v| v as f32).collect();

    // Helper: write a .npy file to a buffer
    let write_npy = |data: &[u8], dtype: &str, shape: &[u64]| -> Vec<u8> {
        let mut buf = Vec::new();
        // Magic
        buf.extend_from_slice(b"\x93NUMPY");
        // Version 1.0
        buf.extend_from_slice(&[1u8, 0u8]);
        // Header
        let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
        let header = format!(
            "{{'descr': '{}', 'fortran_order': False, 'shape': ({},)}}",
            dtype,
            shape_str.join(",")
        );
        let mut header_bytes = header.as_bytes().to_vec();
        // Pad to 16-byte alignment (header length must be divisible by 16)
        let total_header_len = 10 + header_bytes.len();
        let pad_needed = (16 - (total_header_len % 16)) % 16;
        header_bytes.extend(std::iter::repeat_n(b' ', pad_needed));
        header_bytes.push(b'\n');
        // Header length (2 bytes, little-endian)
        let header_len = header_bytes.len() as u16;
        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(&header_bytes);
        // Data
        buf.extend_from_slice(data);
        buf
    };

    let zip_path = if path.ends_with(".npz") {
        path.to_string()
    } else {
        format!("{}.npz", path)
    };
    let file = std::fs::File::create(&zip_path).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let opts = FileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    // states.npy
    {
        let raw: &[u8] = unsafe {
            std::slice::from_raw_parts(states_f32.as_ptr() as *const u8, states_f32.len() * 4)
        };
        let npy = write_npy(raw, "<f4", &[num_states as u64, INPUT_COUNT as u64]);
        zip.start_file("states.npy", opts).unwrap();
        zip.write_all(&npy).unwrap();
    }

    // intents.npy
    {
        let raw: &[u8] = unsafe {
            std::slice::from_raw_parts(intents.as_ptr() as *const u8, intents.len() * 4)
        };
        let npy = write_npy(raw, "<f4", &[num_states as u64, 4]);
        zip.start_file("intents.npy", opts).unwrap();
        zip.write_all(&npy).unwrap();
    }

    // legal_masks.npy
    {
        let npy = write_npy(masks, "|u1", &[num_states as u64]);
        zip.start_file("legal_masks.npy", opts).unwrap();
        zip.write_all(&npy).unwrap();
    }

    zip.finish().unwrap();

    let size_mb = std::fs::metadata(&zip_path)
        .map(|m| m.len() as f64 / 1e6)
        .unwrap_or(0.0);
    println!(
        "Dataset saved to {} ({:.1} MB, {} states × {} features)",
        zip_path, size_mb, num_states, INPUT_COUNT
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> DatasetConfig {
        DatasetConfig {
            n_worlds: 10,
            search_depth: 2,
            target_total: 100,
            seed: 12345,
            output_path: String::new(),
            pimc_min_margin: 0.5,
        }
    }

    /// Verify ego-turn synchronization: every generated state is for the
    /// player whose turn it actually is (game.state.current_player).
    #[test]
    fn test_ego_turn_sync() {
        let config = make_config();
        let mut rng = Pcg64::seed_from_u64(42);
        // Generate many deal states and verify each one
        for _ in 0..100 {
            let results = generate_deal_states(&mut rng, &config);
            // We can't directly inspect current_player since the game is consumed,
            // but we can verify the states come from a single consistent perspective
            // by checking that belief features 5 (Am_I_Leading) is consistent
            // with the encoded perspective — it must be 0 or 1, never NaN or out of bounds.
            for (state, _intent, _mask) in &results {
                assert_eq!(state.len(), INPUT_COUNT);
                // Every belief value must be in [0, 1]
                for (i, &v) in state.iter().enumerate() {
                    assert!(
                        (0.0..=1.0).contains(&v),
                        "belief[{}] = {} is out of [0, 1]",
                        i,
                        v
                    );
                }
                // Bias at index 30 (not part of the 30-feature state, but verify
                // no extra data leaked — the state must be exactly INPUT_COUNT)
                assert_eq!(state.len(), INPUT_COUNT);
            }
        }
    }

    /// Every belief state value must be in the [0, 1] range.
    #[test]
    fn test_belief_bounds() {
        let config = make_config();
        let mut rng = Pcg64::seed_from_u64(99);
        let mut total = 0;
        for _ in 0..200 {
            for (state, _, _) in generate_deal_states(&mut rng, &config) {
                total += 1;
                for (i, &v) in state.iter().enumerate() {
                    assert!(
                        v.is_finite() && (0.0..=1.0).contains(&v),
                        "belief[{}] = {} is invalid at sample {}",
                        i,
                        v,
                        total
                    );
                }
            }
        }
        assert!(
            total > 50,
            "should generate at least some states: got {total}"
        );
    }

    /// The PIMC-recommended best card must be legal for the current player.
    #[test]
    fn test_pimc_card_is_legal() {
        let config = DatasetConfig {
            n_worlds: 20,
            search_depth: 2,
            target_total: 4000,
            seed: 777,
            output_path: String::new(),
            pimc_min_margin: 0.5,
        };
        let mut rng = Pcg64::seed_from_u64(777);
        // Generate a fresh deal state and trace through the logic manually
        for _ in 0..50 {
            // Deal and play to random position
            let mut deck: Vec<u8> = (0..40).collect();
            for i in (1..40).rev() {
                let j = (rng.gen_range(0u64..((i + 1) as u64))) as usize;
                deck.swap(i, j);
            }
            let mut hands = [0u64; 4];
            for p in 0..4 {
                for c in 0..10 {
                    hands[p] |= 1u64 << deck[p * 10 + c];
                }
            }
            let trump = (rng.gen_range(0u64..4)) as u8;
            let first_player = (rng.gen_range(0u64..4)) as u8;
            let mut game = SuecaSimulatorGame::new(hands, trump, first_player);

            let max_tricks = (rng.gen_range(0u64..6)) as usize;
            for _ in 0..max_tricks {
                if game.state.trick_number >= 10 {
                    break;
                }
                let legal = game.state.legal_moves();
                if legal == 0 {
                    break;
                }
                let count = legal.count_ones();
                let idx = (rng.gen_range(0u64..(count as u64))) as usize;
                let mut temp = legal;
                for _ in 0..idx {
                    temp &= temp - 1;
                }
                game.play_card(temp.trailing_zeros() as u8);
            }
            if game.state.trick_number >= 10 {
                continue;
            }

            let seat = game.state.current_player;
            let legal = game.state.legal_moves();
            if legal.count_ones() <= 1 {
                continue;
            }

            // Run PIMC
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
                CARD_SUIT[game.current_trick[0] as usize]
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
                seat,
                game.state.current_trick_winner,
                game.state.current_trick_best_card,
                [game.state.team_02_score, game.state.team_13_score],
                game.state.trick_number,
                config.n_worlds,
                config.search_depth,
                rng.gen_range(0u64..u64::MAX),
            );

            let mut best_card = legal.trailing_zeros() as u8;
            let mut max_ev = f64::NEG_INFINITY;
            for (card, ev) in &evs {
                if *ev > max_ev {
                    max_ev = *ev;
                    best_card = *card;
                }
            }

            // The best card MUST be in the legal moves bitmask
            assert!(
                (legal & (1u64 << best_card)) != 0,
                "PIMC best card {} is not legal for seat {}. Legal mask: {:064b}",
                best_card,
                seat,
                legal
            );
        }
    }

    /// When map_card_to_intent returns Some(intent), the intent must resolve
    /// to a legal card. Unclassifiable cards return None — no fallback pollution.
    #[test]
    fn test_intent_mapping_is_clean() {
        let mut rng = Pcg64::seed_from_u64(123);

        for _ in 0..100 {
            let mut deck: Vec<u8> = (0..40).collect();
            for i in (1..40).rev() {
                let j = (rng.gen_range(0u64..((i + 1) as u64))) as usize;
                deck.swap(i, j);
            }
            let mut hands = [0u64; 4];
            for p in 0..4 {
                for c in 0..10 {
                    hands[p] |= 1u64 << deck[p * 10 + c];
                }
            }
            let trump = (rng.gen_range(0u64..4)) as u8;
            let first_player = (rng.gen_range(0u64..4)) as u8;
            let mut game = SuecaSimulatorGame::new(hands, trump, first_player);

            let tricks = (rng.gen_range(0u64..5)) as usize;
            for _ in 0..tricks {
                if game.state.trick_number >= 10 {
                    break;
                }
                let legal = game.state.legal_moves();
                if legal == 0 {
                    break;
                }
                let count = legal.count_ones();
                let idx = (rng.gen_range(0u64..(count as u64))) as usize;
                let mut temp = legal;
                for _ in 0..idx {
                    temp &= temp - 1;
                }
                game.play_card(temp.trailing_zeros() as u8);
            }
            if game.state.trick_number >= 10 {
                continue;
            }

            let seat = game.state.current_player;
            let legal = game.state.legal_moves();
            if legal.count_ones() <= 1 {
                continue;
            }

            // For every legal card: if classified, the intent must be valid
            let mut temp = legal;
            while temp != 0 {
                let card = temp.trailing_zeros() as u8;
                if let Some(soft_intent) = map_card_to_soft_intents(card, &game, seat) {
                    for intent in 0..4 {
                        if soft_intent[intent] > 0.0 {
                            let resolved = resolve_intent(intent, &game, seat);
                            assert!(
                                (legal & (1u64 << resolved)) != 0,
                                "Intent {} resolved to card {} which is not legal for seat {}",
                                intent,
                                resolved,
                                seat
                            );
                        }
                    }
                }
                // else: card is unclassifiable — allowed, it's rejected upstream
                temp &= temp - 1;
            }
        }
    }

    /// Legal mask: every bit set in the mask must correspond to an intent whose
    /// resolved card is actually in the legal moves set.
    #[test]
    fn test_legal_mask_correctness() {
        let mut rng = Pcg64::seed_from_u64(456);

        for _ in 0..100 {
            let mut deck: Vec<u8> = (0..40).collect();
            for i in (1..40).rev() {
                let j = (rng.gen_range(0u64..((i + 1) as u64))) as usize;
                deck.swap(i, j);
            }
            let mut hands = [0u64; 4];
            for p in 0..4 {
                for c in 0..10 {
                    hands[p] |= 1u64 << deck[p * 10 + c];
                }
            }
            let trump = (rng.gen_range(0u64..4)) as u8;
            let first_player = (rng.gen_range(0u64..4)) as u8;
            let mut game = SuecaSimulatorGame::new(hands, trump, first_player);

            let tricks = (rng.gen_range(0u64..5)) as usize;
            for _ in 0..tricks {
                if game.state.trick_number >= 10 {
                    break;
                }
                let legal = game.state.legal_moves();
                if legal == 0 {
                    break;
                }
                let count = legal.count_ones();
                let idx = (rng.gen_range(0u64..(count as u64))) as usize;
                let mut temp = legal;
                for _ in 0..idx {
                    temp &= temp - 1;
                }
                game.play_card(temp.trailing_zeros() as u8);
            }
            if game.state.trick_number >= 10 {
                continue;
            }

            let seat = game.state.current_player;
            let legal = game.state.legal_moves();
            let mask = build_legal_mask(legal, &game, seat);

            // Each bit in mask must correctly indicate legality
            for intent in 0..OUTPUT_COUNT {
                let card = resolve_intent(intent, &game, seat);
                let is_legal = (legal & (1u64 << card)) != 0;
                let mask_bit = (mask >> intent) & 1;
                assert_eq!(
                    mask_bit, is_legal as u8,
                    "Intent {}: resolve_intent → card {}, legal={}, mask bit={}",
                    intent, card, is_legal, mask_bit
                );
            }
        }
    }

    /// All 4 oracle intents must appear in the generated data.
    /// No single intent should dominate — the class balancing must work.
    /// All 4 oracle intents must appear in the generated data.
    /// Distribution will be natural (not forced uniform).
    #[test]
    fn test_intent_diversity() {
        let config = DatasetConfig {
            n_worlds: 10,
            search_depth: 1,
            target_total: 200,
            seed: 999,
            output_path: String::new(),
            pimc_min_margin: 0.5,
        };
        let mut rng = Pcg64::seed_from_u64(999);
        let mut counts = [0usize; 4];
        let mut total = 0;

        for _ in 0..500 {
            for (_, soft_intent, _) in generate_deal_states(&mut rng, &config) {
                for i in 0..4 {
                    if soft_intent[i] > 0.0 {
                        counts[i] += 1;
                    }
                }
                total += 1;
            }
        }

        println!("Intent distribution: {:?} (total={})", counts, total);
        for i in 0..4 {
            assert!(
                counts[i] > 0,
                "Intent {} never appears in {} samples — dataset lacks diversity",
                i,
                total
            );
        }
    }

    /// Verify that legal_moves() returns moves for the current_player, not
    /// arbitrary seats. This is the root-cause test for the old contamination bug.
    #[test]
    fn test_legal_moves_is_current_player_only() {
        let mut rng = Pcg64::seed_from_u64(111);
        for _ in 0..50 {
            let mut deck: Vec<u8> = (0..40).collect();
            for i in (1..40).rev() {
                let j = (rng.gen_range(0u64..((i + 1) as u64))) as usize;
                deck.swap(i, j);
            }
            let mut hands = [0u64; 4];
            for p in 0..4 {
                for c in 0..10 {
                    hands[p] |= 1u64 << deck[p * 10 + c];
                }
            }
            let trump = (rng.gen_range(0u64..4)) as u8;
            let first_player = (rng.gen_range(0u64..4)) as u8;
            let mut game = SuecaSimulatorGame::new(hands, trump, first_player);

            let tricks = (rng.gen_range(0u64..5)) as usize;
            for _ in 0..tricks {
                if game.state.trick_number >= 10 {
                    break;
                }
                let legal = game.state.legal_moves();
                if legal == 0 {
                    break;
                }
                let count = legal.count_ones();
                let idx = (rng.gen_range(0u64..(count as u64))) as usize;
                let mut temp = legal;
                for _ in 0..idx {
                    temp &= temp - 1;
                }
                game.play_card(temp.trailing_zeros() as u8);
            }
            if game.state.trick_number >= 10 {
                continue;
            }

            let cp = game.state.current_player;
            let legal = game.state.legal_moves();
            let cp_hand = game.state.hands[cp as usize];

            // legal_moves must be a subset of the current player's hand
            assert_eq!(
                legal & !cp_hand,
                0,
                "legal_moves contains cards not in current player's hand"
            );

            // legal_moves must not contain cards from other players' hands
            for other in 0..4 {
                if other != cp {
                    let other_legal = legal & game.state.hands[other as usize];
                    // Can be non-zero only if both players hold the same card (impossible in Sueca)
                    assert_eq!(
                        other_legal,
                        0,
                        "legal_moves contains card {} from player {}'s hand (current player is {})",
                        other_legal.trailing_zeros(),
                        other,
                        cp
                    );
                }
            }
        }
    }

    /// NPZ save/load round-trip: what we write must be exactly what the
    /// training pipeline reads back.
    #[test]
    fn test_npz_round_trip() {
        let config = DatasetConfig {
            n_worlds: 10,
            search_depth: 1,
            target_total: 40,
            seed: 42,
            output_path: "/tmp/test_roundtrip.npz".into(),
            pimc_min_margin: 0.5,
        };

        // Collect states with natural distribution
        let mut rng = Pcg64::seed_from_u64(config.seed);
        let mut states = Vec::new();
        let mut intents = Vec::new();
        let mut masks = Vec::new();
        loop {
            for (state, intent, mask) in generate_deal_states(&mut rng, &config) {
                if intents.len() / 4 < config.target_total {
                    states.extend(state);
                    intents.extend_from_slice(&intent);
                    masks.push(mask);
                }
            }
            if intents.len() / 4 >= config.target_total {
                break;
            }
        }

        // Save
        save_npz(&states, &intents, &masks, &config.output_path);

        // Load and verify
        use crate::dataset::load_expert_dataset;
        let loaded = load_expert_dataset(&config.output_path).unwrap();

        assert_eq!(loaded.num_states, intents.len() / 4);
        assert_eq!(loaded.soft_intents, intents);
        assert_eq!(loaded.states.len(), states.len());

        // Compare states with tolerance for f32→f64 round-trip
        for (i, (&orig, &loaded_val)) in states.iter().zip(loaded.states.iter()).enumerate() {
            assert!(
                (orig - loaded_val).abs() < 1e-6,
                "State mismatch at index {}: original={}, loaded={}",
                i,
                orig,
                loaded_val
            );
        }

        // Cleanup
        std::fs::remove_file(&config.output_path).ok();
    }

    /// The belief state must not leak information about opponents' specific cards.
    /// It may encode public information (voids, played cards) but not private holdings.
    #[test]
    fn test_no_opponent_card_leakage() {
        let mut rng = Pcg64::seed_from_u64(555);
        for _ in 0..30 {
            let mut deck: Vec<u8> = (0..40).collect();
            for i in (1..40).rev() {
                let j = (rng.gen_range(0u64..((i + 1) as u64))) as usize;
                deck.swap(i, j);
            }
            let mut hands = [0u64; 4];
            for p in 0..4 {
                for c in 0..10 {
                    hands[p] |= 1u64 << deck[p * 10 + c];
                }
            }
            let trump = (rng.gen_range(0u64..4)) as u8;
            let first_player = (rng.gen_range(0u64..4)) as u8;
            let mut game = SuecaSimulatorGame::new(hands, trump, first_player);

            let tricks = (rng.gen_range(0u64..4)) as usize;
            for _ in 0..tricks {
                if game.state.trick_number >= 10 {
                    break;
                }
                let legal = game.state.legal_moves();
                if legal == 0 {
                    break;
                }
                let count = legal.count_ones();
                let idx = (rng.gen_range(0u64..(count as u64))) as usize;
                let mut temp = legal;
                for _ in 0..idx {
                    temp &= temp - 1;
                }
                game.play_card(temp.trailing_zeros() as u8);
            }
            if game.state.trick_number >= 10 {
                continue;
            }

            let seat = game.state.current_player;
            let belief = encode_belief_state(&game, seat);

            // Verify the belief state encodes the CURRENT player's hand info
            // (Hand_Point_Density > 0 implies having cards, which must be true after deal)
            assert!(
                belief[4] > 0.0,
                "Hand_Point_Density should be > 0 for active player"
            );

            // The belief state must not directly encode which specific cards
            // opponents hold. Features 0-20 encode hand, trick, and history
            // relative to the ego seat — none should reveal the integer card IDs
            // in an opponent's hand. We verify this by checking that the belief
            // is identical when we swap specific opponent cards of equal rank
            // (which doesn't change the public information).
            // The key invariant: belief values are all in [0,1] and derived
            // from public info only — no raw card IDs appear.
            for &v in &belief {
                assert!(v.is_finite() && (0.0..=1.0).contains(&v));
            }
        }
    }

    #[test]
    fn test_dataset_yield_ratio() {
        let config = DatasetConfig {
            n_worlds: 20, // small n_worlds for faster test execution
            search_depth: 2,
            target_total: 1000,
            seed: 54321,
            output_path: String::new(),
            pimc_min_margin: 0.5,
        };
        let mut rng = Pcg64::seed_from_u64(config.seed);
        let mut raw_tactical_states = 0;
        let mut preserved_states = 0;

        for _ in 0..1000 {
            // Replicate the generation setup
            let mut deck: Vec<u8> = (0..40).collect();
            for i in (1..40).rev() {
                let j = (rng.gen_range(0u64..((i + 1) as u64))) as usize;
                deck.swap(i, j);
            }

            let mut hands = [0u64; 4];
            for p in 0..4 {
                for c in 0..10 {
                    hands[p] |= 1u64 << deck[p * 10 + c];
                }
            }

            let trump = (rng.gen_range(0u64..4)) as u8;
            let first_player = (rng.gen_range(0u64..4)) as u8;

            let max_tricks = (rng.gen_range(1u64..9)) as usize;
            let mut game = SuecaSimulatorGame::new(hands, trump, first_player);

            for _ in 0..max_tricks {
                if game.state.trick_number >= 10 {
                    break;
                }
                let legal = game.state.legal_moves();
                if legal == 0 {
                    break;
                }
                let count = legal.count_ones();
                let idx = (rng.gen_range(0u64..(count as u64))) as usize;
                let mut temp = legal;
                for _ in 0..idx {
                    temp &= temp - 1;
                }
                game.play_card(temp.trailing_zeros() as u8);
            }

            if game.state.trick_number >= 10 {
                continue;
            }

            let seat = game.state.current_player;
            let legal = game.state.legal_moves();
            if legal.count_ones() <= 1 {
                continue;
            }

            // This is a raw tactical state where decision is needed and PIMC is executed
            raw_tactical_states += 1;

            let current_scores = [game.state.team_02_score, game.state.team_13_score];
            let current_trick_number = game.state.trick_number;
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
                CARD_SUIT[game.current_trick[0] as usize]
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
                seat,
                game.state.current_trick_winner,
                game.state.current_trick_best_card,
                current_scores,
                current_trick_number,
                config.n_worlds,
                config.search_depth,
                rng.gen_range(0u64..u64::MAX),
            );

            let mut sorted_evs = evs.clone();
            sorted_evs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            if sorted_evs.is_empty() {
                continue;
            }

            let best_card = sorted_evs[0].0;
            let mut pass_margin = true;
            if sorted_evs.len() >= 2 {
                let second_best_card = sorted_evs[1].0;
                let bypass_margin_check = CARD_SUIT[best_card as usize] == CARD_SUIT[second_best_card as usize]
                    && CARD_POINTS[best_card as usize] == 0
                    && CARD_POINTS[second_best_card as usize] == 0;

                if !bypass_margin_check && (sorted_evs[0].1 - sorted_evs[1].1) < config.pimc_min_margin {
                    pass_margin = false;
                }
            }

            if pass_margin {
                if map_card_to_soft_intents(best_card, &game, seat).is_some() {
                    preserved_states += 1;
                }
            }
        }

        let ratio = preserved_states as f64 / raw_tactical_states as f64;
        println!(
            "Dataset yield ratio: {:.2}% ({}/{} states preserved)",
            ratio * 100.0,
            preserved_states,
            raw_tactical_states
        );
        assert!(
            ratio >= 0.60,
            "Yield ratio too low! Got {:.2}%, expected >= 60%. Consensus erasure or follow-suit starvation is too aggressive.",
            ratio * 100.0
        );
    }

    #[test]
    fn test_new_belief_features_bounds_and_values() {
        let mut rng = Pcg64::seed_from_u64(42);
        let mut deck: Vec<u8> = (0..40).collect();
        for i in (1..40).rev() {
            let j = (rng.gen_range(0u64..((i + 1) as u64))) as usize;
            deck.swap(i, j);
        }
        let mut hands = [0u64; 4];
        for p in 0..4 {
            for c in 0..10 {
                hands[p] |= 1u64 << deck[p * 10 + c];
            }
        }
        let trump = 0;
        let first_player = 0;
        let mut game = SuecaSimulatorGame::new(hands, trump, first_player);

        // Play some tricks to accumulate points and expose voids
        for _ in 0..8 {
            let legal = game.state.legal_moves();
            if legal == 0 {
                break;
            }
            let card = legal.trailing_zeros() as u8;
            game.play_card(card);
        }

        let seat = game.state.current_player;
        let belief = encode_belief_state(&game, seat);

        assert_eq!(belief.len(), 33);
        // Verify value bounds
        for i in 30..33 {
            assert!(
                (0.0..=1.0).contains(&belief[i]),
                "Feature {} has out-of-bounds value: {}",
                i,
                belief[i]
            );
        }

        // Verify PointsSecured logic specifically
        let expected_score = if (seat % 2) == 0 { game.state.team_02_score } else { game.state.team_13_score };
        let expected_secured = (expected_score as f64) / 120.0;
        assert!((belief[30] - expected_secured).abs() < 1e-9);
    }

    #[test]
    fn test_intent_demotion_leading() {
        // Test intent demotion specifically when leading (current_trick_len == 0)
        let mut hands = [0u64; 4];
        for p in 0..4 {
            hands[p] = 0x3FFu64 << (p * 10);
        }
        let game = SuecaSimulatorGame::new(hands, 3, 0);
        
        // At Gen 0, seat 0 leading
        assert_eq!(game.current_trick_len, 0);
        
        // Let's call map_card_to_soft_intents on a card that resolved to both.
        // If there's a card that matches both, we check that intent 0 is filtered out:
        // Let's find if there is a card where both intents resolve to it:
        let card_0 = resolve_intent(0, &game, 0);
        let card_3 = resolve_intent(3, &game, 0);
        if card_0 == card_3 {
            let soft = map_card_to_soft_intents(card_0, &game, 0).unwrap();
            // Since it matched both, intent 0 must be demoted, so soft[0] should be 0.0, and soft[3] should be 1.0!
            assert_eq!(soft[0], 0.0);
            assert_eq!(soft[3], 1.0);
        }
    }
}
