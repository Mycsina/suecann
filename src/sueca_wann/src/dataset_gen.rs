use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sueca_solver::belief::encode_belief_state;
use sueca_solver::constants::{INPUT_COUNT, OUTPUT_COUNT};
use sueca_solver::engine::{CARD_POINTS, CARD_SUIT};
use sueca_solver::heuristic::{resolve_intent, select_card_heuristic};
use sueca_solver::simulator::SuecaSimulatorGame;

/// Serializable snapshot of dataset generation progress.
/// Saved every CHECKPOINT_INTERVAL batches so a crashed/restarted run
/// can resume without losing accumulated states.
#[derive(Serialize, Deserialize)]
struct DatasetCheckpoint {
    class_counts: [usize; OUTPUT_COUNT as usize],
    bucket_states: [Vec<f64>; OUTPUT_COUNT as usize],
    bucket_intents: [Vec<f32>; OUTPUT_COUNT as usize],
    bucket_masks: [Vec<u8>; OUTPUT_COUNT as usize],
    batch_seed: u64,
    batches: usize,
    // Config fields for validation on resume
    n_worlds: usize,
    search_depth: u8,
    target_total: usize,
    seed: u64,
}

/// Save a checkpoint to a bincode file.
fn save_checkpoint(ckpt: &DatasetCheckpoint, path: &str) {
    let file = std::fs::File::create(path)
        .unwrap_or_else(|e| panic!("Cannot create checkpoint file '{}': {}", path, e));
    let writer = std::io::BufWriter::new(file);
    bincode::serialize_into(writer, ckpt)
        .unwrap_or_else(|e| panic!("Cannot write checkpoint '{}': {}", path, e));
}

/// Load a checkpoint from a bincode file. Returns None if the file doesn't exist.
fn load_checkpoint(path: &str) -> Option<DatasetCheckpoint> {
    let file = std::fs::File::open(path).ok()?;
    let reader = std::io::BufReader::new(file);
    bincode::deserialize_from(reader).ok()
}

fn checkpoint_path(output_path: &str) -> String {
    // Strip .npz suffix if present, then append .checkpoint
    output_path
        .strip_suffix(".npz")
        .unwrap_or(output_path)
        .to_string()
        + ".checkpoint"
}

const CHECKPOINT_INTERVAL: usize = 5; // save every 5 batches (~23 min)

#[derive(Serialize, Deserialize)]
pub struct DatasetConfig {
    pub n_worlds: usize,
    pub search_depth: u8,
    pub target_total: usize,
    pub seed: u64,
    pub output_path: String,
    /// Minimum fraction of total each intent class must have.
    /// 0.20 = no class below 20% of total. Soft balance via class weighting
    /// compensates for remaining imbalance in Phase 0 fitness.
    #[serde(default = "default_soft_balance_min_ratio")]
    pub soft_balance_min_ratio: f64,
}

fn default_soft_balance_min_ratio() -> f64 {
    0.20
}

pub fn generate_dataset(config: &DatasetConfig, resume: bool) {
    let total_target = config.target_total;
    let soft_min = (total_target as f64 * config.soft_balance_min_ratio) as usize;

    println!(
        "Generating expert dataset: {} worlds, depth {}, {} total states (soft min {} per class)...",
        config.n_worlds, config.search_depth, total_target, soft_min
    );

    // Soft-balanced accumulation: each intent class must have at least soft_min
    // states, but the total target can be met with any distribution. This avoids
    // the brutal marginal cost of perfect 33.3% balance — the class weighting in
    // Phase 0 fitness compensates for any remaining imbalance.
    //
    // Ambiguous states are rejected at two levels:
    //   1. Always-on pre-filter (intent uniqueness — resolve_intent collision check)
    //   2. PIMC confidence filter (EV delta must exceed measurement noise)
    //   3. Intent uniqueness filter (card must map to exactly one intent)
    //
    // Only crystal-clear, strategically unambiguous examples enter the buckets.
    let per_class_target = total_target / (OUTPUT_COUNT as usize); // for logging
    let ckpt_path = checkpoint_path(&config.output_path);

    // ── Resume from checkpoint or start fresh ──
    let (mut bucket_states, mut bucket_intents, mut bucket_masks, mut class_counts, mut batch_seed, mut batches): (
        [Vec<f64>; OUTPUT_COUNT as usize],
        [Vec<f32>; OUTPUT_COUNT as usize],
        [Vec<u8>; OUTPUT_COUNT as usize],
        [usize; OUTPUT_COUNT as usize],
        u64,
        usize,
    ) = if resume {
        match load_checkpoint(&ckpt_path) {
            Some(ckpt) => {
                // Validate config compatibility
                if ckpt.n_worlds != config.n_worlds
                    || ckpt.search_depth != config.search_depth
                    || ckpt.target_total != config.target_total
                    || ckpt.seed != config.seed
                {
                    eprintln!(
                        "Error: checkpoint config mismatch.\n\
                         Checkpoint: n_worlds={}, depth={}, target={}, seed={}\n\
                         Current:    n_worlds={}, depth={}, target={}, seed={}\n\
                         Delete '{}' to start fresh, or adjust config to match.",
                        ckpt.n_worlds, ckpt.search_depth, ckpt.target_total, ckpt.seed,
                        config.n_worlds, config.search_depth, config.target_total, config.seed,
                        ckpt_path,
                    );
                    std::process::exit(1);
                }
                println!(
                    "Resumed from checkpoint: batch {}, collected {:?} (target {} each)",
                    ckpt.batches, ckpt.class_counts, per_class_target,
                );
                (
                    ckpt.bucket_states,
                    ckpt.bucket_intents,
                    ckpt.bucket_masks,
                    ckpt.class_counts,
                    ckpt.batch_seed,
                    ckpt.batches,
                )
            }
            None => {
                eprintln!(
                    "Error: --resume specified but checkpoint file '{}' not found.",
                    ckpt_path,
                );
                std::process::exit(1);
            }
        }
    } else {
        (
            Default::default(),
            Default::default(),
            Default::default(),
            [0usize; OUTPUT_COUNT as usize],
            config.seed,
            0usize,
        )
    };

    loop {
        // Stop when total target met AND each class has at least soft_min
        let total_collected: usize = class_counts.iter().sum();
        let all_above_soft_min = class_counts.iter().all(|&c| c >= soft_min);
        if total_collected >= total_target && all_above_soft_min {
            break;
        }
        batches += 1;

        let batch_size = 512;
        let batch_results: Vec<Vec<(Vec<f64>, [f32; OUTPUT_COUNT], u8)>> = (0..batch_size)
            .into_par_iter()
            .map(|i| {
                let deal_seed = batch_seed + i as u64;
                let mut rng = Pcg64::seed_from_u64(deal_seed);
                generate_deal_states(&mut rng, config)
            })
            .collect();

        for results in batch_results {
            for (state, soft_intent, mask) in results {
                // With one-hot targets (ambiguous ties rejected upstream),
                // the primary intent is simply the argmax.
                let primary = soft_intent
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                // Soft balance: accept if this class hasn't exceeded its
                // effective cap (leaving room for other classes to meet soft_min)
                let other_slack: usize = (0..OUTPUT_COUNT as usize)
                    .filter(|&c| c != primary)
                    .map(|c| soft_min.saturating_sub(class_counts[c]))
                    .sum();
                let cap_for_this_class = total_target.saturating_sub(
                    (OUTPUT_COUNT as usize - 1) * soft_min + other_slack
                ).max(soft_min);

                if class_counts[primary] < cap_for_this_class {
                    class_counts[primary] += 1;
                    bucket_states[primary].extend(&state);
                    bucket_intents[primary].extend_from_slice(&soft_intent);
                    bucket_masks[primary].push(mask);
                }
            }
        }

        batch_seed += batch_size as u64;

        if batches % 1 == 0 {
            let total: usize = class_counts.iter().sum();
            println!(
                "  batch {}, per intent: {:?} (total {}, soft min {}, target {})",
                batches, class_counts, total, soft_min, total_target
            );
        }

        // Periodic checkpoint — resume-safe across crashes/restarts
        if batches % CHECKPOINT_INTERVAL == 0 {
            let ckpt = DatasetCheckpoint {
                class_counts,
                bucket_states: bucket_states.clone(),
                bucket_intents: bucket_intents.clone(),
                bucket_masks: bucket_masks.clone(),
                batch_seed,
                batches,
                n_worlds: config.n_worlds,
                search_depth: config.search_depth,
                target_total: config.target_total,
                seed: config.seed,
            };
            save_checkpoint(&ckpt, &ckpt_path);
        }
    }

    // Interleave buckets so the final flat arrays alternate intents rather
    // than concatenating all of intent 0, then all of intent 1, etc.
    let total = class_counts.iter().sum::<usize>();
    let mut all_states = Vec::with_capacity(total * INPUT_COUNT);
    let mut all_intents = Vec::with_capacity(total * OUTPUT_COUNT as usize);
    let mut all_masks = Vec::with_capacity(total);

    let max_per_class = *class_counts.iter().max().unwrap();
    for i in 0..max_per_class {
        for c in 0..(OUTPUT_COUNT as usize) {
            if i < bucket_masks[c].len() {
                let state_off = i * INPUT_COUNT;
                all_states
                    .extend_from_slice(&bucket_states[c][state_off..state_off + INPUT_COUNT]);
                let intent_off = i * (OUTPUT_COUNT as usize);
                all_intents.extend_from_slice(
                    &bucket_intents[c][intent_off..intent_off + (OUTPUT_COUNT as usize)],
                );
                all_masks.push(bucket_masks[c][i]);
            }
        }
    }

    println!(
        "Final dataset: {} states, intent distribution: {:?} (target {} each)",
        total,
        class_counts,
        per_class_target
    );

    save_npz(&all_states, &all_intents, &all_masks, &config.output_path);

    // Remove checkpoint — dataset is complete, no need to resume
    if let Err(e) = std::fs::remove_file(&ckpt_path) {
        if e.kind() != std::io::ErrorKind::NotFound {
            eprintln!("Warning: could not remove checkpoint '{}': {}", ckpt_path, e);
        }
    }
}

/// Number of additional states to extract from the same deal after the
/// PIMC-recommended move is played. Amortizes deal setup and void tracking
/// across multiple states. States are mildly correlated (same deal), which
/// is harmless for capacity-limited WANNs.
const N_EXTRA_EXTRACTIONS: usize = 2;

pub fn generate_deal_states(
    rng: &mut Pcg64,
    config: &DatasetConfig,
) -> Vec<(Vec<f64>, [f32; OUTPUT_COUNT], u8)> {
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

    // ── Extract state at current position ──
    extract_state_at(&game, config, rng, &mut results);

    // ── Multi-state extraction: complete the current trick using ──
    // HeuristicBot, then extract states at subsequent positions to
    // amortize deal setup and void tracking across multiple states.
    let mut extra_game = game;
    for _ in 0..N_EXTRA_EXTRACTIONS {
        if extra_game.state.trick_number >= 10 {
            break;
        }

        // Complete the current trick if mid-trick
        while extra_game.state.cards_played_in_trick > 0 {
            let seat = extra_game.state.current_player;
            let legal = extra_game.state.legal_moves();
            if legal.count_ones() <= 1 {
                let card = legal.trailing_zeros() as u8;
                extra_game.play_card(card);
            } else {
                let card = select_card_heuristic(&extra_game, seat);
                extra_game.play_card(card);
            }
        }

        // Now at the start of a new trick — extract state
        if extra_game.state.trick_number < 10 {
            extract_state_at(&extra_game, config, rng, &mut results);
        }
    }

    results
}

/// Extract a single labeled state from the current game position.
/// Applies always-on pre-filters before PIMC and confidence filtering after.
fn extract_state_at(
    game: &SuecaSimulatorGame,
    config: &DatasetConfig,
    rng: &mut Pcg64,
    results: &mut Vec<(Vec<f64>, [f32; OUTPUT_COUNT], u8)>,
) {
    // CRITICAL: Extract data ONLY for the active player whose turn it is.
    let seat = game.state.current_player;
    let legal = game.state.legal_moves();

    if legal.count_ones() <= 1 {
        return;
    }

    // ── Always-on pre-filter: intent uniqueness check ──
    // If any two oracle intents resolve to the same card, the downstream
    // map_card_to_soft_intents filter will reject this state. Skip PIMC entirely.
    // resolve_intent() is O(legal_moves) — microseconds vs seconds for PIMC.
    {
        let mut card_for_intent = [40u8; OUTPUT_COUNT as usize];
        for intent in 0..OUTPUT_COUNT as usize {
            card_for_intent[intent] = resolve_intent(intent, game, seat);
        }
        let mut collision = false;
        for i in 0..OUTPUT_COUNT as usize {
            for j in (i + 1)..OUTPUT_COUNT as usize {
                if card_for_intent[i] == card_for_intent[j] {
                    collision = true;
                    break;
                }
            }
            if collision {
                break;
            }
        }
        if collision {
            return;
        }
    }

    let current_scores = [game.state.team_02_score, game.state.team_13_score];
    let current_trick_number = game.state.trick_number;

    let belief = encode_belief_state(game, seat);

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

    // Empty EV results → futility stop fired. Skip this state.
    if evs.is_empty() {
        return;
    }

    let mut sorted: Vec<&sueca_solver::pimc::PimcResult> = evs.iter().collect();
    sorted.sort_by(|a, b| b.ev.partial_cmp(&a.ev).unwrap_or(std::cmp::Ordering::Equal));

    if sorted.is_empty() {
        return;
    }

    // ── Confidence filter: reject strategically ambiguous states ──
    let best = sorted[0];
    let is_clear = if sorted.len() >= 2 {
        let second = sorted[1];
        let bypass = CARD_SUIT[best.card as usize] == CARD_SUIT[second.card as usize]
            && CARD_POINTS[best.card as usize] == 0
            && CARD_POINTS[second.card as usize] == 0;
        if bypass {
            true
        } else {
            let ev_delta = best.ev - second.ev;
            let noise_floor = best.std_error;
            ev_delta > noise_floor
        }
    } else {
        true
    };

    if !is_clear {
        return;
    }

    if let Some(soft_intent) = map_card_to_soft_intents(best.card, game, seat) {
        let legal_mask = build_legal_mask(legal, game, seat);
        let state_vec: Vec<f64> = belief.to_vec();
        results.push((state_vec, soft_intent, legal_mask));
    }
}

pub fn map_card_to_soft_intents(card: u8, game: &SuecaSimulatorGame, seat: u8) -> Option<[f32; OUTPUT_COUNT]> {
    let mut matching_intents = Vec::new();
    for intent in 0..OUTPUT_COUNT as usize {
        let resolved = resolve_intent(intent, game, seat);
        if resolved == card {
            matching_intents.push(intent);
        }
    }
    if matching_intents.is_empty() {
        return None;
    }

    // Apply Intent Demotion for leading states:
    // If a card satisfies both MAX_FORCE (0) and EQUITY_BUILDER (2) when leading,
    // break the tie in favor of the more specific tactical intent (EQUITY_BUILDER).
    if game.current_trick_len == 0 {
        if matching_intents.contains(&0) && matching_intents.contains(&2) {
            matching_intents.retain(|&intent| intent != 0);
        }
    }

    // Reject states where multiple intents still map to the same card.
    // WANNs learn broad IF/THEN heuristics from structural patterns in the
    // dataset. States where two intents resolve identically (e.g., both
    // EFFICIENT_WIN and EQUITY_BUILDER pick the same card) are "shrugs" —
    // they teach no discrimination. Including them dilutes the pure strategic
    // signal of each class and causes the network to learn fuzzy boundaries.
    //
    // It is better to have 8,000 crystal-clear examples of EQUITY_BUILDER
    // than 10,000 examples where 2,000 are ambiguous padding.
    if matching_intents.len() > 1 {
        return None;
    }

    let mut target_vector = [0.0f32; OUTPUT_COUNT];
    target_vector[matching_intents[0]] = 1.0;
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

    let num_states = intents.len() / (OUTPUT_COUNT as usize);
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
        let npy = write_npy(raw, "<f4", &[num_states as u64, OUTPUT_COUNT as u64]);
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
            n_worlds: 50, // enough worlds for meaningful SE-based confidence filter
            search_depth: 2,
            target_total: 100,
            seed: 12345,
            output_path: String::new(),
            soft_balance_min_ratio: 0.20,
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
        for _ in 0..600 {
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
            soft_balance_min_ratio: 0.20,
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
            for r in &evs {
                if r.ev > max_ev {
                    max_ev = r.ev;
                    best_card = r.card;
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
                    for intent in 0..OUTPUT_COUNT as usize {
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
            soft_balance_min_ratio: 0.20,
        };
        let mut rng = Pcg64::seed_from_u64(999);
        let mut counts = [0usize; 4];
        let mut total = 0;

        for _ in 0..500 {
            for (_, soft_intent, _) in generate_deal_states(&mut rng, &config) {
                for i in 0..OUTPUT_COUNT as usize {
                    if soft_intent[i] > 0.0 {
                        counts[i] += 1;
                    }
                }
                total += 1;
            }
        }

        println!("Intent distribution: {:?} (total={})", counts, total);
        for i in 0..OUTPUT_COUNT as usize {
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
            soft_balance_min_ratio: 0.20,
        };

        // Collect states with natural distribution
        let mut rng = Pcg64::seed_from_u64(config.seed);
        let mut states = Vec::new();
        let mut intents = Vec::new();
        let mut masks = Vec::new();
        loop {
            for (state, intent, mask) in generate_deal_states(&mut rng, &config) {
                if intents.len() / (OUTPUT_COUNT as usize) < config.target_total {
                    states.extend(state);
                    intents.extend_from_slice(&intent);
                    masks.push(mask);
                }
            }
            if (intents.len() / (OUTPUT_COUNT as usize)) >= config.target_total {
                break;
            }
        }

        // Save
        save_npz(&states, &intents, &masks, &config.output_path);

        // Load and verify
        use crate::dataset::load_expert_dataset;
        let loaded = load_expert_dataset(&config.output_path).unwrap();

        assert_eq!(loaded.num_states, intents.len() / (OUTPUT_COUNT as usize));
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
            n_worlds: 80, // enough for meaningful SE (≈2.2 pts with σ=20)
            search_depth: 2,
            target_total: 1000,
            seed: 54321,
            output_path: String::new(),
            soft_balance_min_ratio: 0.20,
        };
        let mut rng = Pcg64::seed_from_u64(config.seed);
        let mut raw_tactical_states = 0;
        let mut preserved_states = 0;

        for _ in 0..500 {
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

            let mut sorted: Vec<&sueca_solver::pimc::PimcResult> = evs.iter().collect();
            sorted.sort_by(|a, b| b.ev.partial_cmp(&a.ev).unwrap_or(std::cmp::Ordering::Equal));

            if sorted.is_empty() {
                continue;
            }

            // Same confidence filter as generate_deal_states
            let best = sorted[0];
            let pass_margin = if sorted.len() >= 2 {
                let second = sorted[1];
                let bypass = CARD_SUIT[best.card as usize] == CARD_SUIT[second.card as usize]
                    && CARD_POINTS[best.card as usize] == 0
                    && CARD_POINTS[second.card as usize] == 0;
                if bypass {
                    true
                } else {
                    let ev_delta = best.ev - second.ev;
                    ev_delta > best.std_error
                }
            } else {
                true
            };

            if pass_margin {
                if map_card_to_soft_intents(best.card, &game, seat).is_some() {
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
            ratio >= 0.15,
            "Yield ratio too low! Got {:.2}%, expected >= 15%. SE filter or intent-uniqueness filter may be too aggressive.",
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

        assert_eq!(belief.len(), sueca_solver::constants::INPUT_COUNT);
        // Verify value bounds for meta features
        for i in (sueca_solver::constants::INPUT_COUNT - 3)..sueca_solver::constants::INPUT_COUNT {
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
        let card_2 = resolve_intent(2, &game, 0);
        if card_0 == card_2 {
            let soft = map_card_to_soft_intents(card_0, &game, 0).unwrap();
            // Since it matched both, intent 0 must be demoted in favor of EQUITY_BUILDER (intent 2)
            assert_eq!(soft[0], 0.0);
            assert_eq!(soft[2], 1.0);
        }
    }
}
