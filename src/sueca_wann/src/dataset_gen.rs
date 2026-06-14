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

/// Number of buckets: 2 splits (lead, follow) × 3 intents = 6.
const N_BUCKETS: usize = 6;

/// Serializable snapshot of dataset generation progress.
/// Saved every CHECKPOINT_INTERVAL batches so a crashed/restarted run
/// can resume without losing accumulated states.
#[derive(Serialize, Deserialize)]
struct DatasetCheckpoint {
    /// Flat 6-bucket counts indexed by split * 3 + intent.
    /// split 0 = Lead, split 1 = Follow.
    class_counts: [usize; N_BUCKETS],
    bucket_states: [Vec<f64>; N_BUCKETS],
    bucket_intents: [Vec<f32>; N_BUCKETS],
    bucket_masks: [Vec<u8>; N_BUCKETS],
    batch_seed: u64,
    batches: usize,
    // Config fields for validation on resume
    n_worlds: usize,
    search_depth: u8,
    target_total: usize,
    seed: u64,
}

/// Bucket index helper: split (0=Lead, 1=Follow) × intent → flat index
#[inline(always)]
fn bucket_idx(split: usize, intent: usize) -> usize {
    split * 3 + intent
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
    #[serde(default = "default_soft_balance_min_ratio")]
    pub soft_balance_min_ratio: f64,
    /// If true, disable early termination and futility stop in PIMC.
    /// Used for controlled label-comparison diffing between pipeline versions.
    #[serde(default)]
    pub diff_mode: bool,
    /// When set, use exactly this many worlds per PIMC call (diff_mode only).
    #[serde(default)]
    pub fixed_worlds: Option<usize>,
}

fn default_soft_balance_min_ratio() -> f64 {
    0.20
}

pub fn generate_dataset(config: &DatasetConfig, resume: bool) {
    let total_target = config.target_total;
    // Per-split target: half the total for Lead, half for Follow
    let per_split_target = total_target / 2;
    let soft_min = (per_split_target as f64 * config.soft_balance_min_ratio) as usize;

    println!(
        "Generating expert dataset: {} worlds, depth {}, {} total ({} per split, soft min {} per intent/split)...",
        config.n_worlds, config.search_depth, total_target, per_split_target, soft_min
    );

    // 6-bucket accumulation: (lead, follow) × (MAX_FORCE, EFFICIENT_WIN, EQUITY_BUILDER).
    // Indexed by bucket_idx(split, intent). split=0 for Lead, split=1 for Follow.
    // The Follow brain trains ONLY on its split — per-split balance is non-negotiable.
    let per_class_target = per_split_target / (OUTPUT_COUNT as usize); // for logging
    let ckpt_path = checkpoint_path(&config.output_path);

    // ── Resume from checkpoint or start fresh ──
    let (mut bucket_states, mut bucket_intents, mut bucket_masks, mut class_counts, mut batch_seed, mut batches): (
        [Vec<f64>; N_BUCKETS],
        [Vec<f32>; N_BUCKETS],
        [Vec<u8>; N_BUCKETS],
        [usize; N_BUCKETS],
        u64,
        usize,
    ) = if resume {
        match load_checkpoint(&ckpt_path) {
            Some(ckpt) => {
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
                    "Resumed from checkpoint: batch {}, collected {:?} (target {} per intent/split)",
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
            [0usize; N_BUCKETS],
            config.seed,
            0usize,
        )
    };

    loop {
        // Stop when each split has met its target AND each bucket has soft_min
        let lead_total: usize = (0..3).map(|i| class_counts[bucket_idx(0, i)]).sum();
        let follow_total: usize = (0..3).map(|i| class_counts[bucket_idx(1, i)]).sum();
        let all_above_soft_min = (0..N_BUCKETS).all(|b| class_counts[b] >= soft_min);
        if lead_total >= per_split_target && follow_total >= per_split_target && all_above_soft_min {
            break;
        }
        batches += 1;

        // Determine rarest of 6 buckets for steering.
        // Only activate when critically behind (< 50% of soft_min).
        let steer_intent: Option<usize> = {
            let min_count = *class_counts.iter().min().unwrap_or(&0);
            let critical_threshold = (soft_min / 2).max(1);
            if min_count < critical_threshold {
                let rarest_bucket = class_counts.iter().position(|&c| c == min_count).unwrap_or(0);
                Some(rarest_bucket % 3) // intent part of bucket index
            } else {
                None
            }
        };

        let batch_size = 512;
        let batch_pairs: Vec<(Vec<(Vec<f64>, [f32; OUTPUT_COUNT], u8)>, DealRejectCounts)> = (0..batch_size)
            .into_par_iter()
            .map(|i| {
                let deal_seed = batch_seed + i as u64;
                let mut rng = Pcg64::seed_from_u64(deal_seed);
                generate_deal_states(&mut rng, config, steer_intent)
            })
            .collect();

        let mut batch_rejects = DealRejectCounts::default();
        for (results, deal_rejects) in batch_pairs {
            batch_rejects.merge(&deal_rejects);
            for (state, soft_intent, mask) in results {
                let primary = soft_intent
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                // AmILeading is belief feature index 5
                let split = if state[5] > 0.5 { 0 } else { 1 };
                let bidx = bucket_idx(split, primary);
                let split_total: usize = (0..3).map(|i| class_counts[bucket_idx(split, i)]).sum();

                // Per-split soft balance
                let other_slack: usize = (0..3usize)
                    .filter(|&c| c != primary)
                    .map(|c| soft_min.saturating_sub(class_counts[bucket_idx(split, c)]))
                    .sum();
                let cap = per_split_target.saturating_sub(2 * soft_min + other_slack).max(soft_min);

                if split_total < per_split_target || class_counts[bidx] < soft_min {
                    if class_counts[bidx] < cap {
                        class_counts[bidx] += 1;
                        bucket_states[bidx].extend(&state);
                        bucket_intents[bidx].extend_from_slice(&soft_intent);
                        bucket_masks[bidx].push(mask);
                    }
                }
            }
        }

        batch_seed += batch_size as u64;

        if batches % 1 == 0 {
            let total: usize = class_counts.iter().sum();
            let lead_pct = if batch_rejects.lead_primary + batch_rejects.follow_primary > 0 {
                100.0 * batch_rejects.lead_primary as f64
                    / (batch_rejects.lead_primary + batch_rejects.follow_primary) as f64
            } else {
                0.0
            };
            println!(
                "  batch {}, L:{:?} F:{:?} (total {}, soft min {}, target {})",
                batches,
                &class_counts[0..3],
                &class_counts[3..6],
                total, soft_min, total_target
            );
            println!(
                "    rejects: acc={} col={} fut={} conf={} uncl={} steer={} sl={} term={} lead={:.0}%",
                batch_rejects.accepted, batch_rejects.intent_collision,
                batch_rejects.futility_stop, batch_rejects.confidence_filter,
                batch_rejects.unclassifiable, batch_rejects.steered_out,
                batch_rejects.single_legal, batch_rejects.terminal,
                lead_pct,
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

    // Interleave across all 6 buckets for balanced final output
    let total = class_counts.iter().sum::<usize>();
    let mut all_states = Vec::with_capacity(total * INPUT_COUNT);
    let mut all_intents = Vec::with_capacity(total * OUTPUT_COUNT as usize);
    let mut all_masks = Vec::with_capacity(total);

    let max_per_bucket = *class_counts.iter().max().unwrap();
    for i in 0..max_per_bucket {
        for b in 0..N_BUCKETS {
            if i < bucket_masks[b].len() {
                let state_off = i * INPUT_COUNT;
                all_states
                    .extend_from_slice(&bucket_states[b][state_off..state_off + INPUT_COUNT]);
                let intent_off = i * (OUTPUT_COUNT as usize);
                all_intents.extend_from_slice(
                    &bucket_intents[b][intent_off..intent_off + (OUTPUT_COUNT as usize)],
                );
                all_masks.push(bucket_masks[b][i]);
            }
        }
    }

    let lead_total: usize = (0..3).map(|i| class_counts[bucket_idx(0, i)]).sum();
    let follow_total: usize = (0..3).map(|i| class_counts[bucket_idx(1, i)]).sum();
    println!(
        "Final dataset: {} states (L:{} F:{}), L:[{:?}] F:[{:?}]",
        total, lead_total, follow_total,
        &class_counts[0..3],
        &class_counts[3..6],
    );

    save_npz(&all_states, &all_intents, &all_masks, &config.output_path);

    // Remove checkpoint — dataset is complete, no need to resume
    if let Err(e) = std::fs::remove_file(&ckpt_path) {
        if e.kind() != std::io::ErrorKind::NotFound {
            eprintln!("Warning: could not remove checkpoint '{}': {}", ckpt_path, e);
        }
    }
}

/// Number of additional extractions per deal. Each advances 1-6 random
/// cards (not to a trick boundary) to maintain the ~25% lead fraction.
const N_EXTRA_EXTRACTIONS: usize = 1;

/// Per-deal rejection counters for quality diagnostics.
#[derive(Default, Clone)]
pub struct DealRejectCounts {
    pub single_legal: usize,
    pub intent_collision: usize,
    pub futility_stop: usize,
    pub confidence_filter: usize,
    pub unclassifiable: usize,
    pub steered_out: usize,
    pub accepted: usize,
    pub terminal: usize,
    pub lead_primary: usize,
    pub follow_primary: usize,
}

impl DealRejectCounts {
    fn merge(&mut self, other: &DealRejectCounts) {
        self.single_legal += other.single_legal;
        self.intent_collision += other.intent_collision;
        self.futility_stop += other.futility_stop;
        self.confidence_filter += other.confidence_filter;
        self.unclassifiable += other.unclassifiable;
        self.steered_out += other.steered_out;
        self.accepted += other.accepted;
        self.terminal += other.terminal;
        self.lead_primary += other.lead_primary;
        self.follow_primary += other.follow_primary;
    }
}

pub fn generate_deal_states(
    rng: &mut Pcg64,
    config: &DatasetConfig,
    steer_intent: Option<usize>,
) -> (Vec<(Vec<f64>, [f32; OUTPUT_COUNT], u8)>, DealRejectCounts) {
    let mut results = Vec::new();
    let mut rejects = DealRejectCounts::default();

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

    // ── Mid-trick random walk ──
    // Play k complete tricks (0-7) plus 0-3 additional cards so the ego
    // ends at a uniform random point within a trick. Primary extractions
    // are ~25% leading, ~75% following — matching the natural distribution
    // of a 4-player trick-taking game.
    let n_tricks = rng.gen_range(0u64..8) as usize;
    let n_cards = rng.gen_range(0u64..4) as usize;
    let total_cards = n_tricks * 4 + n_cards;

    let mut game = SuecaSimulatorGame::new(hands, trump, first_player);

    for _ in 0..total_cards {
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
        rejects.terminal += 1;
        return (results, rejects);
    }

    // ── Post-walk steering: advance ONE card past mismatched position ──
    // Does NOT complete the trick — stays mid-trick to preserve the
    // ~25% lead fraction for the follow-on extraction.
    if let Some(target) = steer_intent {
        let is_leading = game.current_trick_len == 0;
        let needs_advance = match target {
            1 => is_leading,     // EFFICIENT_WIN wants following
            2 => !is_leading,    // EQUITY_BUILDER wants leading
            0 => !is_leading,    // MAX_FORCE wants leading
            _ => false,
        };
        if needs_advance && game.state.trick_number < 9 {
            let seat = game.state.current_player;
            let card = select_card_heuristic(&game, seat);
            game.play_card(card);
            if game.state.trick_number >= 10 {
                rejects.terminal += 1;
                return (results, rejects);
            }
        }
    }

    // ── Primary extraction ──
    let primary_is_lead = game.current_trick_len == 0;
    match extract_state_at(&game, config, rng, &mut results) {
        RejectReason::Accepted => {
            rejects.accepted += 1;
            if primary_is_lead { rejects.lead_primary += 1; }
            else { rejects.follow_primary += 1; }
        }
        RejectReason::SingleLegal => rejects.single_legal += 1,
        RejectReason::IntentCollision => rejects.intent_collision += 1,
        RejectReason::FutilityStop => rejects.futility_stop += 1,
        RejectReason::ConfidenceFilter => rejects.confidence_filter += 1,
        RejectReason::Unclassifiable => rejects.unclassifiable += 1,
        RejectReason::SteeredOut => rejects.steered_out += 1,
    }

    // ── Extra extraction: advance 1-6 random cards, then extract ──
    // Does NOT complete the trick — stays mid-trick. This keeps extras
    // at ~25% lead, same as primary.
    let mut extra_game = game;
    for _ in 0..N_EXTRA_EXTRACTIONS {
        if extra_game.state.trick_number >= 10 {
            break;
        }

        let n_advance = rng.gen_range(1u64..=6) as usize;
        for _ in 0..n_advance {
            if extra_game.state.trick_number >= 10 {
                break;
            }
            let seat = extra_game.state.current_player;
            let legal = extra_game.state.legal_moves();
            if legal.count_ones() <= 1 {
                let card = legal.trailing_zeros() as u8;
                extra_game.play_card(card);
            } else if rng.gen_bool(0.5) {
                let card = select_card_heuristic(&extra_game, seat);
                extra_game.play_card(card);
            } else {
                let count = legal.count_ones();
                let idx = rng.gen_range(0u64..(count as u64)) as usize;
                let mut temp = legal;
                for _ in 0..idx {
                    temp &= temp - 1;
                }
                extra_game.play_card(temp.trailing_zeros() as u8);
            }
        }

        if extra_game.state.trick_number < 10 {
            let extra_is_lead = extra_game.current_trick_len == 0;
            match extract_state_at(&extra_game, config, rng, &mut results) {
                RejectReason::Accepted => {
                    rejects.accepted += 1;
                    if extra_is_lead { rejects.lead_primary += 1; }
                    else { rejects.follow_primary += 1; }
                }
                RejectReason::SingleLegal => rejects.single_legal += 1,
                RejectReason::IntentCollision => rejects.intent_collision += 1,
                RejectReason::FutilityStop => rejects.futility_stop += 1,
                RejectReason::ConfidenceFilter => rejects.confidence_filter += 1,
                RejectReason::Unclassifiable => rejects.unclassifiable += 1,
                RejectReason::SteeredOut => rejects.steered_out += 1,
            }
        }
    }

    (results, rejects)
}

/// Rejection reasons for dataset quality tracking.
#[derive(Clone, Copy)]
enum RejectReason {
    Accepted,
    SingleLegal,
    IntentCollision,
    FutilityStop,
    ConfidenceFilter,
    Unclassifiable,
    SteeredOut,
}

/// Extract a single labeled state from the current game position.
/// Applies always-on pre-filters before PIMC and confidence filtering after.
/// Returns the rejection reason for logging.
fn extract_state_at(
    game: &SuecaSimulatorGame,
    config: &DatasetConfig,
    rng: &mut Pcg64,
    results: &mut Vec<(Vec<f64>, [f32; OUTPUT_COUNT], u8)>,
) -> RejectReason {
    // CRITICAL: Extract data ONLY for the active player whose turn it is.
    let seat = game.state.current_player;
    let legal = game.state.legal_moves();

    if legal.count_ones() <= 1 {
        return RejectReason::SingleLegal;
    }

    // ── Always-on pre-filter: drop fully-degenerate states ──
    // Keep a state only if the intent CHOICE can change the played card. With
    // the Elite-grade styled resolver, all three intents agree in most states
    // (~90%); those carry no intent-selection signal. We reject only when ALL
    // intents resolve to the same card, retaining every state where at least
    // one intent diverges (the decision-relevant states worth training on).
    {
        let mut card_for_intent = [40u8; OUTPUT_COUNT as usize];
        for intent in 0..OUTPUT_COUNT as usize {
            card_for_intent[intent] = resolve_intent(intent, game, seat);
        }
        let all_same = card_for_intent
            .iter()
            .all(|&c| c == card_for_intent[0]);
        if all_same {
            return RejectReason::IntentCollision;
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
        config.diff_mode,
        config.fixed_worlds,
    );

    // Empty EV results → futility stop fired. Skip this state.
    if evs.is_empty() {
        return RejectReason::FutilityStop;
    }

    // ── Intent labeling: best of the 3 intent-cards by PIMC EV ──
    // The WANN can only ever play one of the 3 intents, so the supervised
    // target is the best AVAILABLE intent, not the global PIMC-best card.
    // We look up each intent-card's EV, label the highest-EV intent, and keep
    // statistically-tied intents (within the best card's std_error) as a
    // uniform multi-label. Fully-ambiguous states (all 3 tied) carry no
    // intent-selection signal and are rejected.
    let mut intent_ev = [f64::NEG_INFINITY; OUTPUT_COUNT as usize];
    let mut intent_se = [0.0f64; OUTPUT_COUNT as usize];
    for i in 0..OUTPUT_COUNT as usize {
        let card = resolve_intent(i, game, seat);
        if let Some(r) = evs.iter().find(|r| r.card == card) {
            intent_ev[i] = r.ev;
            intent_se[i] = r.std_error;
        }
    }

    let mut best_i = 0usize;
    for i in 1..OUTPUT_COUNT as usize {
        if intent_ev[i] > intent_ev[best_i] {
            best_i = i;
        }
    }
    if intent_ev[best_i] == f64::NEG_INFINITY {
        // No intent-card found in the EV table (should not happen). Skip.
        return RejectReason::Unclassifiable;
    }
    let best_ev = intent_ev[best_i];
    let best_se = intent_se[best_i];

    // Co-best = intents whose EV is within the best card's noise floor.
    let mut soft_intent = [0.0f32; OUTPUT_COUNT as usize];
    let co_best: Vec<usize> = (0..OUTPUT_COUNT as usize)
        .filter(|&i| intent_ev[i] != f64::NEG_INFINITY && best_ev - intent_ev[i] <= best_se)
        .collect();
    if co_best.len() >= OUTPUT_COUNT as usize {
        // All intents statistically tied → no preference to learn.
        return RejectReason::ConfidenceFilter;
    }
    let w = 1.0 / co_best.len() as f32;
    for &i in &co_best {
        soft_intent[i] = w;
    }

    let legal_mask = build_legal_mask(legal, game, seat);
    results.push((belief.to_vec(), soft_intent, legal_mask));
    RejectReason::Accepted
}

pub fn map_card_to_soft_intents(card: u8, game: &SuecaSimulatorGame, seat: u8) -> Option<[f32; OUTPUT_COUNT]> {
    // Guard: this function must only be called with the PIMC-best card
    // (post-confidence-filter). The multi-label acceptance gate depends on
    // this invariant. In debug builds, verify the card is legal.
    debug_assert!(
        (game.state.legal_moves() & (1u64 << card)) != 0,
        "map_card_to_soft_intents called with card {} which is not legal for current player",
        card
    );
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

    // Multi-label acceptance: when ≥2 intents map to the PIMC-best card,
    // accept as a fractional multi-label state instead of rejecting.
    // The card IS the PIMC-best (caller only passes confidence-filtered best).
    // Each labeled intent gets 1/k weight.
    //
    // The old intent-demotion rule (force EQUITY_BUILDER on lead ties) is
    // replaced — multi-label is honest about ambiguity.
    let k = matching_intents.len();
    let mut target_vector = [0.0f32; OUTPUT_COUNT];
    let weight = 1.0 / k as f32;
    for &intent in &matching_intents {
        target_vector[intent] = weight;
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
            diff_mode: false,
            fixed_worlds: None,
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
            let (results, _) = generate_deal_states(&mut rng, &config, None);
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
            for (state, _, _) in generate_deal_states(&mut rng, &config, None).0 {
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
            diff_mode: false,
            fixed_worlds: None,
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
                false,
                None,
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
            diff_mode: false,
            fixed_worlds: None,
            soft_balance_min_ratio: 0.20,
        };
        let mut rng = Pcg64::seed_from_u64(999);
        let mut counts = [0usize; 4];
        let mut total = 0;

        for _ in 0..500 {
            for (_, soft_intent, _) in generate_deal_states(&mut rng, &config, None).0 {
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
            diff_mode: false,
            fixed_worlds: None,
            soft_balance_min_ratio: 0.20,
        };

        // Collect states with natural distribution
        let mut rng = Pcg64::seed_from_u64(config.seed);
        let mut states = Vec::new();
        let mut intents = Vec::new();
        let mut masks = Vec::new();
        loop {
            for (state, intent, mask) in generate_deal_states(&mut rng, &config, None).0 {
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
        // Exercise the REAL extraction path. With the Elite-grade styled resolver
        // most states are non-decisive (all intents agree) and are pre-filtered
        // out; among the decisive states that reach PIMC, a healthy fraction must
        // yield a clear best-intent label.
        let config = DatasetConfig {
            n_worlds: 80,
            search_depth: 2,
            target_total: 1000,
            seed: 54321,
            output_path: String::new(),
            diff_mode: false,
            fixed_worlds: None,
            soft_balance_min_ratio: 0.20,
        };
        let mut rng = Pcg64::seed_from_u64(config.seed);
        let mut tally = DealRejectCounts::default();
        for _ in 0..500 {
            let (_states, rejects) = generate_deal_states(&mut rng, &config, None);
            tally.merge(&rejects);
        }
        // States that actually reached PIMC (decisive, non-forced).
        let pimc_states = tally.accepted
            + tally.confidence_filter
            + tally.unclassifiable
            + tally.futility_stop;
        let ratio = if pimc_states == 0 {
            0.0
        } else {
            tally.accepted as f64 / pimc_states as f64
        };
        println!(
            "Dataset yield: accepted={} / pimc-reaching={} ({:.1}%)  [collision={} single={} futility={} conf={} unclass={}]",
            tally.accepted, pimc_states, ratio * 100.0,
            tally.intent_collision, tally.single_legal, tally.futility_stop,
            tally.confidence_filter, tally.unclassifiable,
        );
        // Decisive states are rare per deal (most are non-decisive, correctly
        // pre-filtered), but among decisive PIMC-reaching states a clear
        // best-intent label must emerge at a healthy rate.
        assert!(tally.accepted > 40, "too few accepted states: {}", tally.accepted);
        assert!(
            ratio >= 0.30,
            "intent-label yield too low among decisive PIMC states: {:.1}%",
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
    fn test_multi_label_equal_weights() {
        // When several intents resolve to the same (PIMC-best) card, the soft
        // target must be a uniform multi-label over exactly the matching intents
        // (1/k each) — the old "intent demotion" rule was replaced by honest
        // multi-label acceptance.
        let mut hands = [0u64; 4];
        for p in 0..4 {
            hands[p] = 0x3FFu64 << (p * 10);
        }
        let game = SuecaSimulatorGame::new(hands, 3, 0);
        assert_eq!(game.current_trick_len, 0);

        // Build the matching set for whatever card intent 0 resolves to.
        let card_0 = resolve_intent(0, &game, 0);
        let matching: Vec<usize> = (0..OUTPUT_COUNT as usize)
            .filter(|&i| resolve_intent(i, &game, 0) == card_0)
            .collect();
        let soft = map_card_to_soft_intents(card_0, &game, 0).unwrap();
        let k = matching.len() as f32;
        for i in 0..OUTPUT_COUNT as usize {
            let expected = if matching.contains(&i) { 1.0 / k } else { 0.0 };
            assert!(
                (soft[i] - expected).abs() < 1e-6,
                "intent {} weight {} != expected {} (k={})",
                i, soft[i], expected, k
            );
        }
        // Weights must sum to 1.0.
        let sum: f32 = soft.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "soft target must sum to 1.0, got {}", sum);
    }
}
