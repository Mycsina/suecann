// ===========================================================================
// Stage B expert dataset generator (2026-06-19 resolver/action-space overhaul)
// ===========================================================================
//
// Phase-0 target changed from 3-intent soft labels to a CARD-MATCH target over
// the free-card teacher: for each sampled ego-turn state we run the supra-Elite
// rollout teacher (`solve_pimc_rollout_serial`) to get an EV for every legal
// card, then store the bitmask of teacher-best cards (ties within one stderr).
// Each record also carries a compact `PhiCtx` so Phase-0 training can resolve
// the WANN's 6 knobs to a card via the shared φ-utility resolver and score it
// against the mask. See `heuristic::resolve_card_phi_utility_ctx`.
//
// The dealing + mid-trick random-walk + extra-extraction scaffolding is kept
// from the prior generator: it yields a realistic state mix (~25% lead / ~75%
// follow, spread across all trick numbers).

use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sueca_solver::belief::encode_belief_state;
use sueca_solver::constants::{INPUT_COUNT, PHI_FEATURE_COUNT};
use sueca_solver::engine::{CARD_SUIT};
use sueca_solver::heuristic::PhiCtx;
use sueca_solver::simulator::SuecaSimulatorGame;

/// Bumped whenever the φ feature set or record layout changes. The loader
/// rejects datasets whose version does not match — stale data must be
/// regenerated (same discipline as the belief-feature count gate).
pub const DATASET_VERSION: u32 = 2;

/// Number of extra states extracted per deal (advance 1-6 random cards between
/// each, without completing the trick, to preserve the ~25% lead fraction).
const N_EXTRA_EXTRACTIONS: usize = 6;

/// One sampled ego-turn state: belief input + teacher-best-cards mask + the
/// compact context needed to resolve the WANN's knobs to a card.
#[derive(Clone, Debug)]
pub struct StateRecord {
    pub belief: Vec<f64>, // length INPUT_COUNT
    /// Bitmask of teacher-best legal cards (cards within one stderr of the best
    /// rollout EV). The Phase-0 fitness is `1` iff the resolver's card is set.
    pub best_cards: u64,
    pub ctx: PhiCtx,
}

#[derive(Serialize, Deserialize)]
pub struct DatasetConfig {
    pub n_worlds: usize,
    pub search_depth: u8,
    pub target_total: usize,
    pub seed: u64,
    pub output_path: String,
    /// Vestigial under card-match fitness (no per-intent class balancing). Kept
    /// for CLI / config-file backward compatibility.
    #[serde(default = "default_soft_balance_min_ratio")]
    pub soft_balance_min_ratio: f64,
    /// If true, disable early termination and futility stop in PIMC. Diff/debug
    /// only; ignored by the rollout teacher.
    #[serde(default)]
    pub diff_mode: bool,
    /// When set, use exactly this many worlds per PIMC call (diff_mode only).
    #[serde(default)]
    pub fixed_worlds: Option<usize>,
    /// When true (canonical v6+), label with the flat Monte-Carlo rollout
    /// teacher (`solve_pimc_rollout`, Elite playouts) — supra-Elite. When false,
    /// use alpha-beta PIMC (only ties Elite). The free-card target is identical
    /// in shape either way; only the EV source differs.
    #[serde(default)]
    pub use_rollout_teacher: bool,
}

fn default_soft_balance_min_ratio() -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Checkpoint (bincode) — flat primitive vectors, no PhiCtx serde needed.
// ---------------------------------------------------------------------------
#[derive(Serialize, Deserialize)]
struct DatasetCheckpoint {
    states: Vec<f32>,
    best_cards: Vec<u64>,
    ctx_trump: Vec<u8>,
    ctx_hand: Vec<u64>,
    ctx_trick: Vec<u8>, // len N*4
    ctx_trick_len: Vec<u8>,
    batch_seed: u64,
    batches: usize,
    n_worlds: usize,
    search_depth: u8,
    target_total: usize,
    seed: u64,
    use_rollout_teacher: bool,
}

const CHECKPOINT_INTERVAL: usize = 5; // save every 5 batches

fn checkpoint_path(output_path: &str) -> String {
    output_path
        .strip_suffix(".npz")
        .unwrap_or(output_path)
        .to_string()
        + ".checkpoint"
}

fn save_checkpoint(ckpt: &DatasetCheckpoint, path: &str) {
    let file = std::fs::File::create(path)
        .unwrap_or_else(|e| panic!("Cannot create checkpoint '{}': {}", path, e));
    bincode::serialize_into(std::io::BufWriter::new(file), ckpt)
        .unwrap_or_else(|e| panic!("Cannot write checkpoint '{}': {}", path, e));
}

fn load_checkpoint(path: &str) -> Option<DatasetCheckpoint> {
    let file = std::fs::File::open(path).ok()?;
    bincode::deserialize_from(std::io::BufReader::new(file)).ok()
}

// ---------------------------------------------------------------------------
// Per-state extraction (the teacher label)
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, Default)]
pub struct DealRejectCounts {
    pub accepted: usize,
    pub single_legal: usize,
    pub futility_stop: usize,
    pub all_tied: usize,
    pub terminal: usize,
    pub lead_primary: usize,
    pub follow_primary: usize,
}

/// Run the teacher at the current ego decision point and produce a record if
/// the state carries a non-trivial card-selection signal.
fn extract_state(
    game: &SuecaSimulatorGame,
    config: &DatasetConfig,
    rng: &mut Pcg64,
    out: &mut Vec<StateRecord>,
) -> bool {
    let seat = game.state.current_player;
    let legal = game.state.legal_moves();

    // Skip forced positions (a single legal card carries no selection signal).
    if legal.count_ones() < 2 {
        return false;
    }

    let belief = encode_belief_state(game, seat);
    let ctx = PhiCtx::from_game(game, seat);

    // Teacher: per-legal-card EV. solve_pimc_rollout_serial is safe inside a
    // rayon par_iter (the generation batch is parallelized across deals).
    let evs = if config.use_rollout_teacher {
        sueca_solver::pimc::solve_pimc_rollout_serial(game, config.n_worlds, rng.gen_range(0u64..u64::MAX))
    } else {
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
        let trick_cards = &game.current_trick[..game.current_trick_len];
        sueca_solver::pimc::solve_pimc(
            seat,
            game.state.hands[seat as usize],
            played_cards_mask,
            game.voids,
            target_sizes,
            game.state.trump,
            led_suit,
            trick_cards,
            seat,
            game.state.current_trick_winner,
            game.state.current_trick_best_card,
            [game.state.team_02_score, game.state.team_13_score],
            game.state.trick_number,
            config.n_worlds,
            config.search_depth,
            rng.gen_range(0u64..u64::MAX),
            config.diff_mode,
            config.fixed_worlds,
        )
    };

    if evs.is_empty() {
        return false;
    }

    // Best EV and its stderr.
    let mut best_ev = f64::NEG_INFINITY;
    let mut best_se = 0.0f64;
    for r in &evs {
        if r.ev > best_ev {
            best_ev = r.ev;
            best_se = r.std_error;
        }
    }

    // best_cards = legal cards whose EV is within one stderr of the best
    // (statistical ties are multi-label — the resolver gets full credit for any).
    let mut best_cards = 0u64;
    for r in &evs {
        if best_ev - r.ev <= best_se {
            best_cards |= 1u64 << r.card;
        }
    }

    // Reject fully-ambiguous states (every legal card tied → no preference).
    if best_cards.count_ones() as u32 == legal.count_ones() {
        return false;
    }

    out.push(StateRecord {
        belief: belief.to_vec(),
        best_cards,
        ctx,
    });
    true
}

/// Generate states for one deal: deal, random-walk to a uniform point within a
/// trick, extract the primary state, then `N_EXTRA_EXTRACTIONS` extra states.
/// Public so diagnostics/tests can drive a single deal deterministically.
pub fn generate_deal_states(
    rng: &mut Pcg64,
    config: &DatasetConfig,
) -> (Vec<StateRecord>, DealRejectCounts) {
    let mut results = Vec::new();
    let mut rejects = DealRejectCounts::default();

    // ── Deal 40 cards across 4 hands ──
    let mut deck: Vec<u8> = (0..40).collect();
    for i in (1..40).rev() {
        let j = rng.gen_range(0u64..((i + 1) as u64)) as usize;
        deck.swap(i, j);
    }
    let mut hands = [0u64; 4];
    for p in 0..4 {
        for c in 0..10 {
            hands[p] |= 1u64 << deck[p * 10 + c];
        }
    }
    let trump = rng.gen_range(0u64..4) as u8;
    let first_player = rng.gen_range(0u64..4) as u8;

    // ── Mid-trick random walk to a uniform point within a trick ──
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
        let idx = rng.gen_range(0u64..(count as u64)) as usize;
        let mut temp = legal;
        for _ in 0..idx {
            temp &= temp - 1;
        }
        game.play_card(temp.trailing_zeros() as u8);
    }
    if game.state.trick_number >= 10 {
        rejects.terminal += 1;
        return (results, rejects);
    }

    // ── Primary extraction ──
    let primary_is_lead = game.current_trick_len == 0;
    if extract_state(&game, config, rng, &mut results) {
        rejects.accepted += 1;
        if primary_is_lead {
            rejects.lead_primary += 1;
        } else {
            rejects.follow_primary += 1;
        }
    } else if game.state.legal_moves().count_ones() < 2 {
        rejects.single_legal += 1;
    } else {
        rejects.all_tied += 1;
    }

    // ── Extra extractions: advance 1-6 random cards, then extract ──
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
            let legal = extra_game.state.legal_moves();
            if legal == 0 {
                break;
            }
            let count = legal.count_ones();
            let idx = rng.gen_range(0u64..(count as u64)) as usize;
            let mut temp = legal;
            for _ in 0..idx {
                temp &= temp - 1;
            }
            extra_game.play_card(temp.trailing_zeros() as u8);
        }
        if extra_game.state.trick_number >= 10 {
            break;
        }
        // Use a heuristic move to advance past a position without random-walking
        // into an identical distribution (keeps extras decorrelated from primary).
        let _ = extract_state(&extra_game, config, rng, &mut results);
    }

    (results, rejects)
}

// ---------------------------------------------------------------------------
// Top-level driver
// ---------------------------------------------------------------------------
pub fn generate_dataset(config: &DatasetConfig, resume: bool) {
    let target = config.target_total;
    let ckpt_path = checkpoint_path(&config.output_path);

    println!(
        "Generating Stage-B card-match dataset: {} worlds, {} target states (teacher: {})...",
        config.n_worlds,
        target,
        if config.use_rollout_teacher { "rollout" } else { "alphabeta" }
    );

    let (mut states, mut best_cards, mut ctx_trump, mut ctx_hand, mut ctx_trick, mut ctx_trick_len, batch_seed, mut batches) =
        if resume {
            match load_checkpoint(&ckpt_path) {
                Some(ckpt)
                    if ckpt.n_worlds == config.n_worlds
                        && ckpt.search_depth == config.search_depth
                        && ckpt.target_total == config.target_total
                        && ckpt.seed == config.seed
                        && ckpt.use_rollout_teacher == config.use_rollout_teacher =>
                {
                    let n = ckpt.best_cards.len();
                    println!("Resumed from checkpoint: {} states, {} batches done.", n, ckpt.batches);
                    (
                        ckpt.states,
                        ckpt.best_cards,
                        ckpt.ctx_trump,
                        ckpt.ctx_hand,
                        ckpt.ctx_trick,
                        ckpt.ctx_trick_len,
                        ckpt.batch_seed,
                        ckpt.batches,
                    )
                }
                _ => {
                    eprintln!("Checkpoint config mismatch — starting fresh.");
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), config.seed, 0)
                }
            }
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), config.seed, 0)
        };

    let batch_deals = 64usize;
    let mut tot_rejects = DealRejectCounts::default();

    while best_cards.len() < target {
        // Parallel batch of deals. Each deal uses solve_pimc_rollout_serial
        // (serial internally) so there is no nested-raylon deadlock.
        let base = batch_seed.wrapping_add(batches as u64 * batch_deals as u64);
        let collected: Vec<(Vec<StateRecord>, DealRejectCounts)> = (0..batch_deals)
            .into_par_iter()
            .map(|i| {
                let mut rng = Pcg64::seed_from_u64(base.wrapping_mul(2654435761u64).wrapping_add(i as u64));
                generate_deal_states(&mut rng, config)
            })
            .collect();

        let mut accepted_here = 0usize;
        for (recs, rej) in collected {
            tot_rejects.accepted += rej.accepted;
            tot_rejects.lead_primary += rej.lead_primary;
            tot_rejects.follow_primary += rej.follow_primary;
            tot_rejects.single_legal += rej.single_legal;
            tot_rejects.all_tied += rej.all_tied;
            tot_rejects.futility_stop += rej.futility_stop;
            tot_rejects.terminal += rej.terminal;
            for r in recs {
                if best_cards.len() >= target {
                    break;
                }
                states.extend(r.belief.iter().map(|&v| v as f32));
                best_cards.push(r.best_cards);
                ctx_trump.push(r.ctx.trump);
                ctx_hand.push(r.ctx.hand);
                ctx_trick.extend_from_slice(&r.ctx.trick_cards);
                ctx_trick_len.push(r.ctx.trick_len);
                accepted_here += 1;
            }
        }
        // (accepted_here is the total states stored this batch = primary + extra
        // extractions; do NOT add it to tot_rejects.accepted, which counts only
        // primary acceptances so the lead/follow percentages below sum to ~100%.)
        batches += 1;

        if batches % CHECKPOINT_INTERVAL == 0 {
            save_checkpoint(
                &DatasetCheckpoint {
                    states: states.clone(),
                    best_cards: best_cards.clone(),
                    ctx_trump: ctx_trump.clone(),
                    ctx_hand: ctx_hand.clone(),
                    ctx_trick: ctx_trick.clone(),
                    ctx_trick_len: ctx_trick_len.clone(),
                    batch_seed,
                    batches,
                    n_worlds: config.n_worlds,
                    search_depth: config.search_depth,
                    target_total: config.target_total,
                    seed: config.seed,
                    use_rollout_teacher: config.use_rollout_teacher,
                },
                &ckpt_path,
            );
            println!(
                "  batch {}: {} states (target {}), {} batches done — checkpoint saved",
                batches,
                best_cards.len(),
                target,
                batches
            );
        }
    }

    let n = best_cards.len();
    println!(
        "Collected {} states. Split (primary, this run): lead {:.0}% / follow {:.0}%.",
        n,
        if tot_rejects.accepted == 0 { 0.0 } else { 100.0 * tot_rejects.lead_primary as f64 / tot_rejects.accepted as f64 },
        if tot_rejects.accepted == 0 { 0.0 } else { 100.0 * tot_rejects.follow_primary as f64 / tot_rejects.accepted as f64 },
    );

    save_npz(
        &states,
        &best_cards,
        &ctx_trump,
        &ctx_hand,
        &ctx_trick,
        &ctx_trick_len,
        n,
        &config.output_path,
    );
    let _ = std::fs::remove_file(&ckpt_path); // success — drop checkpoint
    println!("Wrote {} states to {}", n, config.output_path);
}

// ---------------------------------------------------------------------------
// npz serialization
// ---------------------------------------------------------------------------
fn write_npy(data: &[u8], dtype: &str, shape: &[u64]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"\x93NUMPY");
    buf.extend_from_slice(&[1u8, 0u8]);
    let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
    let header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': ({},)}}",
        dtype,
        shape_str.join(",")
    );
    let mut header_bytes = header.as_bytes().to_vec();
    let total_header_len = 10 + header_bytes.len();
    let pad = (16 - (total_header_len % 16)) % 16;
    header_bytes.extend(std::iter::repeat_n(b' ', pad));
    header_bytes.push(b'\n');
    buf.extend_from_slice(&(header_bytes.len() as u16).to_le_bytes());
    buf.extend_from_slice(&header_bytes);
    buf.extend_from_slice(data);
    buf
}

#[allow(clippy::too_many_arguments)]
fn save_npz(
    states: &[f32],
    best_cards: &[u64],
    ctx_trump: &[u8],
    ctx_hand: &[u64],
    ctx_trick: &[u8],
    ctx_trick_len: &[u8],
    n: usize,
    path: &str,
) {
    use std::io::Write;
    use zip::write::FileOptions;

    let zip_path = if path.ends_with(".npz") { path.to_string() } else { format!("{}.npz", path) };
    let file = std::fs::File::create(&zip_path).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let opts = FileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    macro_rules! raw_bytes {
        ($v:expr) => {{
            let v = $v;
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * std::mem::size_of_val(&v[0])) }
        }};
    }

    // states.npy  (N, INPUT_COUNT) f32
    zip.start_file("states.npy", opts).unwrap();
    zip.write_all(&write_npy(raw_bytes!(states), "<f4", &[n as u64, INPUT_COUNT as u64])).unwrap();
    // best_cards.npy  (N,) u64
    zip.start_file("best_cards.npy", opts).unwrap();
    zip.write_all(&write_npy(raw_bytes!(best_cards), "<u8", &[n as u64])).unwrap();
    // ctx_trump.npy  (N,) u8
    zip.start_file("ctx_trump.npy", opts).unwrap();
    zip.write_all(&write_npy(ctx_trump, "<u1", &[n as u64])).unwrap();
    // ctx_hand.npy  (N,) u64
    zip.start_file("ctx_hand.npy", opts).unwrap();
    zip.write_all(&write_npy(raw_bytes!(ctx_hand), "<u8", &[n as u64])).unwrap();
    // ctx_trick.npy  (N, 4) u8
    zip.start_file("ctx_trick.npy", opts).unwrap();
    zip.write_all(&write_npy(ctx_trick, "<u1", &[n as u64, 4u64])).unwrap();
    // ctx_trick_len.npy  (N,) u8
    zip.start_file("ctx_trick_len.npy", opts).unwrap();
    zip.write_all(&write_npy(ctx_trick_len, "<u1", &[n as u64])).unwrap();
    // version.npy  (1,) u32 — staleness gate
    let ver = [DATASET_VERSION];
    zip.start_file("version.npy", opts).unwrap();
    zip.write_all(&write_npy(raw_bytes!(&ver[..]), "<u4", &[1u64])).unwrap();

    zip.finish().unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deal_generates_some_records() {
        let config = DatasetConfig {
            n_worlds: 10,
            search_depth: 2,
            target_total: 0,
            seed: 7,
            output_path: String::new(),
            soft_balance_min_ratio: 0.0,
            diff_mode: false,
            fixed_worlds: None,
            use_rollout_teacher: true,
        };
        let mut rng = Pcg64::seed_from_u64(7);
        let (recs, rejects) = generate_deal_states(&mut rng, &config);
        // We cannot assert >0 records for every deal (some reject), but the
        // machinery must run and return well-formed records when present.
        for r in &recs {
            assert_eq!(r.belief.len(), INPUT_COUNT);
            assert_ne!(r.best_cards, 0);
            assert!(r.best_cards.count_ones() >= 1);
        }
        let _ = rejects; // accepted/terminal/etc. are informational
    }

    #[test]
    fn test_best_cards_mask_respects_legality() {
        // The teacher-best mask must only ever set bits for legal cards of the
        // ego player at the sampled state — never an illegal card.
        let config = DatasetConfig {
            n_worlds: 10,
            search_depth: 2,
            target_total: 0,
            seed: 99,
            output_path: String::new(),
            soft_balance_min_ratio: 0.0,
            diff_mode: false,
            fixed_worlds: None,
            use_rollout_teacher: true,
        };
        let mut rng = Pcg64::seed_from_u64(99);
        for _ in 0..16 {
            let (recs, _) = generate_deal_states(&mut rng, &config);
            for r in &recs {
                let legal = r.ctx.legal();
                assert_eq!(r.best_cards & !legal, 0, "best_cards sets an illegal bit");
                assert_ne!(r.best_cards & legal, 0, "best_cards has no legal card");
            }
        }
    }

    /// Bit-exact save → load round-trip: every field that training reads must
    /// survive the npz serialization unchanged (incl. the u64 masks/hands and
    /// the 4-byte-per-state ctx_trick packing), and the version gate must pass.
    #[test]
    fn dataset_save_load_roundtrip_is_exact() {
        let n = 12usize;
        // Build deterministic, non-trivial arrays covering all card slots.
        let states: Vec<f32> = (0..(n * INPUT_COUNT)).map(|i| (i as f32) / 100.0).collect();
        let best_cards: Vec<u64> = (0..n).map(|i| 0x1Fu64 << (i as u64 % 8)).collect();
        let ctx_trump: Vec<u8> = (0..n).map(|i| (i % 4) as u8).collect();
        let ctx_hand: Vec<u64> = (0..n).map(|i| 0xDEAD_BEEF_u64.wrapping_add(i as u64)).collect();
        let ctx_trick: Vec<u8> = (0..(n * 4)).map(|i| (i % 40) as u8).collect();
        let ctx_trick_len: Vec<u8> = (0..n).map(|i| (i % 4) as u8).collect();

        let path = "/tmp/stageb_roundtrip_test.npz";
        save_npz(
            &states, &best_cards, &ctx_trump, &ctx_hand, &ctx_trick, &ctx_trick_len, n, path,
        );

        let loaded = crate::dataset::load_expert_dataset(path).expect("load failed");
        assert_eq!(loaded.num_states, n, "num_states mismatch");
        assert_eq!(loaded.best_cards, best_cards, "best_cards not bit-exact");
        assert_eq!(loaded.ctx_hand, ctx_hand, "ctx_hand not bit-exact");
        assert_eq!(loaded.ctx_trump, ctx_trump, "ctx_trump not bit-exact");
        assert_eq!(loaded.ctx_trick, ctx_trick, "ctx_trick packing corrupted");
        assert_eq!(loaded.ctx_trick_len, ctx_trick_len, "ctx_trick_len not bit-exact");
        // Beliefs survive the f32→f64 round trip to single precision.
        for (a, b) in loaded.states.iter().zip(states.iter()) {
            assert!((*a as f32 - *b).abs() < 1e-6, "belief value drifted");
        }
        // The reconstructed PhiCtx must read back the exact ctx fields.
        for i in 0..n {
            let ctx = loaded.ctx(i);
            assert_eq!(ctx.trump, ctx_trump[i]);
            assert_eq!(ctx.hand, ctx_hand[i]);
            assert_eq!(ctx.trick_len, ctx_trick_len[i]);
            assert_eq!(&ctx.trick_cards[..], &ctx_trick[i * 4..i * 4 + 4]);
        }
        let _ = std::fs::remove_file(path);
    }

    /// The version gate must REJECT a dataset missing the version tag (or with
    /// a stale version), so old v1 intent datasets fail loudly rather than load
    /// as garbage.
    #[test]
    fn loader_rejects_missing_version_tag() {
        // An npz with states but no version.npy → the loader must error.
        let path = "/tmp/stageb_noversion_test.npz";
        use std::io::Write;
        use zip::write::FileOptions;
        let file = std::fs::File::create(path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        let opts = zip::write::FileOptions::default();
        zip.start_file("states.npy", opts).unwrap();
        // minimal valid npy header for an empty f32 array
        zip.write_all(b"\x93NUMPY\x01\x00").unwrap();
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (0, 35),}";
        let mut hb = header.as_bytes().to_vec();
        while (10 + hb.len()) % 16 != 0 { hb.push(b' '); }
        hb.push(b'\n');
        zip.write_all(&(hb.len() as u16).to_le_bytes()).unwrap();
        zip.write_all(&hb).unwrap();
        zip.finish().unwrap();
        let res = crate::dataset::load_expert_dataset(path);
        assert!(res.is_err(), "loader accepted a dataset without a version tag");
        let _ = std::fs::remove_file(path);
    }
}

// Silence the unused-import warning for PHI_FEATURE_COUNT (kept for clarity so
// readers see the knob dimension the dataset feeds).
#[allow(dead_code)]
const _PHI_KNOB_COUNT: usize = PHI_FEATURE_COUNT;
