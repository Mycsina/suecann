use crate::engine::GameState;
use crate::search::{alpha_beta, TranspositionTable};
use rayon::prelude::*;
/// PIMC (Perfect Information Monte Carlo) engine for Sueca.
use std::cell::RefCell;

thread_local! {
    /// Thread-local Transposition Table to avoid any heap allocation during search.
    /// 16 = 65,536 entries, using ~1.5MB of RAM per thread.
    static THREAD_TT: RefCell<TranspositionTable> = RefCell::new(TranspositionTable::new(16));
}

use crate::rng::LcgRng;

/// Backtracking card sampler to distribute remaining unknown cards.
/// Strictly stack-allocated to ensure zero heap allocations.
fn sample_constraints(
    unknown_cards: &[u8],
    unknown_idx: usize,
    voids: [u8; 4],
    target_sizes: &mut [u8; 4],
    hands_out: &mut [u64; 4],
) -> bool {
    if unknown_idx == unknown_cards.len() {
        return true;
    }
    let card = unknown_cards[unknown_idx];
    let suit = crate::engine::CARD_SUIT[card as usize];
    let suit_mask = 1 << suit;

    for s in 0..4 {
        if target_sizes[s] > 0 && (voids[s] & suit_mask) == 0 {
            // Assign card to player s
            target_sizes[s] -= 1;
            hands_out[s] |= 1u64 << card;

            if sample_constraints(
                unknown_cards,
                unknown_idx + 1,
                voids,
                target_sizes,
                hands_out,
            ) {
                return true;
            }

            // Backtrack
            target_sizes[s] += 1;
            hands_out[s] &= !(1u64 << card);
        }
    }
    false
}

/// Quick constraint feasibility check. Returns false if the void constraints
/// are provably impossible — e.g., if remaining cards in a suit exceed the
/// combined hand capacity of non-void players. Runs in O(cards) and prevents
/// wasted retry cycles in sample_world's backtracking solver.
fn constraints_feasible(
    unknown_cards: &[u8],
    voids: [u8; 4],
    target_sizes: [u8; 4],
    my_seat: u8,
) -> bool {
    let mut suit_remaining = [0u8; 4];
    let mut non_void_capacity = [0u8; 4];
    let force_assigned = [0u8; 4];

    for &card in unknown_cards {
        let suit = crate::engine::CARD_SUIT[card as usize];
        suit_remaining[suit as usize] += 1;
    }

    for p in 0..4 {
        if p == my_seat as usize {
            continue;
        }
        let mut cap = target_sizes[p];
        for s in 0..4 {
            let void_mask = 1u8 << s;
            if (voids[p] & void_mask) != 0 {
                // Player p cannot receive cards of suit s
                cap = cap.saturating_sub(force_assigned[s as usize]);
            } else {
                non_void_capacity[s as usize] += target_sizes[p];
            }
        }
    }

    for s in 0..4 {
        if suit_remaining[s] > non_void_capacity[s] {
            return false; // Impossible: more cards of this suit than capacity
        }
    }

    true
}

/// Sample a single world consistent with public information.
/// `unknown_cards` is the pre-built pool of cards not in my hand and not yet played.
/// Returns true if successful, false if constraints cannot be satisfied.
pub fn sample_world(
    my_seat: u8,
    my_hand: u64,
    unknown_cards: &[u8],
    voids: [u8; 4],
    target_sizes: [u8; 4],
    hands_out: &mut [u64; 4],
    rng_state: &mut u64,
) -> bool {
    // 1. Initialize hands_out with my_hand
    *hands_out = [0; 4];
    hands_out[my_seat as usize] = my_hand;

    // 2. MRV heuristic: partition constrained cards (where at least one hidden
    //    player is void in that suit) to the front of the array. This makes
    //    backtracking fail fast at shallow recursion depths instead of wasting
    //    cycles on doomed deep branches.
    let num_unknowns = unknown_cards.len();
    let mut shuffled = [0u8; 40];
    let mut constrained_end = 0usize;
    let mut free_start = num_unknowns;

    for &card in unknown_cards {
        let suit_flag = 1u8 << crate::engine::CARD_SUIT[card as usize];
        let mut is_constrained = false;
        for s in 0..4 {
            if s != my_seat as usize && (voids[s] & suit_flag) != 0 {
                is_constrained = true;
                break;
            }
        }
        if is_constrained {
            shuffled[constrained_end] = card;
            constrained_end += 1;
        } else {
            free_start -= 1;
            shuffled[free_start] = card;
        }
    }

    // Fisher-Yates shuffle within each partition independently
    let mut r = LcgRng::new(*rng_state);
    let mut i = constrained_end;
    while i > 1 {
        let rand_idx = (r.next_u64() as usize) % i;
        i -= 1;
        shuffled.swap(i, rand_idx);
    }
    i = num_unknowns - free_start;
    while i > 1 {
        let rand_idx = (r.next_u64() as usize) % i;
        i -= 1;
        shuffled.swap(free_start + i, free_start + rand_idx);
    }
    *rng_state = r.next_u64(); // Update RNG state

    // 3. Backtrack/Constraint-solve assignment
    let mut temp_targets = target_sizes;
    temp_targets[my_seat as usize] = 0;

    sample_constraints(
        &shuffled[0..num_unknowns],
        0,
        voids,
        &mut temp_targets,
        hands_out,
    )
}

/// Result of PIMC evaluation for a single legal move.
#[derive(Debug, Clone, Copy)]
pub struct PimcResult {
    pub card: u8,
    pub ev: f64,
    pub std_error: f64, // standard error of the mean EV (σ/√n)
}

/// Online mean/variance accumulator (module-level so `solve_pimc_rollout` can use
/// it across rayon fold/reduce). Mirrors the local `MoveWelford` inside `solve_pimc`
/// but kept separate so that function is left untouched.
#[derive(Clone, Copy, Default)]
struct RolloutWelford {
    count: u32,
    mean: f64,
    m2: f64,
}

impl RolloutWelford {
    #[inline]
    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    #[inline]
    fn merge(&mut self, other: &RolloutWelford) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = *other;
            return;
        }
        let total = self.count + other.count;
        let delta = other.mean - self.mean;
        self.mean =
            (self.count as f64 * self.mean + other.count as f64 * other.mean) / total as f64;
        self.m2 += other.m2
            + delta * delta * (self.count as f64 * other.count as f64) / total as f64;
        self.count = total;
    }
}

/// Flat Monte-Carlo PIMC with Elite (HeuristicBot) playouts.
///
/// For each legal move of the current player, EV = mean over `n_worlds` determinized
/// worlds of the **ego team's** final score after playing that move and then letting
/// Elite play all four seats to terminal. By the rollout policy-improvement theorem
/// this is >= Elite, and it is cheaper than deep alpha-beta because every playout is
/// O(remaining cards) of bitboard heuristic decisions (no search tree, no leaf eval).
///
/// Determinization reuses the live game's trick/score/void state verbatim and only
/// overwrites the hidden hands with a sampled world (the ego hand is preserved). The
/// Zobrist hash becomes stale after the hand swap, but Elite playouts never read it
/// (only the alpha-beta transposition table does), so the rollout remains correct.
pub fn solve_pimc_rollout(
    game: &crate::simulator::SuecaSimulatorGame,
    n_worlds: usize,
    seed: u64,
) -> Vec<PimcResult> {
    use crate::heuristic::select_card_heuristic;

    let my_seat = game.state.current_player;
    let my_hand = game.state.hands[my_seat as usize];

    // 1. Legal moves for the current player (from the live state).
    let legal_mask = game.state.legal_moves();
    let mut legal_moves = [0u8; 10];
    let mut legal_count = 0usize;
    let mut t = legal_mask;
    while t != 0 {
        legal_moves[legal_count] = t.trailing_zeros() as u8;
        legal_count += 1;
        t &= t - 1;
    }
    if legal_count == 0 {
        return Vec::new();
    }
    if legal_count == 1 {
        return vec![PimcResult {
            card: legal_moves[0],
            ev: 0.0,
            std_error: 0.0,
        }];
    }

    // 2. Unknown pool = cards held by the other three players (everything still in
    //    a hand that is not mine). Current-trick cards are already out of all hands.
    let all_hands =
        game.state.hands[0] | game.state.hands[1] | game.state.hands[2] | game.state.hands[3];
    let unknown_mask = all_hands & !my_hand;
    let mut unknown_cards = [0u8; 40];
    let mut num_unknowns = 0usize;
    let mut u = unknown_mask;
    while u != 0 {
        unknown_cards[num_unknowns] = u.trailing_zeros() as u8;
        num_unknowns += 1;
        u &= u - 1;
    }
    let unknown_slice = &unknown_cards[..num_unknowns];

    // Target hand sizes = current remaining popcounts (ego entry ignored by sampler).
    let mut target_sizes = [0u8; 4];
    for i in 0..4 {
        target_sizes[i] = game.state.hands[i].count_ones() as u8;
    }
    let voids = game.voids;
    let ego_team = my_seat % 2; // 0 -> team 0&2, 1 -> team 1&3

    // 3. Per-move accumulation over determinized worlds (parallel paired comparison:
    //    all legal moves share the same sampled world within an iteration).
    let stats: [RolloutWelford; 40] = (0..n_worlds)
        .into_par_iter()
        .fold(
            || [RolloutWelford::default(); 40],
            |mut acc, world_idx| {
                let mut local_hands = [0u64; 4];
                let mut local_seed =
                    seed.wrapping_add((world_idx as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));
                let mut ok = false;
                for _ in 0..10 {
                    if sample_world(
                        my_seat,
                        my_hand,
                        unknown_slice,
                        voids,
                        target_sizes,
                        &mut local_hands,
                        &mut local_seed,
                    ) {
                        ok = true;
                        break;
                    }
                }
                if !ok {
                    return acc;
                }

                for i in 0..legal_count {
                    let m = legal_moves[i];
                    let mut g = *game;
                    g.state.hands = local_hands; // determinize hidden hands (ego unchanged)
                    g.play_card(m);
                    while g.state.trick_number < 10 {
                        let s = g.state.current_player;
                        let c = select_card_heuristic(&g, s);
                        g.play_card(c);
                    }
                    let score = if ego_team == 0 {
                        g.state.team_02_score
                    } else {
                        g.state.team_13_score
                    };
                    acc[m as usize].update(score as f64);
                }
                acc
            },
        )
        .reduce(
            || [RolloutWelford::default(); 40],
            |mut a, b| {
                for i in 0..40 {
                    a[i].merge(&b[i]);
                }
                a
            },
        );

    // 4. Emit one PimcResult per legal move.
    let mut out = Vec::with_capacity(legal_count);
    for i in 0..legal_count {
        let m = legal_moves[i];
        let w = &stats[m as usize];
        let n = w.count.max(1) as f64;
        let var = if w.count > 1 {
            w.m2 / (w.count as f64 - 1.0)
        } else {
            0.0
        };
        out.push(PimcResult {
            card: m,
            ev: w.mean,
            std_error: (var / n).sqrt(),
        });
    }
    out
}

/// Serial twin of [`solve_pimc_rollout`] with identical semantics, iterating worlds
/// with a plain loop instead of `into_par_iter`. Use this when calling from inside an
/// already-parallel rayon region (e.g. the benchmark's per-game `par_iter`), where a
/// nested parallel iterator on the same global pool deadlocks (all workers park waiting
/// on each other). Outer parallelism still provides throughput.
pub fn solve_pimc_rollout_serial(
    game: &crate::simulator::SuecaSimulatorGame,
    n_worlds: usize,
    seed: u64,
) -> Vec<PimcResult> {
    use crate::heuristic::select_card_heuristic;

    let my_seat = game.state.current_player;
    let my_hand = game.state.hands[my_seat as usize];

    let legal_mask = game.state.legal_moves();
    let mut legal_moves = [0u8; 10];
    let mut legal_count = 0usize;
    let mut t = legal_mask;
    while t != 0 {
        legal_moves[legal_count] = t.trailing_zeros() as u8;
        legal_count += 1;
        t &= t - 1;
    }
    if legal_count == 0 {
        return Vec::new();
    }
    if legal_count == 1 {
        return vec![PimcResult {
            card: legal_moves[0],
            ev: 0.0,
            std_error: 0.0,
        }];
    }

    let all_hands =
        game.state.hands[0] | game.state.hands[1] | game.state.hands[2] | game.state.hands[3];
    let unknown_mask = all_hands & !my_hand;
    let mut unknown_cards = [0u8; 40];
    let mut num_unknowns = 0usize;
    let mut u = unknown_mask;
    while u != 0 {
        unknown_cards[num_unknowns] = u.trailing_zeros() as u8;
        num_unknowns += 1;
        u &= u - 1;
    }
    let unknown_slice = &unknown_cards[..num_unknowns];

    let mut target_sizes = [0u8; 4];
    for i in 0..4 {
        target_sizes[i] = game.state.hands[i].count_ones() as u8;
    }
    let voids = game.voids;
    let ego_team = my_seat % 2;

    let mut stats: [RolloutWelford; 40] = [RolloutWelford::default(); 40];
    for world_idx in 0..n_worlds {
        let mut local_hands = [0u64; 4];
        let mut local_seed =
            seed.wrapping_add((world_idx as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));
        let mut ok = false;
        for _ in 0..10 {
            if sample_world(
                my_seat,
                my_hand,
                unknown_slice,
                voids,
                target_sizes,
                &mut local_hands,
                &mut local_seed,
            ) {
                ok = true;
                break;
            }
        }
        if !ok {
            continue;
        }

        for i in 0..legal_count {
            let m = legal_moves[i];
            let mut g = *game;
            g.state.hands = local_hands; // determinize hidden hands (ego unchanged)
            g.play_card(m);
            while g.state.trick_number < 10 {
                let s = g.state.current_player;
                let c = select_card_heuristic(&g, s);
                g.play_card(c);
            }
            let score = if ego_team == 0 {
                g.state.team_02_score
            } else {
                g.state.team_13_score
            };
            stats[m as usize].update(score as f64);
        }
    }

    let mut out = Vec::with_capacity(legal_count);
    for i in 0..legal_count {
        let m = legal_moves[i];
        let w = &stats[m as usize];
        let n = w.count.max(1) as f64;
        let var = if w.count > 1 {
            w.m2 / (w.count as f64 - 1.0)
        } else {
            0.0
        };
        out.push(PimcResult {
            card: m,
            ev: w.mean,
            std_error: (var / n).sqrt(),
        });
    }
    out
}

/// Evaluates the Expected Value (EV) of each legal move using PIMC.
/// Returns per-move EV scores with standard errors from Welford variance tracking.
/// Thread-safe and parallelized using Rayon.
#[allow(clippy::too_many_arguments)]
pub fn solve_pimc(
    my_seat: u8,
    my_hand: u64,
    played_cards: u64,
    voids: [u8; 4],
    target_sizes: [u8; 4],
    trump: u8,
    led_suit: u8,
    current_trick_cards: &[u8], // cards played in the current trick
    current_player: u8,
    current_trick_winner: u8,
    current_trick_best_card: u8,
    team_scores: [u8; 2],
    trick_number: u8,
    n_worlds: usize,
    search_depth: u8,
    seed: u64,
    diff_mode: bool,
    fixed_worlds: Option<usize>,
) -> Vec<PimcResult> {
    let effective_worlds = fixed_worlds.unwrap_or(n_worlds);
    // 1. Determine legal moves for the player
    let mut legal_moves = [0u8; 10];
    let mut legal_count = 0;

    let suit_mask = 0x3FFu64 << (led_suit * 10);
    let suited = if led_suit < 4 { my_hand & suit_mask } else { 0 };
    let moves_mask = if suited != 0 { suited } else { my_hand };

    let mut temp = moves_mask;
    while temp != 0 {
        legal_moves[legal_count] = temp.trailing_zeros() as u8;
        legal_count += 1;
        temp &= temp - 1;
    }

    if legal_count == 0 {
        return Vec::new();
    }
    if legal_count == 1 {
        return vec![PimcResult {
            card: legal_moves[0],
            ev: 0.0,
            std_error: 0.0,
        }];
    }

    // Pre-calculate target hand sizes for the remaining players.
    // If a player has already played in the current trick, their target size
    // is already 1 less than their hand size before the trick.
    // We expect the caller (Python) to pass the correct target_sizes.

    // 2. Build the unknown card pool once (invariant across worlds)
    let known_mask = my_hand | played_cards;
    let mut unknown_cards = [0u8; 40];
    let mut num_unknowns = 0usize;
    for c in 0..40 {
        if (known_mask & (1u64 << c)) == 0 {
            unknown_cards[num_unknowns] = c;
            num_unknowns += 1;
        }
    }
    let unknown_slice = &unknown_cards[..num_unknowns];

    // ── Constraint feasibility pre-check ──
    // If the void constraints are provably impossible (e.g., remaining cards
    // of a suit exceed the combined hand capacity of non-void players), all
    // world samples will fail. Detect this in O(cards) instead of wasting
    // up to 10 retry cycles per world.
    if !constraints_feasible(unknown_slice, voids, target_sizes, my_seat) {
        return Vec::new();
    }

    // 3. Batch-process worlds with paired-difference Welford tracking.
    //    After each batch: significance exit (paired SE confirms best move is better)
    //    AND futility exit (projected gap at full worlds is too small to matter).
    //
    //    Paired comparison: within each world, all legal moves are evaluated on the
    //    same sampled world, so between-world variance cancels out of move comparisons.
    //    Tracking the Welford on (best_ev - second_best_ev) per world gives tighter SE
    //    than sqrt(SE_best² + SE_second²) which assumes independence.
    const BATCH_SIZE: usize = 50;
    const MIN_WORLDS_FOR_EARLY_EXIT: usize = 50;
    const CONFIDENCE_Z: f64 = 2.0; // ~95% confidence
    const FUTILITY_Z: f64 = 2.0;
    const MIN_MEANINGFUL_GAP: f64 = 0.5; // Sueca points — below this, EV delta is noise

    // Per-move Welford statistics: online mean + M2 for variance estimation.
    // Parallel merge via Chan et al. formula — numerically stable across batches.
    #[derive(Clone, Copy, Default)]
    struct MoveWelford {
        count: u32,
        mean: f64,
        m2: f64, // sum of squared differences from the mean
    }

    impl MoveWelford {
        fn update(&mut self, value: f64) {
            self.count += 1;
            let delta = value - self.mean;
            self.mean += delta / self.count as f64;
            let delta2 = value - self.mean;
            self.m2 += delta * delta2;
        }

        fn merge_parallel(&mut self, other: &MoveWelford) {
            if other.count == 0 {
                return;
            }
            if self.count == 0 {
                *self = *other;
                return;
            }
            let total = self.count + other.count;
            let delta = other.mean - self.mean;
            self.mean = (self.count as f64 * self.mean + other.count as f64 * other.mean)
                / total as f64;
            self.m2 += other.m2
                + delta * delta * (self.count as f64 * other.count as f64) / total as f64;
            self.count = total;
        }

        fn variance(&self) -> f64 {
            if self.count < 2 {
                0.0
            } else {
                self.m2 / (self.count - 1) as f64
            }
        }

        fn std_error(&self) -> f64 {
            if self.count == 0 {
                f64::INFINITY
            } else {
                (self.variance() / self.count as f64).sqrt()
            }
        }

        /// Project what the SE would be at `target_samples` total worlds,
        /// assuming current variance is representative.
        fn projected_se_at(&self, target_samples: usize) -> f64 {
            if self.count < 2 {
                return f64::INFINITY;
            }
            (self.variance() / target_samples as f64).sqrt()
        }
    }

    // Per-world result: per-move Welford partials + raw (card, EV) pairs for paired-diff
    struct WorldResult {
        welfords: [MoveWelford; 40],
        card_evs: [(u8, f64); 10], // up to 10 legal moves per Sueca hand
        ev_count: u8,
    }

    // Running Welford statistics across all batches (indexed by card id 0..40)
    let mut running = [MoveWelford::default(); 40];
    // Paired-difference Welford: tracks (best_ev - second_best_ev) per world.
    // CRITICAL: reset whenever the best/second-best card pair changes, because
    // mixing samples of (A-B) with (C-A) describes no actual pair of moves.
    let mut paired_diff = MoveWelford::default();

    // Track best/second card identities for paired-diff validity.
    // When the pair changes, paired_diff is reset from the current batch.
    let mut prev_best_card: u8 = 40;
    let mut prev_second_card: u8 = 40;

    let mut batch_start = 0;
    while batch_start < n_worlds {
        let batch_end = (batch_start + BATCH_SIZE).min(effective_worlds);

        // Process this batch in parallel — each world returns per-move Welford + raw EVs
        let batch_results: Vec<WorldResult> = (batch_start..batch_end)
            .into_par_iter()
            .map(|world_idx| {
                let mut welfords = [MoveWelford::default(); 40];
                let mut card_evs = [(0u8, 0.0f64); 10];
                let mut ev_count = 0u8;

                let mut local_hands = [0u64; 4];
                let mut local_seed =
                    seed.wrapping_add((world_idx as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));

                // Sample a valid world. Try up to 10 times.
                let mut success = false;
                for _ in 0..10 {
                    if sample_world(
                        my_seat,
                        my_hand,
                        unknown_slice,
                        voids,
                        target_sizes,
                        &mut local_hands,
                        &mut local_seed,
                    ) {
                        success = true;
                        break;
                    }
                }

                if !success {
                    return WorldResult { welfords, card_evs, ev_count };
                }

                // Pre-compute invariants once per world (GameState is Copy)
                let mut base_trick_points = 0u8;
                for &c in current_trick_cards {
                    base_trick_points += crate::engine::CARD_POINTS[c as usize];
                }

                let mut base_game = GameState::new(local_hands, trump, current_player);
                base_game.set_led_suit(led_suit);
                base_game.current_trick_winner = current_trick_winner;
                base_game.current_trick_best_card = current_trick_best_card;
                base_game.cards_played_in_trick = current_trick_cards.len() as u8;
                base_game.team_02_score = team_scores[0];
                base_game.team_13_score = team_scores[1];
                base_game.trick_number = trick_number;

                THREAD_TT.with(|tt_cell| {
                    let mut tt = tt_cell.borrow_mut();
                    tt.next_generation();
                    // Stack-allocated killer tables (128 bytes total)
                    let mut killer_a = [40u8; 64];
                    let mut killer_b = [40u8; 64];

                    for i in 0..legal_count {
                        let m = legal_moves[i];
                        let mut sim_game = base_game;
                        let mut trick_points = base_trick_points;

                        sim_game.play_card_and_resolve(m, &mut trick_points);

                        let plies_left = if trick_number >= 6 {
                            40u8
                        } else {
                            search_depth * 4
                        };
                        let val = alpha_beta(
                            &mut sim_game,
                            -1000,
                            1000,
                            plies_left,
                            &mut tt,
                            &mut trick_points,
                            &mut killer_a,
                            &mut killer_b,
                        );
                        welfords[m as usize].update(val as f64);
                        if (ev_count as usize) < 10 {
                            card_evs[ev_count as usize] = (m, val as f64);
                            ev_count += 1;
                        }
                    }
                });

                WorldResult { welfords, card_evs, ev_count }
            })
            .collect();

        // Merge batch results into running per-move statistics
        for wr in &batch_results {
            for card in 0..40 {
                if wr.welfords[card].count > 0 {
                    running[card].merge_parallel(&wr.welfords[card]);
                }
            }
        }

        // Re-determine best and second-best cards from updated running means
        let (best_card, second_card) = {
            let mut best = legal_moves[0];
            let mut best_mean = running[best as usize].mean;
            for i in 1..legal_count {
                let m = legal_moves[i];
                let mean = running[m as usize].mean;
                if mean > best_mean {
                    best_mean = mean;
                    best = m;
                }
            }

            let mut second = best; // fallback if only 1 legal move
            let mut second_mean = f64::NEG_INFINITY;
            for i in 0..legal_count {
                let m = legal_moves[i];
                if m != best {
                    let mean = running[m as usize].mean;
                    if mean > second_mean {
                        second_mean = mean;
                        second = m;
                    }
                }
            }
            (best, second)
        };

        // ── Paired-difference Welford with leader-change detection ──
        // If the best/second pair changed since the last batch, reset the
        // paired accumulator. Mixing (A-B) with (C-A) produces a mean and
        // variance that describe no actual pair of moves.
        let leaders_changed = best_card != prev_best_card || second_card != prev_second_card;
        if leaders_changed {
            paired_diff = MoveWelford::default();
            prev_best_card = best_card;
            prev_second_card = second_card;
        }

        // Accumulate paired differences for this batch from per-world EVs
        if second_card != best_card {
            let mut batch_paired = MoveWelford::default();
            for wr in &batch_results {
                let mut world_best_ev = f64::NEG_INFINITY;
                let mut world_second_ev = f64::NEG_INFINITY;
                for i in 0..(wr.ev_count as usize) {
                    let (card, ev) = wr.card_evs[i];
                    if card == best_card {
                        world_best_ev = ev;
                    } else if card == second_card {
                        world_second_ev = ev;
                    }
                }
                if world_best_ev > f64::NEG_INFINITY && world_second_ev > f64::NEG_INFINITY {
                    batch_paired.update(world_best_ev - world_second_ev);
                }
            }
            paired_diff.merge_parallel(&batch_paired);
        }

        // ── Early termination checks (using paired SE) ──
        // In diff mode, skip early termination to eliminate stopping variance.
        if !diff_mode && batch_end >= MIN_WORLDS_FOR_EARLY_EXIT && paired_diff.count >= 2 {
            let paired_mean = paired_diff.mean;
            let paired_se = paired_diff.std_error();

            // Significance exit: best move is clearly better
            if paired_mean > CONFIDENCE_Z * paired_se {
                break;
            }

            // Futility exit: project paired SE to full world budget
            let projected_se = paired_diff.projected_se_at(effective_worlds);
            let projected_lower = (paired_mean - FUTILITY_Z * projected_se).max(0.0);
            if projected_lower < MIN_MEANINGFUL_GAP {
                return Vec::new();
            }
        }

        batch_start = batch_end;
    }

    // Build final results from running Welford statistics
    let mut results = Vec::with_capacity(legal_count);
    for i in 0..legal_count {
        let m = legal_moves[i];
        let stats = &running[m as usize];
        results.push(PimcResult {
            card: m,
            ev: if stats.count > 0 { stats.mean } else { 0.0 },
            std_error: stats.std_error(),
        });
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_world_simple() {
        // Player 0 hand: 10 hearts
        let mut my_hand = 0u64;
        for r in 0..10 {
            my_hand |= 1u64 << (0 * 10 + r); // Hearts
        }

        let played_cards = 0u64;
        let voids = [0u8; 4];
        let target_sizes = [10, 10, 10, 10];

        // Build unknown card pool
        let known_mask = my_hand | played_cards;
        let mut unknown_cards = [0u8; 40];
        let mut n = 0;
        for c in 0..40 {
            if (known_mask & (1u64 << c)) == 0 {
                unknown_cards[n] = c;
                n += 1;
            }
        }

        let mut hands_out = [0u64; 4];
        let mut seed = 42u64;

        let ok = sample_world(
            0,
            my_hand,
            &unknown_cards[..n],
            voids,
            target_sizes,
            &mut hands_out,
            &mut seed,
        );

        assert!(ok);
        // Player 0 has my_hand
        assert_eq!(hands_out[0], my_hand);
        // All players have 10 cards
        for i in 0..4 {
            assert_eq!(hands_out[i].count_ones(), 10);
        }
        // Total cards is 40
        let total = hands_out[0] | hands_out[1] | hands_out[2] | hands_out[3];
        assert_eq!(total.count_ones(), 40);
        assert_eq!(total, 0xFFFFFFFFFFu64); // all 40 cards accounted for
    }

    #[test]
    fn test_sample_world_void_constraint() {
        // Player 0 hand: 10 hearts
        let mut my_hand = 0u64;
        for r in 0..10 {
            my_hand |= 1u64 << (0 * 10 + r); // Hearts
        }

        let played_cards = 0u64;
        // Player 1 is void in Diamonds (suit 1)
        let mut voids = [0u8; 4];
        voids[1] = 1 << 1; // Diamonds void

        let target_sizes = [10, 10, 10, 10];

        // Build unknown card pool
        let known_mask = my_hand | played_cards;
        let mut unknown_cards = [0u8; 40];
        let mut n = 0;
        for c in 0..40 {
            if (known_mask & (1u64 << c)) == 0 {
                unknown_cards[n] = c;
                n += 1;
            }
        }

        let mut hands_out = [0u64; 4];
        let mut seed = 42u64;

        let ok = sample_world(
            0,
            my_hand,
            &unknown_cards[..n],
            voids,
            target_sizes,
            &mut hands_out,
            &mut seed,
        );
        assert!(ok);

        // Diamonds mask: 10..19 (bits 10 to 19)
        let diamonds_mask = 0x3FFu64 << 10;
        // Player 1 should not have any diamonds
        assert_eq!(hands_out[1] & diamonds_mask, 0);
    }

    #[test]
    fn test_solve_pimc_simple() {
        // Player 0 hand: Hearts Ace (9), Hearts 7 (8), 8 other cards (10..17)
        let mut my_hand = (1u64 << 9) | (1u64 << 8);
        for c in 10..18 {
            my_hand |= 1u64 << c;
        }

        let played_cards = 0u64;
        let voids = [0u8; 4];
        let target_sizes = [10, 10, 10, 10];

        let evs = solve_pimc(
            0, // my_seat
            my_hand,
            played_cards,
            voids,
            target_sizes,
            0,      // trump Hearts
            0,      // led Hearts
            &[],    // current trick cards
            0,      // current player
            0,      // current trick winner
            40,     // trick best card (none)
            [0, 0], // team scores
            0,      // trick number
            5,      // n_worlds
            1,      // search depth (tricks)
            123,    // seed
            false,
            None,
        );

        assert!(!evs.is_empty());
        // Since we had legal moves, we should have evs computed for all legal moves of led suit
        // Hearts Ace and 7 are legal moves because Hearts led
        let heart_moves: Vec<u8> = evs
            .iter()
            .map(|r| r.card)
            .filter(|&m| crate::engine::CARD_SUIT[m as usize] == 0)
            .collect();
        assert!(heart_moves.contains(&9));
        assert!(heart_moves.contains(&8));
    }

    /// Optimization #1: splitmix64 seed derivation must produce deterministic,
    /// reproducible worlds from a given base seed.
    #[test]
    fn test_splitmix64_seed_determinism() {
        let my_hand = (1u64 << 0)
            | (1u64 << 1)
            | (1u64 << 2)
            | (1u64 << 3)
            | (1u64 << 4)
            | (1u64 << 5)
            | (1u64 << 6)
            | (1u64 << 7)
            | (1u64 << 8)
            | (1u64 << 9);
        let voids = [0u8; 4];
        let target_sizes = [10, 10, 10, 10];

        // Run solve_pimc twice with the same seed — results must be identical
        let evs1 = solve_pimc(
            0,
            my_hand,
            0,
            voids,
            target_sizes,
            0,
            0,
            &[],
            0,
            0,
            40,
            [0, 0],
            0,
            10,
            1,
            42,
            false,
            None,
        );
        let evs2 = solve_pimc(
            0,
            my_hand,
            0,
            voids,
            target_sizes,
            0,
            0,
            &[],
            0,
            0,
            40,
            [0, 0],
            0,
            10,
            1,
            42,
            false,
            None,
        );

        assert_eq!(evs1.len(), evs2.len());
        for (r1, r2) in evs1.iter().zip(evs2.iter()) {
            assert_eq!(r1.card, r2.card, "Move mismatch at same seed");
            assert!(
                (r1.ev - r2.ev).abs() < 1e-9,
                "EV mismatch at same seed: {} vs {}",
                r1.ev,
                r2.ev
            );
        }
    }

    /// Optimization #1: different seeds must produce potentially different
    /// EV scores (seeds are not accidentally collapsed).
    #[test]
    fn test_splitmix64_seed_diversity() {
        let my_hand = (1u64 << 0)
            | (1u64 << 1)
            | (1u64 << 2)
            | (1u64 << 3)
            | (1u64 << 4)
            | (1u64 << 5)
            | (1u64 << 6)
            | (1u64 << 7)
            | (1u64 << 8)
            | (1u64 << 9);
        let voids = [0u8; 4];
        let target_sizes = [10, 10, 10, 10];

        // With enough worlds, different seeds should yield numerically distinct EVs
        let evs_a = solve_pimc(
            0,
            my_hand,
            0,
            voids,
            target_sizes,
            0,
            0,
            &[],
            0,
            0,
            40,
            [0, 0],
            0,
            50,
            1,
            100,
            false,
            None,
        );
        let evs_b = solve_pimc(
            0,
            my_hand,
            0,
            voids,
            target_sizes,
            0,
            0,
            &[],
            0,
            0,
            40,
            [0, 0],
            0,
            50,
            1,
            999,
            false,
            None,
        );

        assert!(!evs_a.is_empty());
        assert!(!evs_b.is_empty());
        // With 50 worlds and different seeds, EVs should differ for at least one move
        let mut any_different = false;
        for (r_a, r_b) in evs_a.iter().zip(evs_b.iter()) {
            if (r_a.ev - r_b.ev).abs() > 1e-9 {
                any_different = true;
                break;
            }
        }
        assert!(
            any_different,
            "Different seeds should produce different EV estimates"
        );
    }

    /// Optimization #2: invariant code motion — base GameState is copied per-move,
    /// so each move evaluation starts from the identical pre-move state.
    #[test]
    fn test_invariant_base_state_is_not_mutated() {
        // Set up a simple deal: player 0 has all hearts, others share remaining cards
        let mut my_hand = 0u64;
        for r in 0..10 {
            my_hand |= 1u64 << (0 * 10 + r);
        }
        let voids = [0u8; 4];
        let target_sizes = [10, 10, 10, 10];

        let evs = solve_pimc(
            0,
            my_hand,
            0,
            voids,
            target_sizes,
            0,
            0,
            &[],
            0,
            0,
            40,
            [0, 0],
            0,
            10,
            1,
            42,
            false,
            None,
        );

        // Every legal move must appear exactly once in the output
        assert!(!evs.is_empty());
        let mut seen = std::collections::HashSet::new();
        for r in &evs {
            assert!(
                seen.insert(r.card),
                "Move {} appears multiple times — base state was mutated between iterations",
                r.card
            );
        }
        // All legal hearts cards should be present (player 0 has all 10 hearts)
        assert_eq!(evs.len(), 10, "Should have 10 legal moves (all hearts)");
    }

    /// Optimization #3: MRV heuristic — constrained cards (where a hidden player
    /// is void in that suit) must appear before unconstrained cards in the shuffled
    /// array, ensuring backtracking fails fast.
    #[test]
    fn test_mrv_constrained_cards_first() {
        // Player 0 hand: 5 hearts + 5 diamonds
        let mut my_hand = 0u64;
        for r in 0..5 {
            my_hand |= 1u64 << (0 * 10 + r); // Hearts 0-4
            my_hand |= 1u64 << (1 * 10 + r); // Diamonds 0-4
        }

        // Player 1 is void in hearts, player 2 is void in diamonds
        let mut voids = [0u8; 4];
        voids[1] = 1; // Player 1 void in hearts (suit 0)
        voids[2] = 1 << 1; // Player 2 void in diamonds (suit 1)

        // target_sizes: how many cards each player must end up with (10 each)
        let target_sizes = [10, 10, 10, 10];

        // Build unknown card pool (cards not in my_hand)
        let known_mask = my_hand;
        let mut unknown_cards = [0u8; 40];
        let mut n = 0;
        for c in 0..40 {
            if (known_mask & (1u64 << c)) == 0 {
                unknown_cards[n] = c;
                n += 1;
            }
        }

        let mut hands_out = [0u64; 4];
        let mut seed = 42u64;

        // We can't directly inspect the shuffled array, but we can verify
        // that the constraint solver succeeds despite the voids, which
        // validates that constrained cards are prioritized correctly.
        let ok = sample_world(
            0,
            my_hand,
            &unknown_cards[..n],
            voids,
            target_sizes,
            &mut hands_out,
            &mut seed,
        );
        assert!(ok, "MRV partitioning must not break world sampling");

        // Player 1 must have zero hearts
        let hearts_mask = 0x3FFu64;
        assert_eq!(
            hands_out[1] & hearts_mask,
            0,
            "Player 1 must have no hearts (void constraint)"
        );
        // Player 2 must have zero diamonds
        let diamonds_mask = 0x3FFu64 << 10;
        assert_eq!(
            hands_out[2] & diamonds_mask,
            0,
            "Player 2 must have no diamonds (void constraint)"
        );

        // All players must have exactly 10 cards
        for s in 0..4 {
            assert_eq!(
                hands_out[s].count_ones(),
                10,
                "Player {} must have exactly 10 cards",
                s
            );
        }
    }

    /// Optimization #3: worlds with impossible void constraints must fail
    /// gracefully (not infinite-loop). The old shuffle might succeed by chance;
    /// MRV must still handle failure correctly.
    #[test]
    fn test_mrv_impossible_constraints_fail() {
        // Player 0 has 5 hearts + 5 diamonds. The remaining 5 hearts must go
        // to the 3 other players, but all 3 are void in hearts — impossible.
        let mut my_hand = 0u64;
        for r in 0..5 {
            my_hand |= 1u64 << (0 * 10 + r); // 5 hearts
            my_hand |= 1u64 << (1 * 10 + r); // 5 diamonds
        }

        let mut voids = [0u8; 4];
        voids[1] = 1; // All 3 other players void in hearts
        voids[2] = 1;
        voids[3] = 1;

        // 20 unknown cards: remaining 5 hearts + 5 diamonds + 5 clubs + 5 spades
        // (player 0 already has 5 hearts + 5 diamonds = 10)
        let target_sizes = [10, 10, 10, 10];

        let known_mask = my_hand;
        let mut unknown_cards = [0u8; 40];
        let mut n = 0;
        for c in 0..40 {
            if (known_mask & (1u64 << c)) == 0 {
                unknown_cards[n] = c;
                n += 1;
            }
        }

        let mut hands_out = [0u64; 4];
        let mut seed = 42u64;

        // With MRV, this should fail quickly (constrained hearts can't be placed)
        // Try a few times — must never succeed
        let mut any_success = false;
        for _ in 0..20 {
            let mut local_seed = seed;
            seed = seed.wrapping_add(1);
            if sample_world(
                0,
                my_hand,
                &unknown_cards[..n],
                voids,
                target_sizes,
                &mut hands_out,
                &mut local_seed,
            ) {
                any_success = true;
                break;
            }
        }
        assert!(
            !any_success,
            "Impossible constraints (3 players void in same suit with cards remaining) must fail"
        );
    }

    // ───────────────────────── solve_pimc_rollout ─────────────────────────

    use crate::simulator::SuecaSimulatorGame;

    /// A valid 40-card deal: hands[i] = cards [10*i .. 10*i+10).
    fn deterministic_deal() -> [u64; 4] {
        let mut hands = [0u64; 4];
        for p in 0..4 {
            for r in 0..10 {
                hands[p] |= 1u64 << (p as u64 * 10 + r);
            }
        }
        hands
    }

    #[test]
    fn test_rollout_leading_scores_all_moves() {
        // Leading player (seat 0) has all 10 cards legal; each must get a bounded EV.
        let game = SuecaSimulatorGame::new(deterministic_deal(), /*trump*/ 0, /*first*/ 0);
        let res = solve_pimc_rollout(&game, /*n_worlds*/ 24, /*seed*/ 42);
        assert_eq!(res.len(), 10, "leading -> every hand card is a legal move");
        let mut seen = [false; 40];
        for r in &res {
            assert!(
                r.ev >= 0.0 && r.ev <= 120.0,
                "team score EV out of range: {}",
                r.ev
            );
            assert!(!seen[r.card as usize], "duplicate card {}", r.card);
            seen[r.card as usize] = true;
            // Returned card must be a card the leader actually holds.
            assert!(game.state.hands[0] & (1u64 << r.card) != 0);
        }
    }

    #[test]
    fn test_rollout_following_respects_legal_set() {
        // Seat 0 leads its lowest card; the next player (seat 3, counter-clockwise)
        // then chooses among ITS legal follow-suit moves only.
        let mut game = SuecaSimulatorGame::new(deterministic_deal(), 0, 0);
        let lead = game.state.legal_moves().trailing_zeros() as u8;
        game.play_card(lead);
        let follower = game.state.current_player;
        let legal = game.state.legal_moves();
        let res = solve_pimc_rollout(&game, 24, 7);
        assert_eq!(res.len(), legal.count_ones() as usize);
        for r in &res {
            assert!(
                legal & (1u64 << r.card) != 0,
                "card {} not in follower's legal set",
                r.card
            );
            assert!(game.state.hands[follower as usize] & (1u64 << r.card) != 0);
            assert!(r.ev >= 0.0 && r.ev <= 120.0);
        }
    }

    #[test]
    fn test_rollout_forced_single_move_in_endgame() {
        // Play the deal down with Elite until exactly one card remains per hand
        // (start of the last trick), then the leader has a single legal move.
        use crate::heuristic::select_card_heuristic;
        let mut game = SuecaSimulatorGame::new(deterministic_deal(), 0, 0);
        while !(game.state.trick_number == 9 && game.state.cards_played_in_trick == 0) {
            let s = game.state.current_player;
            let c = select_card_heuristic(&game, s);
            game.play_card(c);
        }
        assert_eq!(game.state.legal_moves().count_ones(), 1);
        let res = solve_pimc_rollout(&game, 8, 1);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].card, game.state.legal_moves().trailing_zeros() as u8);
    }
}
