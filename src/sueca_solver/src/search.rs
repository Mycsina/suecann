use crate::engine::GameState;

// Compile-time LCG for Zobrist table generation
const fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

const fn generate_zobrist_table() -> [[u64; 40]; 4] {
    let mut table = [[0u64; 40]; 4];
    let mut rng = 0x123456789abcdef0u64;
    let mut p = 0;
    while p < 4 {
        let mut c = 0;
        while c < 40 {
            rng = lcg_next(rng);
            table[p][c] = rng;
            c += 1;
        }
        p += 1;
    }
    table
}

const fn generate_player_zobrist() -> [u64; 4] {
    // Continue from where the table generation left off (4*40 = 160 iterations)
    let mut rng = 0x123456789abcdef0u64;
    let mut i = 0;
    while i < 160 {
        rng = lcg_next(rng);
        i += 1;
    }
    let mut players = [0u64; 4];
    let mut p = 0;
    while p < 4 {
        rng = lcg_next(rng);
        players[p] = rng;
        p += 1;
    }
    players
}

const fn generate_led_zobrist() -> [u64; 5] {
    let mut rng = 0x123456789abcdef0u64;
    let mut i = 0;
    while i < 164 {
        rng = lcg_next(rng);
        i += 1;
    }
    let mut led = [0u64; 5];
    let mut l = 0;
    while l < 5 {
        rng = lcg_next(rng);
        led[l] = rng;
        l += 1;
    }
    led
}

pub(crate) static ZOBRIST_TABLE: [[u64; 40]; 4] = generate_zobrist_table();
pub(crate) static PLAYER_ZOBRIST: [u64; 4] = generate_player_zobrist();
pub(crate) static LED_SUIT_ZOBRIST: [u64; 5] = generate_led_zobrist();

/// O(1) hash read — the Zobrist hash is maintained incrementally in
/// play_card_and_resolve. This was previously O(cards_remaining) and
/// consumed ~14% of total CPU in dataset generation.
#[inline(always)]
pub fn get_hash(state: &GameState) -> u64 {
    debug_assert!({
        // Full recomputation check (stripped in release builds)
        let mut hash = 0u64;
        for p in 0..4 {
            let mut hand = state.hands[p];
            while hand != 0 {
                let card = hand.trailing_zeros() as usize;
                hash ^= ZOBRIST_TABLE[p][card];
                hand &= hand - 1;
            }
        }
        hash ^= PLAYER_ZOBRIST[state.current_player as usize];
        hash ^= LED_SUIT_ZOBRIST[state.led_suit as usize];
        hash == state.hash
    }, "Incremental Zobrist hash mismatch");
    state.hash
}

#[derive(Clone, Copy, Default)]
pub struct TTEntry {
    pub key: u64,
    pub value: i16,
    pub flag: u8, // 0 = Exact, 1 = LowerBound (Beta-cut), 2 = UpperBound (Alpha-cut)
    pub depth: u8,
    pub generation: u16,
    pub best_move: u8,
}

pub struct TranspositionTable {
    pub table: Vec<TTEntry>,
    pub mask: usize,
    pub generation: u16,
}

impl TranspositionTable {
    pub fn new(size_power_of_two: usize) -> Self {
        let size = 1 << size_power_of_two;
        Self {
            table: vec![TTEntry::default(); size],
            mask: size - 1,
            generation: 0,
        }
    }

    pub fn next_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    #[inline(always)]
    pub fn lookup(&self, key: u64, depth: u8) -> Option<(i16, u8, u8)> {
        let idx = (key as usize) & self.mask;
        let entry = &self.table[idx];
        if entry.key == key && entry.generation == self.generation && entry.depth >= depth {
            Some((entry.value, entry.flag, entry.best_move))
        } else {
            None
        }
    }

    /// Return the cached best_move for this position, regardless of depth.
    /// Used for move ordering even when the full TT entry is too shallow for a cutoff.
    #[inline(always)]
    pub fn lookup_best_move(&self, key: u64) -> Option<u8> {
        let idx = (key as usize) & self.mask;
        let entry = &self.table[idx];
        if entry.key == key && entry.generation == self.generation && entry.best_move < 40 {
            Some(entry.best_move)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn store(&mut self, key: u64, value: i16, flag: u8, depth: u8, best_move: u8) {
        let idx = (key as usize) & self.mask;
        // Simple replace scheme
        self.table[idx] = TTEntry {
            key,
            value,
            flag,
            depth,
            generation: self.generation,
            best_move,
        };
    }
}

/// Search for optimal play using alpha-beta minimax.
/// Returns score for Team 0-2 (0..120).
pub fn alpha_beta(
    state: &mut GameState,
    mut alpha: i16,
    mut beta: i16,
    plies_left: u8,
    tt: &mut TranspositionTable,
    trick_points: &mut u8,
) -> i16 {
    if state.is_terminal() {
        return state.team_02_score as i16;
    }

    if plies_left == 0 {
        // Depth limit reached. Evaluate using static/heuristic score:
        // A simple baseline is current score + potential remaining points.
        // For a true PIMC bot, we roll out using HeuristicBot, but for the search tree,
        // we can return the current score.
        // Let's return the current score for Team 0-2 as the static evaluation value.
        return state.team_02_score as i16;
    }

    let hash = get_hash(state);
    if let Some((val, flag, _best)) = tt.lookup(hash, plies_left) {
        if flag == 0 {
            return val;
        } else if flag == 1 {
            alpha = alpha.max(val);
        } else if flag == 2 {
            beta = beta.min(val);
        }
        if alpha >= beta {
            return val;
        }
    }

    let curr_player = state.current_player;
    let is_maximizing = curr_player == 0 || curr_player == 2;
    let legal = state.legal_moves();

    let mut best_val = if is_maximizing { -1000 } else { 1000 };
    let mut best_move_card = 40;

    // ── TT best move: try it first with make/unmake ──
    let tt_best = tt.lookup_best_move(hash).filter(|&m| (legal & (1u64 << m)) != 0);
    if let Some(best) = tt_best {
        let snap = state.save_snapshot();
        let pre_points = *trick_points;
        state.play_card_and_resolve(best, trick_points);

        let val = alpha_beta(state, alpha, beta, plies_left - 1, tt, trick_points);

        // Unmake
        state.restore_snapshot(&snap);
        *trick_points = pre_points;

        if is_maximizing {
            if val > best_val {
                best_val = val;
                best_move_card = best;
            }
            alpha = alpha.max(best_val);
        } else {
            if val < best_val {
                best_val = val;
                best_move_card = best;
            }
            beta = beta.min(best_val);
        }
    }

    // ── Remaining moves: MSB-first bitboard iteration (no array, no sort) ──
    // Card encoding is suit*10+rank (rank 9=Ace, 0=2). Iterating MSB-first
    // naturally yields rank-descending within a suit — Ace (rank 9) before
    // 7 (rank 8) before K (rank 7), etc. For follow-suit (all cards in one
    // suit), this is perfect. For non-follow-suit, suit 3 (spades) iterates
    // before suit 0 (hearts), which is arbitrary but cheap.
    let remaining = if tt_best.is_some() {
        legal & !(1u64 << tt_best.unwrap())
    } else {
        legal
    };

    // Process remaining moves directly from the bitboard — MSB-first for
    // rank-descending order. This eliminates the move array (cache pressure),
    // the insertion sort (6.9% of CPU), and the GameState copy (memcpy)
    // by using make/unmake snapshots.
    if alpha < beta {
        // (alpha >= beta means we already pruned after TT best move)
        let mut bitboard = remaining;
        while bitboard != 0 {
            // MSB-first: highest card index = highest rank within a suit
            let m = (63 - bitboard.leading_zeros()) as u8;

            // Make move (save/play, no GameState copy)
            let snap = state.save_snapshot();
            let pre_points = *trick_points;
            state.play_card_and_resolve(m, trick_points);

            let val = alpha_beta(state, alpha, beta, plies_left - 1, tt, trick_points);

            // Unmake move (restore pre-move state)
            state.restore_snapshot(&snap);
            *trick_points = pre_points;

            if is_maximizing {
                if val > best_val {
                    best_val = val;
                    best_move_card = m;
                }
                alpha = alpha.max(best_val);
            } else {
                if val < best_val {
                    best_val = val;
                    best_move_card = m;
                }
                beta = beta.min(best_val);
            }

            if alpha >= beta {
                break; // Prune
            }

            bitboard &= !(1u64 << m); // clear the bit we just processed
        }
    }

    // Determine entry flag
    let flag = if best_val <= alpha {
        2 // UpperBound
    } else if best_val >= beta {
        1 // LowerBound
    } else {
        0 // Exact
    };

    tt.store(hash, best_val, flag, plies_left, best_move_card);

    best_val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_simple() {
        // Construct a simple game state
        // Player 0 hand: Ace of Hearts (9), Seven of Hearts (8)
        // Player 3 hand: King of Hearts (7), Jack of Hearts (6)
        // Player 2 hand: Queen of Hearts (5), Six of Hearts (4)
        // Player 1 hand: Five of Hearts (3), Four of Hearts (2)

        let p0_hand = (1u64 << 9) | (1u64 << 8);
        let p3_hand = (1u64 << 7) | (1u64 << 6);
        let p2_hand = (1u64 << 5) | (1u64 << 4);
        let p1_hand = (1u64 << 3) | (1u64 << 2);

        let hands = [p0_hand, p1_hand, p2_hand, p3_hand];
        let mut state = GameState::new(hands, 0, 0); // Hearts led, trump Hearts

        let mut tt = TranspositionTable::new(10);
        let mut trick_points = 0;

        let val = alpha_beta(&mut state, -1000, 1000, 8, &mut tt, &mut trick_points);

        // Since Team 0-2 holds Ace (11), Seven (10), Queen (2), Six (0),
        // they should win all tricks and score 11 + 10 + 2 + 0 + 4 (King) + 3 (Jack) = 30 points.
        assert_eq!(val, 30);
    }
}
