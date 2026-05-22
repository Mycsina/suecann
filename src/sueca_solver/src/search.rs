use crate::engine::GameState;
use std::sync::OnceLock;

// Zobrist hash constants
static ZOBRIST_INIT: OnceLock<()> = OnceLock::new();
static ZOBRIST_TABLE: OnceLock<[[u64; 40]; 4]> = OnceLock::new();
static PLAYER_ZOBRIST: OnceLock<[u64; 4]> = OnceLock::new();
static LED_SUIT_ZOBRIST: OnceLock<[u64; 5]> = OnceLock::new(); // 0..4

use crate::rng::LcgRng;

pub fn init_zobrist() {
    ZOBRIST_INIT.get_or_init(|| {
        let mut rng = LcgRng::new(0x123456789abcdef0);

        let mut table = [[0u64; 40]; 4];
        for p in 0..4 {
            for c in 0..40 {
                table[p][c] = rng.next_u64();
            }
        }
        let _ = ZOBRIST_TABLE.set(table);

        let mut players = [0u64; 4];
        for p in 0..4 {
            players[p] = rng.next_u64();
        }
        let _ = PLAYER_ZOBRIST.set(players);

        let mut led = [0u64; 5];
        for l in 0..5 {
            led[l] = rng.next_u64();
        }
        let _ = LED_SUIT_ZOBRIST.set(led);
    });
}

#[inline(always)]
pub fn get_hash(state: &GameState) -> u64 {
    init_zobrist();
    let table = ZOBRIST_TABLE.get().unwrap();
    let players = PLAYER_ZOBRIST.get().unwrap();
    let led = LED_SUIT_ZOBRIST.get().unwrap();

    let mut hash = 0u64;
    for p in 0..4 {
        let mut hand = state.hands[p];
        while hand != 0 {
            let card = hand.trailing_zeros() as usize;
            hash ^= table[p][card];
            hand &= hand - 1;
        }
    }
    hash ^= players[state.current_player as usize];
    hash ^= led[state.led_suit as usize];
    hash
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
        let val = val as i16;
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

    // Move ordering: try to order legal moves to trigger early cutoffs.
    // Trivial ordering: high cards / trumps first.
    let mut moves = [0u8; 10];
    let mut move_count = 0;
    let mut temp = legal;
    while temp != 0 {
        moves[move_count] = temp.trailing_zeros() as u8;
        move_count += 1;
        temp &= temp - 1;
    }

    // Sort moves by simple heuristic: power rank (card % 10) desc
    moves[0..move_count].sort_by_key(|&c| std::cmp::Reverse(c % 10));

    let mut best_val = if is_maximizing { -1000 } else { 1000 };
    let mut best_move = 40;

    for i in 0..move_count {
        let m = moves[i];
        let mut next_state = *state;
        let mut next_points = *trick_points;
        next_state.play_card_and_resolve(m, &mut next_points);

        let val = alpha_beta(
            &mut next_state,
            alpha,
            beta,
            plies_left - 1,
            tt,
            &mut next_points,
        );

        if is_maximizing {
            if val > best_val {
                best_val = val;
                best_move = m;
            }
            alpha = alpha.max(best_val);
        } else {
            if val < best_val {
                best_val = val;
                best_move = m;
            }
            beta = beta.min(best_val);
        }

        if alpha >= beta {
            break; // Prune
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

    tt.store(hash, best_val as i16, flag, plies_left, best_move);

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
