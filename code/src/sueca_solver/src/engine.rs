/// Bitboard-based game engine for Sueca.
/// remaps cards to 0..39: index = suit * 10 + rank.
/// Ranks (0..9) are ordered by trick-taking power: 2=0, 3=1, 4=2, 5=3, 6=4, Q=5, J=6, K=7, 7=8, A=9.
/// Suits: Hearts=0, Diamonds=1, Clubs=2, Spades=3.
use std::fmt;

use crate::search::{LED_SUIT_ZOBRIST, PLAYER_ZOBRIST, ZOBRIST_TABLE};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Suit {
    Hearts = 0,
    Diamonds = 1,
    Clubs = 2,
    Spades = 3,
}

impl Suit {
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => Suit::Hearts,
            1 => Suit::Diamonds,
            2 => Suit::Clubs,
            3 => Suit::Spades,
            _ => panic!("Invalid suit value: {}", val),
        }
    }
}

pub const CARD_POINTS: [u8; 40] = [
    // Hearts (0..9)
    0, 0, 0, 0, 0, 2, 3, 4, 10, 11, // Diamonds (10..19)
    0, 0, 0, 0, 0, 2, 3, 4, 10, 11, // Clubs (20..29)
    0, 0, 0, 0, 0, 2, 3, 4, 10, 11, // Spades (30..39)
    0, 0, 0, 0, 0, 2, 3, 4, 10, 11,
];

// Pre-computed suit and rank lookups to avoid integer division at runtime.
pub const CARD_SUIT: [u8; 40] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3,
];

pub const CARD_RANK: [u8; 40] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1,
    2, 3, 4, 5, 6, 7, 8, 9,
];

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct GameState {
    pub hands: [u64; 4],             // 4 hands represented as bitboards
    pub trump: u8,                   // 0..3
    pub led_suit: u8,                // 0..3 (4 if none)
    pub current_player: u8,          // 0..3 (counter-clockwise)
    pub current_trick_winner: u8,    // 0..3
    pub current_trick_best_card: u8, // 0..39 (40 if none)
    pub cards_played_in_trick: u8,   // 0..4
    pub team_02_score: u8,           // 0..120
    pub team_13_score: u8,           // 0..120
    pub trick_number: u8,            // 0..10
    pub hash: u64,                   // Incremental Zobrist hash (O(1) get_hash)
}

impl fmt::Debug for GameState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GameState {{ player: {}, trick_number: {}, cards_in_trick: {}, team02: {}, team13: {} }}",
            self.current_player,
            self.trick_number,
            self.cards_played_in_trick,
            self.team_02_score,
            self.team_13_score
        )
    }
}

impl GameState {
    /// Initialize a new game state with incremental Zobrist hash.
    pub fn new(hands: [u64; 4], trump: u8, first_player: u8) -> Self {
        // Compute initial Zobrist hash: all hand cards + current player + led suit
        let mut hash = 0u64;
        for p in 0..4 {
            let mut hand = hands[p];
            while hand != 0 {
                let card = hand.trailing_zeros() as usize;
                hash ^= ZOBRIST_TABLE[p][card];
                hand &= hand - 1;
            }
        }
        hash ^= PLAYER_ZOBRIST[first_player as usize];
        hash ^= LED_SUIT_ZOBRIST[4]; // led_suit starts as 4 (none)

        Self {
            hands,
            trump,
            led_suit: 4,
            current_player: first_player,
            current_trick_winner: first_player,
            current_trick_best_card: 40,
            cards_played_in_trick: 0,
            team_02_score: 0,
            team_13_score: 0,
            trick_number: 0,
            hash,
        }
    }

    /// Set led_suit after construction, maintaining the incremental hash.
    /// Only affects the hash if the value actually changes (e.g., from default 4 to suit 0..3).
    pub fn set_led_suit(&mut self, suit: u8) {
        if suit != self.led_suit {
            self.hash ^= LED_SUIT_ZOBRIST[self.led_suit as usize];
            self.hash ^= LED_SUIT_ZOBRIST[suit as usize];
            self.led_suit = suit;
        }
    }

    /// Check if the game is finished (all 10 tricks played).
    #[inline(always)]
    pub fn is_terminal(&self) -> bool {
        self.trick_number == 10
    }

    /// Get legal moves for the current player as a bitboard mask.
    #[inline(always)]
    pub fn legal_moves(&self) -> u64 {
        let hand = self.hands[self.current_player as usize];
        if self.cards_played_in_trick == 0 {
            return hand;
        }
        let suit_mask = 0x3FFu64 << (self.led_suit * 10);
        let suited = hand & suit_mask;
        if suited != 0 {
            suited
        } else {
            hand
        }
    }

    /// Public beats helper — used by simulator and heuristics.
    /// Returns true if `challenger` outranks `current` given trump and led suit.
    #[inline(always)]
    pub fn beats_card(challenger: u8, current: u8, trump: u8, led_suit: u8) -> bool {
        let ch_suit = CARD_SUIT[challenger as usize];
        let ch_rank = CARD_RANK[challenger as usize];
        let cur_suit = CARD_SUIT[current as usize];
        let cur_rank = CARD_RANK[current as usize];

        let ch_is_trump = ch_suit == trump;
        let cur_is_trump = cur_suit == trump;

        if ch_is_trump && !cur_is_trump {
            return true;
        }
        if !ch_is_trump && cur_is_trump {
            return false;
        }
        if ch_is_trump && cur_is_trump {
            return ch_rank > cur_rank;
        }

        // Neither is trump.
        if ch_suit == led_suit && cur_suit == led_suit {
            return ch_rank > cur_rank;
        }
        if ch_suit == led_suit {
            return true;
        }
        false
    }

    /// Helper to determine if challenger beats the current best card.
    #[inline(always)]
    fn beats(&self, challenger: u8, current: u8) -> bool {
        Self::beats_card(challenger, current, self.trump, self.led_suit)
    }
}

// Counter-clockwise seat order: 0 -> 3 -> 2 -> 1 -> 0
pub const TURN_ORDER: [u8; 4] = [0, 3, 2, 1];

#[inline(always)]
pub fn next_player_after(seat: u8) -> u8 {
    match seat {
        0 => 3,
        3 => 2,
        2 => 1,
        1 => 0,
        _ => panic!("Invalid seat: {}", seat),
    }
}

/// Snapshot of GameState before a move, used for make/unmake in alpha_beta.
/// Captures every field that play_card_and_resolve can mutate, plus the
/// pre-move hash value for incremental Zobrist rollback.
#[derive(Clone, Copy)]
pub struct StateSnapshot {
    hand_seat: u64,
    led_suit: u8,
    current_player: u8,
    current_trick_winner: u8,
    current_trick_best_card: u8,
    cards_played_in_trick: u8,
    team_02_score: u8,
    team_13_score: u8,
    trick_number: u8,
    hash: u64,
}

impl GameState {
    /// Capture all mutable state before making a move.
    #[inline(always)]
    pub fn save_snapshot(&self) -> StateSnapshot {
        let seat = self.current_player as usize;
        StateSnapshot {
            hand_seat: self.hands[seat],
            led_suit: self.led_suit,
            current_player: self.current_player,
            current_trick_winner: self.current_trick_winner,
            current_trick_best_card: self.current_trick_best_card,
            cards_played_in_trick: self.cards_played_in_trick,
            team_02_score: self.team_02_score,
            team_13_score: self.team_13_score,
            trick_number: self.trick_number,
            hash: self.hash,
        }
    }

    /// Restore state from a snapshot (undo a move).
    #[inline(always)]
    pub fn restore_snapshot(&mut self, snap: &StateSnapshot) {
        self.hands[snap.current_player as usize] = snap.hand_seat;
        self.led_suit = snap.led_suit;
        self.current_player = snap.current_player;
        self.current_trick_winner = snap.current_trick_winner;
        self.current_trick_best_card = snap.current_trick_best_card;
        self.cards_played_in_trick = snap.cards_played_in_trick;
        self.team_02_score = snap.team_02_score;
        self.team_13_score = snap.team_13_score;
        self.trick_number = snap.trick_number;
        self.hash = snap.hash;
    }

    /// Play card with trick completion scoring.
    /// Maintains incremental Zobrist hash for O(1) get_hash().
    #[inline]
    pub fn play_card_and_resolve(&mut self, card: u8, trick_points: &mut u8) {
        let seat = self.current_player;

        // Incremental Zobrist: XOR out the played card
        self.hash ^= ZOBRIST_TABLE[seat as usize][card as usize];

        let card_mask = 1u64 << card;
        self.hands[seat as usize] &= !card_mask;
        *trick_points += CARD_POINTS[card as usize];

        let card_suit = CARD_SUIT[card as usize];

        if self.cards_played_in_trick == 0 {
            // First card: led_suit transitions from current value → card_suit
            // Note: self.led_suit is typically 4 (none) but may have been set
            // by set_led_suit() when resuming a mid-trick PIMC game state.
            self.hash ^= LED_SUIT_ZOBRIST[self.led_suit as usize];
            self.hash ^= LED_SUIT_ZOBRIST[card_suit as usize];
            self.led_suit = card_suit;
            self.current_trick_winner = seat;
            self.current_trick_best_card = card;
        } else {
            if self.beats(card, self.current_trick_best_card) {
                self.current_trick_winner = seat;
                self.current_trick_best_card = card;
            }
        }

        self.cards_played_in_trick += 1;

        if self.cards_played_in_trick == 4 {
            // Trick complete. Award accumulated points to the winning team.
            let winner = self.current_trick_winner;
            if (winner & 1) == 0 {
                self.team_02_score += *trick_points;
            } else {
                self.team_13_score += *trick_points;
            }

            // Reset trick state
            *trick_points = 0;
            self.cards_played_in_trick = 0;
            // led_suit transitions back to 4 (none)
            self.hash ^= LED_SUIT_ZOBRIST[self.led_suit as usize];
            self.hash ^= LED_SUIT_ZOBRIST[4];
            self.led_suit = 4;
            self.current_trick_best_card = 40;
            // Player change (XOR old, XOR winner)
            self.hash ^= PLAYER_ZOBRIST[self.current_player as usize];
            self.hash ^= PLAYER_ZOBRIST[winner as usize];
            self.current_player = winner;
            self.trick_number += 1;
        } else {
            let next = next_player_after(seat);
            // Player change (XOR old, XOR new)
            self.hash ^= PLAYER_ZOBRIST[self.current_player as usize];
            self.hash ^= PLAYER_ZOBRIST[next as usize];
            self.current_player = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_beats() {
        let mut state = GameState::new([0; 4], 3, 0); // Spades is trump
        state.set_led_suit(0); // Hearts led

        let ace_hearts = 9;
        let seven_hearts = 8;
        let king_hearts = 7;
        let two_hearts = 0;
        let two_spades = 30; // Trump

        assert!(state.beats(two_spades, ace_hearts));
        assert!(state.beats(seven_hearts, king_hearts));
        assert!(state.beats(ace_hearts, seven_hearts));

        let two_diamonds = 10;
        assert!(!state.beats(two_diamonds, two_hearts));
    }

    fn to_vector(mask: u64) -> Vec<u8> {
        let mut v = Vec::new();
        let mut temp = mask;
        while temp != 0 {
            let card = temp.trailing_zeros() as u8;
            v.push(card);
            temp &= temp - 1;
        }
        v
    }

    fn reference_legal_moves(hand: &[u8], led_suit: u8) -> Vec<u8> {
        if led_suit >= 4 {
            return hand.to_vec();
        }
        let suited: Vec<u8> = hand
            .iter()
            .cloned()
            .filter(|&c| CARD_SUIT[c as usize] == led_suit)
            .collect();
        if suited.is_empty() {
            hand.to_vec()
        } else {
            suited
        }
    }

    proptest! {
        #[test]
        fn prop_move_gen_matches_reference(hand_mask in any::<u64>(), led_suit in 0..5u8) {
            // Clean up hand_mask to represent a hand of cards (max 40 cards)
            let valid_cards_mask = 0xFFFFFFFFFFu64; // bits 0..39
            let hand = hand_mask & valid_cards_mask;

            let mut state = GameState::new([0; 4], 0, 0);
            state.hands[0] = hand;
            state.set_led_suit(led_suit);
            state.cards_played_in_trick = if led_suit == 4 { 0 } else { 1 };

            let bitboard_moves = state.legal_moves();
            let hand_vec = to_vector(hand);
            let naive_moves = reference_legal_moves(&hand_vec, led_suit);

            let mut expected_mask = 0u64;
            for &card in &naive_moves {
                expected_mask |= 1u64 << card;
            }

            assert_eq!(bitboard_moves, expected_mask);
        }
    }

    #[test]
    fn test_deck_points() {
        let total: u8 = CARD_POINTS.iter().sum();
        assert_eq!(total, 120);

        for suit in 0..4 {
            let mut suit_pts = 0;
            for rank in 0..10 {
                suit_pts += CARD_POINTS[(suit * 10 + rank) as usize];
            }
            assert_eq!(suit_pts, 30);
        }
    }

    #[test]
    fn test_turn_order() {
        assert_eq!(next_player_after(0), 3);
        assert_eq!(next_player_after(3), 2);
        assert_eq!(next_player_after(2), 1);
        assert_eq!(next_player_after(1), 0);
    }

    #[test]
    fn test_play_card_logic() {
        let mut hands = [0u64; 4];
        for s in 0..4 {
            hands[s] = 0x3FFu64 << (s * 10);
        }

        let mut state = GameState::new(hands, 3, 0); // Spades is trump, Seat 0 leads

        let mut trick_pts = 0;
        state.play_card_and_resolve(0, &mut trick_pts);

        assert_eq!(state.led_suit, 0);
        assert_eq!(state.current_trick_best_card, 0);
        assert_eq!(state.current_trick_winner, 0);
        assert_eq!(state.current_player, 3); // 0 -> 3

        state.play_card_and_resolve(31, &mut trick_pts);
        assert_eq!(state.current_trick_best_card, 31);
        assert_eq!(state.current_trick_winner, 3);
        assert_eq!(state.current_player, 2); // 3 -> 2
    }
}
