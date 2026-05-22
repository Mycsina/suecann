use crate::engine::GameState;

/// Wrapper around GameState for Sueca simulation state.
#[derive(Clone, Debug)]
pub struct SuecaSimulatorGame {
    pub state: GameState,
    pub current_trick: [u8; 4],
    pub current_trick_seats: [u8; 4],
    pub current_trick_len: usize,
    pub voids: [u8; 4], // Suit bitmask per player: 1 << suit
    pub trick_points: u8,
}

impl SuecaSimulatorGame {
    pub fn new(hands: [u64; 4], trump: u8, first_player: u8) -> Self {
        Self {
            state: GameState::new(hands, trump, first_player),
            current_trick: [40; 4],
            current_trick_seats: [4; 4],
            current_trick_len: 0,
            voids: [0; 4],
            trick_points: 0,
        }
    }

    pub fn play_card(&mut self, card: u8) {
        let seat = self.state.current_player;
        let suit = card / 10;

        // Track voids
        if self.current_trick_len > 0 {
            let led_suit = self.current_trick[0] / 10;
            if suit != led_suit {
                self.voids[seat as usize] |= 1 << led_suit;
            }
        }

        self.current_trick[self.current_trick_len] = card;
        self.current_trick_seats[self.current_trick_len] = seat;
        self.current_trick_len += 1;

        self.state
            .play_card_and_resolve(card, &mut self.trick_points);

        if self.state.cards_played_in_trick == 0 {
            self.current_trick_len = 0;
        }
    }
}

/// Determine trick winner from the current trick cards.
#[inline(always)]
pub fn trick_winner_seat(game: &SuecaSimulatorGame) -> Option<u8> {
    if game.current_trick_len == 0 {
        return None;
    }
    let trump = game.state.trump;
    let led_suit = game.current_trick[0] / 10;
    let mut best_seat = game.current_trick_seats[0];
    let mut best_card = game.current_trick[0];

    for i in 1..game.current_trick_len {
        let card = game.current_trick[i];
        let seat = game.current_trick_seats[i];
        if crate::engine::GameState::beats_card(card, best_card, trump, led_suit) {
            best_seat = seat;
            best_card = card;
        }
    }
    Some(best_seat)
}
