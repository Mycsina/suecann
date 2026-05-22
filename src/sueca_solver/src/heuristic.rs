use crate::engine::{GameState, CARD_POINTS};
use crate::rng::LcgRng;
use crate::simulator::{trick_winner_seat, SuecaSimulatorGame};

// ---------------------------------------------------------------------------
// Card Selection Heuristics
// ---------------------------------------------------------------------------

pub fn select_card_random(legal_mask: u64, rng: &mut LcgRng) -> u8 {
    let count = legal_mask.count_ones() as usize;
    let idx = rng.gen_range(0..count);
    let mut temp = legal_mask;
    for _ in 0..idx {
        temp &= temp - 1;
    }
    temp.trailing_zeros() as u8
}

pub fn select_card_heuristic(game: &SuecaSimulatorGame, seat: u8) -> u8 {
    let legal = game.state.legal_moves();
    let trump = game.state.trump;

    if game.current_trick_len == 0 {
        // Leading: strongest non-trump card in longest non-trump suit.
        let mut counts = [0; 4];
        let mut suit_legal = [0u64; 4];
        for s in 0..4 {
            if s != trump {
                let mask = (0x3FFu64 << (s * 10)) & legal;
                counts[s as usize] = mask.count_ones();
                suit_legal[s as usize] = mask;
            }
        }

        let mut best_suit = None;
        let mut max_val = (0, 0);
        for s in 0..4 {
            if s != trump {
                let count = counts[s as usize];
                if count > 0 && (count, s) >= max_val {
                    max_val = (count, s);
                    best_suit = Some(s);
                }
            }
        }

        if let Some(s) = best_suit {
            let suited = suit_legal[s as usize];
            return 63 - suited.leading_zeros() as u8;
        }

        let trumps = legal & (0x3FFu64 << (trump * 10));
        return trumps.trailing_zeros() as u8;
    }

    let led_suit = game.current_trick[0] / 10;
    let suited = legal & (0x3FFu64 << (led_suit * 10));

    if suited != 0 {
        // Following: try to win cheaply, otherwise dump.
        let winner_seat = trick_winner_seat(game).unwrap();
        let mut winner_card = 0;
        for i in 0..game.current_trick_len {
            if game.current_trick_seats[i] == winner_seat {
                winner_card = game.current_trick[i];
                break;
            }
        }

        if (seat % 2) == (winner_seat % 2) {
            return suited.trailing_zeros() as u8;
        }

        let win_suit = winner_card / 10;
        let win_rank = winner_card % 10;

        let mut beating = 0u64;
        let mut temp = suited;
        while temp != 0 {
            let card = temp.trailing_zeros() as u8;
            let rank = card % 10;
            let beats_winner = if win_suit == trump {
                led_suit == trump && rank > win_rank
            } else if win_suit == led_suit {
                rank > win_rank
            } else {
                true
            };
            if beats_winner {
                beating |= 1u64 << card;
            }
            temp &= temp - 1;
        }

        if beating != 0 {
            return beating.trailing_zeros() as u8;
        }
        suited.trailing_zeros() as u8
    } else {
        // Void: cut with lowest trump if partner isn't winning.
        let winner_seat = trick_winner_seat(game).unwrap();
        let trump_cards = legal & (0x3FFu64 << (trump * 10));
        let non_trump = legal & !(0x3FFu64 << (trump * 10));

        let dump_lowest_off_suit = || -> u8 {
            if non_trump != 0 {
                for r in 0..10 {
                    for s in 0..4 {
                        if s != trump {
                            let card = s * 10 + r;
                            if (non_trump & (1u64 << card)) != 0 {
                                return card;
                            }
                        }
                    }
                }
            }
            legal.trailing_zeros() as u8
        };

        if (seat % 2) == (winner_seat % 2) {
            if non_trump != 0 {
                return dump_lowest_off_suit();
            }
            return trump_cards.trailing_zeros() as u8;
        }

        if trump_cards != 0 {
            let mut winner_card = 0;
            for i in 0..game.current_trick_len {
                if game.current_trick_seats[i] == winner_seat {
                    winner_card = game.current_trick[i];
                    break;
                }
            }

            if (winner_card / 10) == trump {
                let win_rank = winner_card % 10;
                let higher_trumps = trump_cards & (0x3FFu64 << (trump * 10 + win_rank + 1));
                if higher_trumps != 0 {
                    return higher_trumps.trailing_zeros() as u8;
                }
                return dump_lowest_off_suit();
            } else {
                return trump_cards.trailing_zeros() as u8;
            }
        }

        dump_lowest_off_suit()
    }
}

// ---------------------------------------------------------------------------
// WANN Output Intent Resolver (Legal Subsystem)
// ---------------------------------------------------------------------------

#[inline(always)]
fn card_strength(card: u8, trump: u8, led_suit: Option<u8>) -> f64 {
    let suit = card / 10;
    let rank = card % 10;
    if suit == trump {
        100.0 + rank as f64
    } else if let Some(led) = led_suit {
        if suit == led {
            rank as f64
        } else {
            -1.0
        }
    } else {
        -1.0
    }
}

#[inline(always)]
fn would_beat(card: u8, game: &SuecaSimulatorGame) -> bool {
    if game.current_trick_len == 0 {
        return true;
    }
    let trump = game.state.trump;
    let led_suit = game.current_trick[0] / 10;
    let mut best_card = game.current_trick[0];
    for i in 1..game.current_trick_len {
        let c = game.current_trick[i];
        if GameState::beats_card(c, best_card, trump, led_suit) {
            best_card = c;
        }
    }
    GameState::beats_card(card, best_card, trump, led_suit)
}

pub fn resolve_intent(intent: usize, game: &SuecaSimulatorGame, _seat: u8) -> (u8, bool) {
    let legal = game.state.legal_moves();
    let trump = game.state.trump;
    let led_suit = if game.current_trick_len > 0 {
        Some(game.current_trick[0] / 10)
    } else {
        None
    };

    let mut was_illegal = false;
    let card;

    let duck_or_dump = || {
        let mut best_card = 40;
        let mut best_rank = 10;
        let mut temp = legal;
        while temp != 0 {
            let c = temp.trailing_zeros() as u8;
            let rank = c % 10;
            if rank < best_rank || (rank == best_rank && (c / 10) < (best_card / 10)) {
                best_rank = rank;
                best_card = c;
            }
            temp &= temp - 1;
        }
        best_card
    };

    match intent {
        0 => {
            card = duck_or_dump();
        }
        1 => {
            let mut takers = 0u64;
            let mut temp = legal;
            while temp != 0 {
                let c = temp.trailing_zeros() as u8;
                if would_beat(c, game) {
                    takers |= 1u64 << c;
                }
                temp &= temp - 1;
            }
            if takers == 0 {
                was_illegal = true;
                card = duck_or_dump();
            } else {
                let mut best_card = 40;
                let mut best_rank = 10;
                let mut temp = takers;
                while temp != 0 {
                    let c = temp.trailing_zeros() as u8;
                    let rank = c % 10;
                    if rank < best_rank || (rank == best_rank && (c / 10) < (best_card / 10)) {
                        best_rank = rank;
                        best_card = c;
                    }
                    temp &= temp - 1;
                }
                card = best_card;
            }
        }
        2 => {
            let mut best_card = 40;
            let mut best_strength = -10.0;
            let mut temp = legal;
            while temp != 0 {
                let c = temp.trailing_zeros() as u8;
                let strength = card_strength(c, trump, led_suit);
                if strength > best_strength {
                    best_strength = strength;
                    best_card = c;
                }
                temp &= temp - 1;
            }
            card = best_card;
        }
        3 => {
            let mut best_card = 40;
            let mut best_val = (-1, -1, -1);
            let mut temp = legal;
            while temp != 0 {
                let c = temp.trailing_zeros() as u8;
                let pts = CARD_POINTS[c as usize] as i32;
                let rank = (c % 10) as i32;
                let suit = (c / 10) as i32;
                let val = (pts, rank, suit);
                if val > best_val {
                    best_val = val;
                    best_card = c;
                }
                temp &= temp - 1;
            }
            card = best_card;
        }
        4 => {
            if let Some(led) = led_suit {
                if led == trump {
                    let trump_mask = 0x3FFu64 << (trump * 10);
                    let trump_cards = legal & trump_mask;
                    if trump_cards == 0 {
                        was_illegal = true;
                        card = duck_or_dump();
                    } else {
                        card = trump_cards.trailing_zeros() as u8;
                    }
                } else {
                    let has_led_suit = (legal & (0x3FFu64 << (led * 10))) != 0;
                    if has_led_suit {
                        was_illegal = true;
                        card = duck_or_dump();
                    } else {
                        let trump_mask = 0x3FFu64 << (trump * 10);
                        let trump_cards = legal & trump_mask;
                        if trump_cards == 0 {
                            was_illegal = true;
                            card = duck_or_dump();
                        } else {
                            card = trump_cards.trailing_zeros() as u8;
                        }
                    }
                }
            } else {
                was_illegal = true;
                card = duck_or_dump();
            }
        }
        _ => {
            was_illegal = true;
            card = duck_or_dump();
        }
    }

    (card, was_illegal)
}

pub fn select_card_from_outputs(
    outputs: &[f64; 5],
    game: &SuecaSimulatorGame,
    seat: u8,
    rng: &mut LcgRng,
) -> (u8, bool) {
    let mut max_val = outputs[0];
    for i in 1..5 {
        if outputs[i] > max_val {
            max_val = outputs[i];
        }
    }

    let mut best_intents = [0usize; 5];
    let mut best_count = 0;
    for i in 0..5 {
        if (outputs[i] - max_val).abs() < 1e-9 {
            best_intents[best_count] = i;
            best_count += 1;
        }
    }

    let chosen_intent = if best_count == 1 {
        best_intents[0]
    } else {
        best_intents[rng.gen_range(0..best_count)]
    };

    resolve_intent(chosen_intent, game, seat)
}
