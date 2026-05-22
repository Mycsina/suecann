use crate::engine::CARD_POINTS;
use crate::simulator::SuecaSimulatorGame;

// ---------------------------------------------------------------------------
// Bitboard-based belief state encoder
// ---------------------------------------------------------------------------

/// Compute max rank in a given suit within a hand bitmask.
#[inline(always)]
fn max_rank_in_suit(hand: u64, suit: u8) -> i8 {
    let suit_mask = 0x3FFu64 << (suit * 10);
    let suit_hand = hand & suit_mask;
    if suit_hand == 0 {
        -1
    } else {
        let bit_idx = 63 - suit_hand.leading_zeros() as u8;
        (bit_idx - suit * 10) as i8
    }
}

/// Sum card points in a bitmask.
#[inline(always)]
fn sum_points(mut mask: u64) -> usize {
    let mut pts = 0;
    while mask != 0 {
        let card = mask.trailing_zeros() as usize;
        pts += CARD_POINTS[card] as usize;
        mask &= mask - 1;
    }
    pts
}

pub fn encode_belief_state(game: &SuecaSimulatorGame, seat: u8) -> [f64; 21] {
    let mut vec = [0.0f64; 21];
    let hand = game.state.hands[seat as usize];
    let trump = game.state.trump;
    let position = game.current_trick_len;
    let led_suit = if position > 0 {
        Some(game.current_trick[0] / 10)
    } else {
        None
    };

    // --- Hand features (5) ---
    if let Some(led) = led_suit {
        let led_mask = 0x3FFu64 << (led * 10);
        vec[0] = if (hand & led_mask) != 0 { 1.0 } else { 0.0 };
    } else {
        vec[0] = 0.0;
    }

    let trump_mask = 0x3FFu64 << (trump * 10);
    vec[1] = if (hand & trump_mask) != 0 { 1.0 } else { 0.0 };

    if let Some(led) = led_suit {
        let max_r = max_rank_in_suit(hand, led);
        vec[2] = if max_r >= 0 {
            (max_r as f64) / 9.0
        } else {
            0.0
        };
    } else {
        vec[2] = 0.0;
    }

    let max_t = max_rank_in_suit(hand, trump);
    vec[3] = if max_t >= 0 {
        (max_t as f64) / 9.0
    } else {
        0.0
    };

    // Hand point density
    let all_hands =
        game.state.hands[0] | game.state.hands[1] | game.state.hands[2] | game.state.hands[3];
    let remaining_pts = sum_points(all_hands);
    let hand_pts = sum_points(hand);
    vec[4] = if remaining_pts > 0 {
        (hand_pts as f64) / (remaining_pts as f64)
    } else {
        0.0
    };

    // --- Trick features (4) ---
    vec[5] = if position == 0 { 1.0 } else { 0.0 };
    vec[6] = if position == 3 { 1.0 } else { 0.0 };

    let winner_seat = crate::simulator::trick_winner_seat(game);
    if let Some(w_seat) = winner_seat {
        vec[7] = if (seat % 2) == (w_seat % 2) { 1.0 } else { 0.0 };
    } else {
        vec[7] = 0.0;
    }

    let mut current_trick_pts = 0;
    for i in 0..position {
        current_trick_pts += CARD_POINTS[game.current_trick[i] as usize] as usize;
    }
    vec[8] = (current_trick_pts as f64) / 44.0;

    // --- History features (9) ---
    if let Some(led) = led_suit {
        if led != trump {
            let mut cut = false;
            for i in 0..position {
                if (game.current_trick[i] / 10) == trump {
                    cut = true;
                    break;
                }
            }
            vec[9] = if cut { 1.0 } else { 0.0 };
        } else {
            vec[9] = 0.0;
        }
    } else {
        vec[9] = 0.0;
    }

    let partner_seat = (seat + 2) % 4;
    if let Some(led) = led_suit {
        vec[10] = if (game.voids[partner_seat as usize] & (1 << led)) != 0 {
            1.0
        } else {
            0.0
        };
    } else {
        vec[10] = 0.0;
    }
    vec[11] = if (game.voids[partner_seat as usize] & (1 << trump)) != 0 {
        1.0
    } else {
        0.0
    };

    let opp1 = (seat + 1) % 4;
    let opp2 = (seat + 3) % 4;
    let opp_voids = game.voids[opp1 as usize] | game.voids[opp2 as usize];
    if let Some(led) = led_suit {
        vec[12] = if (opp_voids & (1 << led)) != 0 {
            1.0
        } else {
            0.0
        };
    } else {
        vec[12] = 0.0;
    }
    vec[13] = if (opp_voids & (1 << trump)) != 0 {
        1.0
    } else {
        0.0
    };

    // Cards played in previous tricks (not in hands and not in current trick)
    let mut trick_mask = 0u64;
    for i in 0..position {
        trick_mask |= 1u64 << game.current_trick[i];
    }
    let prev_played = (!all_hands) & (!trick_mask);

    if let Some(led) = led_suit {
        vec[14] = if (prev_played & (1u64 << (led * 10 + 9))) != 0 {
            1.0
        } else {
            0.0
        };
        vec[15] = if (prev_played & (1u64 << (led * 10 + 8))) != 0 {
            1.0
        } else {
            0.0
        };
    } else {
        vec[14] = 0.0;
        vec[15] = 0.0;
    }
    vec[16] = if (prev_played & (1u64 << (trump * 10 + 9))) != 0 {
        1.0
    } else {
        0.0
    };

    // Game points remaining
    vec[17] = (remaining_pts as f64) / 120.0;

    // --- Temporal / Strategic features (3) ---
    vec[18] = (game.state.trick_number as f64) / 9.0;

    let trumps_remaining = (all_hands & (0x3FFu64 << (trump * 10))).count_ones();
    vec[19] = (trumps_remaining as f64) / 10.0;

    let our_pts = game.state.team_02_score as i32;
    let opp_pts = game.state.team_13_score as i32;
    let (our_team, opp_team) = if (seat % 2) == 0 {
        (our_pts, opp_pts)
    } else {
        (opp_pts, our_pts)
    };
    let delta = our_team - opp_team;
    vec[20] = ((delta + 120) as f64) / 240.0;

    vec
}
