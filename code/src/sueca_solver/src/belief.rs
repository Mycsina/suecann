use crate::engine::CARD_POINTS;
use crate::simulator::SuecaSimulatorGame;
use crate::constants::INPUT_COUNT;

// ---------------------------------------------------------------------------
// Bitboard-based belief state encoder — 35 features
//
// Design principle: every feature is computable from information a human
// player can observe. No future-trick knowledge, no opponent hand data.
//
// Features are organized into tactical-affordance groups:
//   0-4   Hand presence & shape
//   5-9   Trick state
//  10-13  Void tracking (public)
//  14-18  Card tracking & boss detection
//  19-21  Tactical affordances (can I win? what does it cost?)
//  22-25  Game state
//  26-28  Hand shape (voids, suit length extremes)
//  29-31  Side-suit depletion
//  32-34  Meta
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

/// Get the rank of the highest unplayed card in a suit.
/// Returns the rank index (0=2, ..., 9=A) of the highest card not yet played.
#[inline(always)]
fn highest_unplayed_rank_in_suit(all_hands: u64, suit: u8) -> i8 {
    let suit_mask = 0x3FFu64 << (suit * 10);
    // Cards still held = all_hands & suit_mask
    // Cards already played = prev_played & suit_mask
    // Highest unplayed = highest bit in (all_hands | prev_played) & suit_mask
    // Actually: all unplayed cards are in all_hands. The highest rank card
    // in the suit that is still in someone's hand.
    let suit_remaining = all_hands & suit_mask;
    if suit_remaining == 0 {
        return -1;
    }
    let bit_idx = 63 - suit_remaining.leading_zeros() as u8;
    (bit_idx - suit * 10) as i8
}

pub fn encode_belief_state(game: &SuecaSimulatorGame, seat: u8) -> [f64; INPUT_COUNT] {
    let mut vec = [0.0f64; INPUT_COUNT];
    let hand = game.state.hands[seat as usize];
    let trump = game.state.trump;
    let position = game.current_trick_len;
    let led_suit = if position > 0 {
        Some(crate::engine::CARD_SUIT[game.current_trick[0] as usize])
    } else {
        None
    };

    // Pre-compute masks used across multiple features
    let all_hands =
        game.state.hands[0] | game.state.hands[1] | game.state.hands[2] | game.state.hands[3];
    let trump_mask = 0x3FFu64 << (trump * 10);

    // Cards played in previous tricks (not in hands and not in current trick)
    let mut trick_mask = 0u64;
    for i in 0..position {
        trick_mask |= 1u64 << game.current_trick[i];
    }
    let prev_played = (!all_hands) & (!trick_mask) & 0x000000FFFFFFFFFFu64;

    // --- 0-4: Hand presence & shape ---

    // 0: Has_Led_Suit — do I hold any card of the led suit?
    if let Some(led) = led_suit {
        let led_mask = 0x3FFu64 << (led * 10);
        vec[0] = if (hand & led_mask) != 0 { 1.0 } else { 0.0 };
    }

    // 1: Has_Trump — do I hold any trump card?
    vec[1] = if (hand & trump_mask) != 0 { 1.0 } else { 0.0 };

    // 2: Led_Suit_Count — how many led suit cards do I hold? / 10.0
    if let Some(led) = led_suit {
        let led_mask = 0x3FFu64 << (led * 10);
        vec[2] = ((hand & led_mask).count_ones() as f64) / 10.0;
    }

    // 3: Trump_Count — how many trumps do I hold? / 10.0
    vec[3] = ((hand & trump_mask).count_ones() as f64) / 10.0;

    // 4: Hand_Point_Density — my hand points / total unplayed points
    let remaining_pts = sum_points(all_hands);
    let hand_pts = sum_points(hand);
    vec[4] = if remaining_pts > 0 {
        (hand_pts as f64) / (remaining_pts as f64)
    } else {
        0.0
    };

    // --- 5-9: Trick state ---

    // 5: Am_I_Leading — am I first to play this trick?
    vec[5] = if position == 0 { 1.0 } else { 0.0 };

    // 6: Am_I_Last_To_Play — am I fourth to play?
    vec[6] = if position == 3 { 1.0 } else { 0.0 };

    // 7: Is_Partner_Winning — is partner currently winning the trick?
    let winner_seat = crate::simulator::trick_winner_seat(game);
    if let Some(w_seat) = winner_seat {
        vec[7] = if (seat % 2) == (w_seat % 2) { 1.0 } else { 0.0 };
    }

    // 8: Trick_Point_Value — points in current trick so far / 44.0
    let mut current_trick_pts = 0;
    for i in 0..position {
        current_trick_pts += CARD_POINTS[game.current_trick[i] as usize] as usize;
    }
    vec[8] = (current_trick_pts as f64) / 44.0;

    // 9: Has_Trick_Been_Cut — trump played when led suit ≠ trump?
    if let Some(led) = led_suit {
        if led != trump {
            let mut cut = false;
            for i in 0..position {
                if crate::engine::CARD_SUIT[game.current_trick[i] as usize] == trump {
                    cut = true;
                    break;
                }
            }
            vec[9] = if cut { 1.0 } else { 0.0 };
        }
    }

    // --- 10-13: Void tracking (public information) ---

    let partner_seat = (seat + 2) % 4;
    if let Some(led) = led_suit {
        vec[10] = if (game.voids[partner_seat as usize] & (1 << led)) != 0 {
            1.0
        } else {
            0.0
        };
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
    }
    vec[13] = if (opp_voids & (1 << trump)) != 0 {
        1.0
    } else {
        0.0
    };

    // --- 14-18: Card tracking & boss detection ---

    // 14: Led_Suit_Ace_Played
    if let Some(led) = led_suit {
        vec[14] = if (prev_played & (1u64 << (led * 10 + 9))) != 0 {
            1.0
        } else {
            0.0
        };
    }

    // 15: Led_Suit_7_Played (manilha)
    if let Some(led) = led_suit {
        vec[15] = if (prev_played & (1u64 << (led * 10 + 8))) != 0 {
            1.0
        } else {
            0.0
        };
    }

    // 16: Trump_Ace_Played
    vec[16] = if (prev_played & (1u64 << (trump * 10 + 9))) != 0 {
        1.0
    } else {
        0.0
    };

    // 17: Holds_Boss_Led — do I hold the highest unplayed card in the led suit?
    if let Some(led) = led_suit {
        let highest_unplayed = highest_unplayed_rank_in_suit(all_hands, led);
        if highest_unplayed >= 0 {
            let my_max = max_rank_in_suit(hand, led);
            vec[17] = if my_max == highest_unplayed { 1.0 } else { 0.0 };
        }
    }

    // 18: Holds_Boss_Trump — do I hold the highest unplayed card in trump?
    {
        let highest_unplayed_trump = highest_unplayed_rank_in_suit(all_hands, trump);
        if highest_unplayed_trump >= 0 {
            let my_max_trump = max_rank_in_suit(hand, trump);
            vec[18] = if my_max_trump == highest_unplayed_trump { 1.0 } else { 0.0 };
        }
    }

    // --- 19-21: Tactical affordances (the "can I win?" question) ---

    // 19: Can_Beat_Winner — can any legal card in my hand beat the current winner?
    // Only meaningful when following (position > 0). When leading, always false.
    if position > 0 {
        let legal = game.state.legal_moves();
        let best_card = game.state.current_trick_best_card;
        let mut can_beat = false;
        let mut temp = legal;
        while temp != 0 {
            let c = temp.trailing_zeros() as u8;
            if crate::engine::GameState::beats_card(c, best_card, trump, led_suit.unwrap_or(0)) {
                can_beat = true;
                break;
            }
            temp &= temp - 1;
        }
        vec[19] = if can_beat { 1.0 } else { 0.0 };
    }

    // 20: Min_Winning_Cost — points of cheapest winning card / 11.0 (0 if cannot win)
    if position > 0 {
        let legal = game.state.legal_moves();
        let best_card = game.state.current_trick_best_card;
        let mut cheapest: Option<u8> = None;
        let mut cheapest_pts = 99u32;
        let mut temp = legal;
        while temp != 0 {
            let c = temp.trailing_zeros() as u8;
            if crate::engine::GameState::beats_card(c, best_card, trump, led_suit.unwrap_or(0)) {
                let pts = CARD_POINTS[c as usize] as u32;
                if pts < cheapest_pts {
                    cheapest_pts = pts;
                    cheapest = Some(c);
                }
            }
            temp &= temp - 1;
        }
        vec[20] = if let Some(c) = cheapest {
            (CARD_POINTS[c as usize] as f64) / 11.0
        } else {
            0.0
        };
    }

    // 21: Min_Sacrifice_Cost — points of cheapest legal card / 11.0
    {
        let legal = game.state.legal_moves();
        let mut cheapest_pts = 99u32;
        let mut temp = legal;
        while temp != 0 {
            let c = temp.trailing_zeros() as u8;
            let pts = CARD_POINTS[c as usize] as u32;
            if pts < cheapest_pts {
                cheapest_pts = pts;
            }
            temp &= temp - 1;
        }
        vec[21] = if cheapest_pts < 99 {
            (cheapest_pts as f64) / 11.0
        } else {
            0.0
        };
    }

    // --- 22-25: Game state ---

    // 22: Game_Pts_Remaining — unplayed points / 120.0
    vec[22] = (remaining_pts as f64) / 120.0;

    // 23: Trick_Number — current trick index / 9.0
    vec[23] = (game.state.trick_number as f64) / 9.0;

    // 24: Trumps_Remaining — unplayed trump count / 10.0
    let trumps_remaining = (all_hands & trump_mask).count_ones();
    vec[24] = (trumps_remaining as f64) / 10.0;

    // 25: Score_Delta — (our_pts − opp_pts + 120) / 240
    let our_pts = game.state.team_02_score as i32;
    let opp_pts = game.state.team_13_score as i32;
    let (our_team, opp_team) = if seat.is_multiple_of(2) {
        (our_pts, opp_pts)
    } else {
        (opp_pts, our_pts)
    };
    let delta = our_team - opp_team;
    vec[25] = ((delta + 120) as f64) / 240.0;

    // --- 26-28: Hand shape ---

    // 26: My_Void_Count — how many suits am I void in? / 3.0 (max 3 since Hand_Point_Density>0 implies non-void)
    let mut my_voids = 0u32;
    for s in 0..4 {
        if (hand & (0x3FFu64 << (s * 10))) == 0 {
            my_voids += 1;
        }
    }
    vec[26] = (my_voids as f64) / 3.0;

    // 27: Longest_Side_Suit — max cards held in a single non-trump, non-led suit / 10.0
    {
        let mut max_len = 0u32;
        for s in 0..4 {
            if s == trump {
                continue;
            }
            if let Some(led) = led_suit {
                if s == led {
                    continue;
                }
            }
            let count = (hand & (0x3FFu64 << (s * 10))).count_ones();
            if count > max_len {
                max_len = count;
            }
        }
        vec[27] = (max_len as f64) / 10.0;
    }

    // 28: Shortest_Side_Suit — min cards held in a single non-trump, non-led suit / 10.0
    // (only counts suits I actually have cards in)
    {
        let mut min_len = 99u32;
        for s in 0..4 {
            if s == trump {
                continue;
            }
            if let Some(led) = led_suit {
                if s == led {
                    continue;
                }
            }
            let count = (hand & (0x3FFu64 << (s * 10))).count_ones();
            if count > 0 && count < min_len {
                min_len = count;
            }
        }
        vec[28] = if min_len < 99 {
            (min_len as f64) / 10.0
        } else {
            0.0
        };
    }

    // --- 29-31: Side-suit depletion ---

    // Extract the 3 side suits in ascending order (excluding trump)
    let mut side_suits = [0u8; 3];
    let mut count_side = 0;
    for s in 0..4 {
        if s != trump {
            side_suits[count_side] = s;
            count_side += 1;
        }
    }

    const DEPLETION_LOOKUP: [f64; 11] =
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

    for j in 0..3 {
        let suit = side_suits[j];
        let suit_mask = 0x3FFu64 << (suit * 10);
        let count = (prev_played & suit_mask).count_ones() as usize;
        vec[29 + j] = DEPLETION_LOOKUP[count];
    }

    // --- 32-34: Meta ---

    // 32: Points_Secured_Us — our team's secured game points / 120.0
    let our_score = if (seat % 2) == 0 {
        game.state.team_02_score
    } else {
        game.state.team_13_score
    };
    vec[32] = (our_score as f64) / 120.0;

    // 33: Known_Void_Suits_Count — suits where any player is known void / 4.0
    let mut void_suits_count = 0u32;
    for suit in 0..4 {
        let mut any_void = false;
        for player in 0..4 {
            if (game.voids[player] & (1 << suit)) != 0 {
                any_void = true;
            }
        }
        if any_void {
            void_suits_count += 1;
        }
    }
    vec[33] = (void_suits_count as f64) / 4.0;

    // 34: Depleted_Suits_Count — fully-depleted suits / 4.0
    let mut depleted_suits_count = 0u32;
    for suit in 0..4 {
        let suit_mask = 0x3FFu64 << (suit * 10);
        let count = (prev_played & suit_mask).count_ones();
        if count == 10 {
            depleted_suits_count += 1;
        }
    }
    vec[34] = (depleted_suits_count as f64) / 4.0;

    vec
}
