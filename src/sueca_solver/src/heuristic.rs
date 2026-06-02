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

fn is_partner_secure(
    game: &SuecaSimulatorGame,
    seat: u8,
    winner_card: u8,
    current_hands: u64,
) -> bool {
    let trump = game.state.trump;
    let led_suit = game.current_trick[0] / 10;

    // Check if there are any opponents who still have to play in this trick
    let mut opponents_still_to_play = false;
    for opp in 0..4 {
        if opp % 2 != seat % 2
            && !game.current_trick_seats[0..game.current_trick_len].contains(&opp)
        {
            opponents_still_to_play = true;
            break;
        }
    }

    if !opponents_still_to_play {
        return true;
    }

    // Check if any card outstanding (excluding our own hand) beats the partner's card
    let outstanding_cards = current_hands & !game.state.hands[seat as usize];
    let mut temp = outstanding_cards;
    while temp != 0 {
        let card = temp.trailing_zeros() as u8;
        if GameState::beats_card(card, winner_card, trump, led_suit) {
            return false;
        }
        temp &= temp - 1;
    }

    true
}

fn select_smear_card(suited_mask: u64, led_suit: u8, current_hands: u64) -> u8 {
    let mut best_card = None;
    let mut best_is_master = true;
    let mut best_points = 0;
    let mut best_rank = 10;

    let mut led_master_card = None;
    for r in (0..10).rev() {
        let card = led_suit * 10 + r;
        if (current_hands & (1u64 << card)) != 0 {
            led_master_card = Some(card);
            break;
        }
    }

    let has_multiple = suited_mask.count_ones() > 1;

    let mut temp = suited_mask;
    while temp != 0 {
        let card = temp.trailing_zeros() as u8;
        let is_master = Some(card) == led_master_card && has_multiple;
        let points = CARD_POINTS[card as usize];
        let rank = crate::engine::CARD_RANK[card as usize];

        let is_better = if best_card.is_none() {
            true
        } else if is_master != best_is_master {
            // Prefer non-master
            !is_master
        } else if points != best_points {
            // Prefer higher points
            points > best_points
        } else {
            // Prefer lower rank
            rank < best_rank
        };

        if is_better {
            best_card = Some(card);
            best_is_master = is_master;
            best_points = points;
            best_rank = rank;
        }

        temp &= temp - 1;
    }

    best_card.unwrap()
}

fn select_bleed_card(non_trump_mask: u64, current_hands: u64) -> u8 {
    let mut best_card = None;
    let mut best_is_master = true;
    let mut best_points = 0;
    let mut best_len = 10;
    let mut best_rank = 10;

    let mut temp = non_trump_mask;
    while temp != 0 {
        let card = temp.trailing_zeros() as u8;
        let suit = crate::engine::CARD_SUIT[card as usize];
        let rank = crate::engine::CARD_RANK[card as usize];
        let points = CARD_POINTS[card as usize];
        let len = (non_trump_mask & (0x3FFu64 << (suit * 10))).count_ones();

        let mut master_card = None;
        for r in (0..10).rev() {
            let m_card = suit * 10 + r;
            if (current_hands & (1u64 << m_card)) != 0 {
                master_card = Some(m_card);
                break;
            }
        }
        let is_master = Some(card) == master_card;

        let is_better = if best_card.is_none() {
            true
        } else if is_master != best_is_master {
            !is_master
        } else if points != best_points {
            points > best_points
        } else if len != best_len {
            len < best_len
        } else {
            rank < best_rank
        };

        if is_better {
            best_card = Some(card);
            best_is_master = is_master;
            best_points = points;
            best_len = len;
            best_rank = rank;
        }

        temp &= temp - 1;
    }

    best_card.unwrap()
}

pub fn select_card_heuristic(game: &SuecaSimulatorGame, seat: u8) -> u8 {
    let legal = game.state.legal_moves();
    let trump = game.state.trump;

    let mut current_hands = 0u64;
    for h in &game.state.hands {
        current_hands |= h;
    }

    if game.current_trick_len == 0 {
        // Leading: cash guaranteed master card first, otherwise probe longest non-trump suit.
        if let Some(master_card) = get_deterministic_cash_master_card(game, seat) {
            return master_card;
        }

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
            // Probe with the lowest card in the longest suit to extract trumps or high cards safely.
            return suited.trailing_zeros() as u8;
        }

        let trumps = legal & (0x3FFu64 << (trump * 10));
        return trumps.trailing_zeros() as u8;
    }

    let led_suit = crate::engine::CARD_SUIT[game.current_trick[0] as usize];
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
            if is_partner_secure(game, seat, winner_card, current_hands) {
                return select_smear_card(suited, led_suit, current_hands);
            }
            
            // Partner is winning but NOT secure. Can we protect the trick?
            let win_rank = crate::engine::CARD_RANK[winner_card as usize];
            let mut higher_protective_cards = 0u64;
            let mut temp = suited;
            
            while temp != 0 {
                let card = temp.trailing_zeros() as u8;
                let rank = crate::engine::CARD_RANK[card as usize];
                // If we hold a card higher than our partner's winning card, track it
                if rank > win_rank {
                    higher_protective_cards |= 1u64 << card;
                }
                temp &= temp - 1;
            }

            if higher_protective_cards != 0 {
                // Play our highest card to actively secure the trick and protect our partner's investment
                return 63 - higher_protective_cards.leading_zeros() as u8;
            }
            
            // We can't help or improve the trick, so safely dump our lowest card
            return suited.trailing_zeros() as u8;
        }

        let win_suit = crate::engine::CARD_SUIT[winner_card as usize];
        let win_rank = crate::engine::CARD_RANK[winner_card as usize];

        let mut beating = 0u64;
        let mut temp = suited;
        while temp != 0 {
            let card = temp.trailing_zeros() as u8;
            let rank = crate::engine::CARD_RANK[card as usize];
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
                let mut winner_card = 0;
                for i in 0..game.current_trick_len {
                    if game.current_trick_seats[i] == winner_seat {
                        winner_card = game.current_trick[i];
                        break;
                    }
                }
                if is_partner_secure(game, seat, winner_card, current_hands) {
                    return select_bleed_card(non_trump, current_hands);
                } else {
                    return dump_lowest_off_suit();
                }
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

            if crate::engine::CARD_SUIT[winner_card as usize] == trump {
                let win_rank = crate::engine::CARD_RANK[winner_card as usize];
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
fn would_beat(card: u8, game: &SuecaSimulatorGame) -> bool {
    if game.current_trick_len == 0 {
        return true;
    }
    let trump = game.state.trump;
    let led_suit = crate::engine::CARD_SUIT[game.current_trick[0] as usize];
    let mut best_card = game.current_trick[0];
    for i in 1..game.current_trick_len {
        let c = game.current_trick[i];
        if GameState::beats_card(c, best_card, trump, led_suit) {
            best_card = c;
        }
    }
    GameState::beats_card(card, best_card, trump, led_suit)
}

#[inline(always)]
fn get_suit_priority(suit: u8) -> u8 {
    match suit {
        3 => 0, // Spades highest priority
        0 => 1, // Hearts
        1 => 2, // Diamonds
        2 => 3, // Clubs
        _ => 4,
    }
}

#[inline]
fn get_deterministic_cash_master_card(game: &SuecaSimulatorGame, seat: u8) -> Option<u8> {
    let trump = game.state.trump;
    let hand = game.state.hands[seat as usize];

    // Find all master cards in non-trump suits
    let mut current_hands = 0u64;
    for h in &game.state.hands {
        current_hands |= h;
    }

    let mut master_cards = Vec::new();
    let mut suit_lengths = [0; 4];
    for s in 0..4 {
        let mask = 0x3FFu64 << (s * 10);
        suit_lengths[s as usize] = (hand & mask).count_ones() as usize;

        if s != trump {
            // Find the highest rank card in suit s that is still in play
            for r in (0..10).rev() {
                let card = s * 10 + r;
                if (current_hands & (1u64 << card)) != 0 {
                    // This is the highest rank card of suit s still in play.
                    // If the player holds it, it's a master card for them!
                    if (hand & (1u64 << card)) != 0 {
                        master_cards.push(card);
                    }
                    break;
                }
            }
        }
    }

    if master_cards.is_empty() {
        return None;
    }

    // Prioritize by shortest suit length, then suit priority
    master_cards.sort_by(|&a, &b| {
        let suit_a = crate::engine::CARD_SUIT[a as usize];
        let suit_b = crate::engine::CARD_SUIT[b as usize];
        let len_a = suit_lengths[suit_a as usize];
        let len_b = suit_lengths[suit_b as usize];
        if len_a != len_b {
            len_a.cmp(&len_b) // shorter suit first
        } else {
            get_suit_priority(suit_a).cmp(&get_suit_priority(suit_b)) // higher priority suit first
        }
    });

    Some(master_cards[0])
}

#[inline]
fn get_deterministic_load_points_card(game: &SuecaSimulatorGame, _seat: u8) -> Option<u8> {
    let trump = game.state.trump;
    let legal = game.state.legal_moves();

    let mut point_cards = Vec::new();
    let mut temp = legal;
    while temp != 0 {
        let c = temp.trailing_zeros() as u8;
        let pts = CARD_POINTS[c as usize];
        if pts > 0 {
            point_cards.push(c);
        }
        temp &= temp - 1;
    }

    if point_cards.is_empty() {
        return None;
    }

    point_cards.sort_by(|&a, &b| {
        let pts_a = CARD_POINTS[a as usize];
        let pts_b = CARD_POINTS[b as usize];
        if pts_a != pts_b {
            pts_b.cmp(&pts_a) // higher points first
        } else {
            let is_trump_a = crate::engine::CARD_SUIT[a as usize] == trump;
            let is_trump_b = crate::engine::CARD_SUIT[b as usize] == trump;
            if is_trump_a != is_trump_b {
                // non-trump first
                is_trump_a.cmp(&is_trump_b)
            } else {
                get_suit_priority(crate::engine::CARD_SUIT[a as usize])
                    .cmp(&get_suit_priority(crate::engine::CARD_SUIT[b as usize]))
            }
        }
    });

    Some(point_cards[0])
}

fn get_highest_ranking_card(mask: u64) -> u8 {
    let mut best_card = 40;
    let mut best_rank = 0;
    let mut temp = mask;
    while temp != 0 {
        let c = temp.trailing_zeros() as u8;
        let rank = crate::engine::CARD_RANK[c as usize];
        let suit = crate::engine::CARD_SUIT[c as usize];
        if best_card == 40 || rank > best_rank {
            best_card = c;
            best_rank = rank;
        } else if rank == best_rank
            && get_suit_priority(suit)
                < get_suit_priority(crate::engine::CARD_SUIT[best_card as usize])
        {
            best_card = c;
        }
        temp &= temp - 1;
    }
    best_card
}

pub fn resolve_intent(intent: usize, game: &SuecaSimulatorGame, seat: u8) -> u8 {
    let legal = game.state.legal_moves();
    let trump = game.state.trump;
    let leading = game.current_trick_len == 0;
    let card;

    let duck_or_dump = || {
        let mut best_card = 40;
        let mut best_rank = 10;
        let mut temp = legal;
        while temp != 0 {
            let c = temp.trailing_zeros() as u8;
            let rank = crate::engine::CARD_RANK[c as usize];
            if rank < best_rank
                || (rank == best_rank
                    && crate::engine::CARD_SUIT[c as usize]
                        < crate::engine::CARD_SUIT[best_card as usize])
            {
                best_rank = rank;
                best_card = c;
            }
            temp &= temp - 1;
        }
        best_card
    };

    match intent {
        0 => {
            // MAX_FORCE (Aggressive / Control)
            if leading {
                let trump_mask = 0x3FFu64 << (trump * 10);
                let trump_cards = legal & trump_mask;
                if trump_cards != 0 {
                    card = get_highest_ranking_card(trump_cards);
                } else if let Some(master_card) = get_deterministic_cash_master_card(game, seat) {
                    card = master_card;
                } else {
                    card = duck_or_dump();
                }
            } else {
                let led_suit = crate::engine::CARD_SUIT[game.current_trick[0] as usize];
                let suited = legal & (0x3FFu64 << (led_suit * 10));
                if suited != 0 {
                    card = get_highest_ranking_card(suited);
                } else {
                    let trump_mask = 0x3FFu64 << (trump * 10);
                    let trump_cards = legal & trump_mask;
                    if trump_cards != 0 {
                        card = get_highest_ranking_card(trump_cards);
                    } else {
                        card = get_highest_ranking_card(legal);
                    }
                }
            }
        }
        1 => {
            // MIN_FORCE (Passive / Resource Saving)
            if leading {
                let mut counts = [0; 4];
                let hand = game.state.hands[seat as usize];
                for s in 0..4 {
                    if s != trump {
                        counts[s as usize] = (hand & (0x3FFu64 << (s * 10))).count_ones();
                    }
                }
                let mut longest_suit = None;
                let mut max_count = 0;
                for s in 0..4 {
                    if s != trump {
                        let count = counts[s as usize];
                        if count > 0 && count >= max_count {
                            max_count = count;
                            longest_suit = Some(s);
                        }
                    }
                }
                if let Some(s) = longest_suit {
                    let suit_cards = hand & (0x3FFu64 << (s * 10));
                    card = suit_cards.trailing_zeros() as u8;
                } else {
                    card = duck_or_dump();
                }
            } else {
                let led_suit = crate::engine::CARD_SUIT[game.current_trick[0] as usize];
                let suited = legal & (0x3FFu64 << (led_suit * 10));
                if suited != 0 {
                    card = suited.trailing_zeros() as u8;
                } else {
                    card = duck_or_dump();
                }
            }
        }
        2 => {
            // EFFICIENT_WIN (Tactical Exploitation)
            if leading {
                let trump_mask = 0x3FFu64 << (trump * 10);
                let trump_cards = legal & trump_mask;
                if trump_cards.count_ones() > 1 {
                    card = trump_cards.trailing_zeros() as u8;
                } else {
                    let mut counts = [0; 4];
                    let hand = game.state.hands[seat as usize];
                    for s in 0..4 {
                        if s != trump {
                            counts[s as usize] = (hand & (0x3FFu64 << (s * 10))).count_ones();
                        }
                    }
                    let mut longest_suit = None;
                    let mut max_count = 0;
                    for s in 0..4 {
                        if s != trump {
                            let count = counts[s as usize];
                            if count > 0 && count >= max_count {
                                max_count = count;
                                longest_suit = Some(s);
                            }
                        }
                    }
                    if let Some(s) = longest_suit {
                        let suit_cards = hand & (0x3FFu64 << (s * 10));
                        card = suit_cards.trailing_zeros() as u8;
                    } else {
                        card = duck_or_dump();
                    }
                }
            } else {
                let led_suit = crate::engine::CARD_SUIT[game.current_trick[0] as usize];
                let suited = legal & (0x3FFu64 << (led_suit * 10));
                if suited != 0 {
                    let mut cheapest_winning_card = None;
                    let mut best_rank = 10;
                    let mut temp = suited;
                    while temp != 0 {
                        let c = temp.trailing_zeros() as u8;
                        if would_beat(c, game) {
                            let rank = crate::engine::CARD_RANK[c as usize];
                            if rank < best_rank {
                                best_rank = rank;
                                cheapest_winning_card = Some(c);
                            }
                        }
                        temp &= temp - 1;
                    }
                    if let Some(c) = cheapest_winning_card {
                        card = c;
                    } else {
                        card = suited.trailing_zeros() as u8;
                    }
                } else {
                    let trump_mask = 0x3FFu64 << (trump * 10);
                    let trump_cards = legal & trump_mask;
                    let mut cheapest_winning_trump = None;
                    let mut best_rank = 10;
                    let mut temp = trump_cards;
                    while temp != 0 {
                        let c = temp.trailing_zeros() as u8;
                        if would_beat(c, game) {
                            let rank = crate::engine::CARD_RANK[c as usize];
                            if rank < best_rank {
                                best_rank = rank;
                                cheapest_winning_trump = Some(c);
                            }
                        }
                        temp &= temp - 1;
                    }
                    if let Some(c) = cheapest_winning_trump {
                        card = c;
                    } else {
                        card = duck_or_dump();
                    }
                }
            }
        }
        3 => {
            // EQUITY_BUILDER (Partnership / Voids)
            if leading {
                let mut counts = [0; 4];
                let hand = game.state.hands[seat as usize];
                for s in 0..4 {
                    if s != trump {
                        counts[s as usize] = (hand & (0x3FFu64 << (s * 10))).count_ones();
                    }
                }
                let mut shortest_suit = None;
                let mut min_count = u32::MAX;
                for s in 0..4 {
                    if s != trump {
                        let count = counts[s as usize];
                        if count > 0 && count <= min_count {
                            min_count = count;
                            shortest_suit = Some(s);
                        }
                    }
                }
                if let Some(s) = shortest_suit {
                    let suit_cards = hand & (0x3FFu64 << (s * 10));
                    card = suit_cards.trailing_zeros() as u8;
                } else {
                    card = duck_or_dump();
                }
            } else {
                let partner_winning = game.state.current_trick_winner == (seat + 2) % 4;
                let mut load_card = None;
                if partner_winning {
                    load_card = get_deterministic_load_points_card(game, seat);
                }
                if let Some(c) = load_card {
                    card = c;
                } else {
                    let led_suit = crate::engine::CARD_SUIT[game.current_trick[0] as usize];
                    let suited = legal & (0x3FFu64 << (led_suit * 10));
                    if suited != 0 {
                        card = suited.trailing_zeros() as u8;
                    } else {
                        card = duck_or_dump();
                    }
                }
            }
        }
        _ => {
            card = duck_or_dump();
        }
    }

    card
}

pub fn select_card_from_outputs(
    outputs: &[f64; crate::constants::OUTPUT_COUNT],
    game: &SuecaSimulatorGame,
    seat: u8,
    rng: &mut LcgRng,
) -> u8 {
    let adjusted_outputs = *outputs;

    let mut max_val = adjusted_outputs[0];
    for &val in &adjusted_outputs[1..] {
        if val > max_val {
            max_val = val;
        }
    }

    let mut best_intents = [0usize; crate::constants::OUTPUT_COUNT];
    let mut best_count = 0;
    for (i, &val) in adjusted_outputs.iter().enumerate() {
        if (val - max_val).abs() < 1e-9 {
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

pub fn select_card_heuristic_old(game: &SuecaSimulatorGame, seat: u8) -> u8 {
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

    let led_suit = crate::engine::CARD_SUIT[game.current_trick[0] as usize];
    let suited = legal & (0x3FFu64 << (led_suit * 10));

    if suited != 0 {
        // Following: try to win cheaply, otherwise dump.
        let dump_suited_point_aware = |suited_cards: u64| -> u8 {
            let mut best_dump_card = None;
            let mut min_dump_score = i32::MAX;
            let mut temp = suited_cards;
            while temp != 0 {
                let card = temp.trailing_zeros() as u8;
                let points = crate::engine::CARD_POINTS[card as usize] as i32;
                let rank = crate::engine::CARD_RANK[card as usize] as i32;
                let dump_score = (points << 8) + rank;
                if dump_score < min_dump_score {
                    min_dump_score = dump_score;
                    best_dump_card = Some(card);
                }
                temp &= temp - 1;
            }
            best_dump_card.unwrap()
        };

        let winner_seat = trick_winner_seat(game).unwrap();
        let mut winner_card = 0;
        for i in 0..game.current_trick_len {
            if game.current_trick_seats[i] == winner_seat {
                winner_card = game.current_trick[i];
                break;
            }
        }

        if (seat % 2) == (winner_seat % 2) {
            return dump_suited_point_aware(suited);
        }

        let win_suit = crate::engine::CARD_SUIT[winner_card as usize];
        let win_rank = crate::engine::CARD_RANK[winner_card as usize];

        let mut beating = 0u64;
        let mut temp = suited;
        while temp != 0 {
            let card = temp.trailing_zeros() as u8;
            let rank = crate::engine::CARD_RANK[card as usize];
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
        dump_suited_point_aware(suited)
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

            if crate::engine::CARD_SUIT[winner_card as usize] == trump {
                let win_rank = crate::engine::CARD_RANK[winner_card as usize];
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_archetype_alignment() {
        let mut rng = LcgRng::new(12345);
        for _ in 0..1000 {
            let mut deck: Vec<u8> = (0..40).collect();
            for i in (1..40).rev() {
                let j = rng.gen_range(0..((i + 1) as usize));
                deck.swap(i as usize, j);
            }

            let mut hands = [0u64; 4];
            for player in 0..4 {
                for card_idx in 0..10 {
                    hands[player] |= 1u64 << deck[player * 10 + card_idx];
                }
            }

            let trump = rng.gen_range(0..4) as u8;
            let mut game = SuecaSimulatorGame::new(hands, trump, 0);

            for _ in 0..rng.gen_range(0..15) {
                let legal_moves = game.state.legal_moves();
                if legal_moves.count_ones() <= 1 || game.state.trick_number >= 10 {
                    break;
                }
                let mut temp = legal_moves;
                let mut moves = Vec::new();
                while temp != 0 {
                    moves.push(temp.trailing_zeros() as u8);
                    temp &= temp - 1;
                }
                let play = moves[rng.gen_range(0..moves.len()) as usize];
                game.play_card(play);
            }

            if game.state.trick_number >= 10 {
                continue;
            }

            let seat = game.state.current_player;
            let legal_moves = game.state.legal_moves();
            if legal_moves.count_ones() <= 1 {
                continue;
            }

            let mut temp = legal_moves;
            while temp != 0 {
                let expert_card = temp.trailing_zeros() as u8;

                let mut target_archetype = None;
                for &archetype in &[2, 0, 3, 1] {
                    let resolved_card = resolve_intent(archetype, &game, seat);
                    if resolved_card == expert_card {
                        target_archetype = Some(archetype);
                        break;
                    }
                }

                if let Some(archetype) = target_archetype {
                    let resolved_card = resolve_intent(archetype, &game, seat);
                    assert_eq!(
                        resolved_card, expert_card,
                        "Divergence found: archetype {} resolved to {} instead of expert card {} at state trick_len={} led_suit={}",
                        archetype, resolved_card, expert_card, game.current_trick_len,
                        if game.current_trick_len > 0 { game.current_trick[0]/10 } else { 4 }
                    );
                }

                temp &= temp - 1;
            }
        }
    }
}
