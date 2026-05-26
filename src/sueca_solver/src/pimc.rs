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
    for i in 0..4 {
        hands_out[i] = 0;
    }
    hands_out[my_seat as usize] = my_hand;

    // 2. Copy and shuffle unknown cards on the stack (Fisher-Yates)
    let num_unknowns = unknown_cards.len();
    let mut shuffled = [0u8; 40];
    shuffled[..num_unknowns].copy_from_slice(unknown_cards);

    let mut r = LcgRng::new(*rng_state);
    let mut i = num_unknowns;
    while i > 1 {
        let rand_idx = (r.next_u64() as usize) % i;
        i -= 1;
        shuffled.swap(i, rand_idx);
    }
    *rng_state = r.next_u64(); // Update RNG state

    // 3. Backtrack/Constraint-solve assignment
    let mut temp_targets = target_sizes;
    // My seat needs 0 more cards because it is already filled
    temp_targets[my_seat as usize] = 0;

    sample_constraints(
        &shuffled[0..num_unknowns],
        0,
        voids,
        &mut temp_targets,
        hands_out,
    )
}

/// Evaluates the Expected Value (EV) of each legal move using PIMC.
/// Returns the list of moves and their computed EV scores.
/// Thread-safe and parallelized using Rayon.
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
) -> Vec<(u8, f64)> {
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
        return vec![(legal_moves[0], 0.0)];
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

    // 3. Parallel world evaluation using Rayon
    // Each world runs independent simulations.
    // We pre-generate world seeds to make the Rayon execution deterministic.
    let mut rng = LcgRng::new(seed);
    let mut seeds = Vec::with_capacity(n_worlds);
    for _ in 0..n_worlds {
        seeds.push(rng.next_u64());
    }

    // Thread-local accumulator: (sum_evs, counts) for the 40 possible cards
    type Accum = ([f64; 40], [u32; 40]);

    let (sum_evs, counts) = seeds
        .into_par_iter()
        .fold(
            || ([0.0f64; 40], [0u32; 40]),
            |mut acc: Accum, w_seed| {
                let mut local_hands = [0u64; 4];
                let mut local_seed = w_seed;

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
                    return acc;
                }

                // Evaluate all legal moves in this world
                THREAD_TT.with(|tt_cell| {
                    let mut tt = tt_cell.borrow_mut();
                    tt.next_generation();

                    for i in 0..legal_count {
                        let m = legal_moves[i];

                        let mut sim_game = GameState::new(local_hands, trump, current_player);
                        sim_game.led_suit = led_suit;
                        sim_game.current_trick_winner = current_trick_winner;
                        sim_game.current_trick_best_card = current_trick_best_card;
                        sim_game.cards_played_in_trick = current_trick_cards.len() as u8;
                        sim_game.team_02_score = team_scores[0];
                        sim_game.team_13_score = team_scores[1];
                        sim_game.trick_number = trick_number;

                        let mut trick_points = 0;
                        for &c in current_trick_cards {
                            trick_points += crate::engine::CARD_POINTS[c as usize];
                        }

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
                        );
                        acc.0[m as usize] += val as f64;
                        acc.1[m as usize] += 1;
                    }
                });
                acc
            },
        )
        .reduce(
            || ([0.0f64; 40], [0u32; 40]),
            |a, b| {
                let mut merged = a;
                for i in 0..40 {
                    merged.0[i] += b.0[i];
                    merged.1[i] += b.1[i];
                }
                merged
            },
        );

    let mut final_evs = Vec::with_capacity(legal_count);
    for i in 0..legal_count {
        let m = legal_moves[i];
        let count = counts[m as usize];
        let ev = if count > 0 {
            sum_evs[m as usize] / (count as f64)
        } else {
            0.0
        };
        final_evs.push((m, ev));
    }

    final_evs
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
        );

        assert!(!evs.is_empty());
        // Since we had legal moves, we should have evs computed for all legal moves of led suit
        // Hearts Ace and 7 are legal moves because Hearts led
        let heart_moves: Vec<u8> = evs
            .iter()
            .map(|&(m, _)| m)
            .filter(|&m| crate::engine::CARD_SUIT[m as usize] == 0)
            .collect();
        assert!(heart_moves.contains(&9));
        assert!(heart_moves.contains(&8));
    }
}
