use crate::pimc;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct SuecaState {
    #[pyo3(get, set)]
    pub my_seat: u8,
    #[pyo3(get, set)]
    pub my_hand: Vec<u8>,
    #[pyo3(get, set)]
    pub played_cards: Vec<u8>,
    #[pyo3(get, set)]
    pub voids: [u8; 4],
    #[pyo3(get, set)]
    pub target_sizes: [u8; 4],
    #[pyo3(get, set)]
    pub trump: u8,
    #[pyo3(get, set)]
    pub led_suit: u8,
    #[pyo3(get, set)]
    pub current_trick: Vec<u8>,
    #[pyo3(get, set)]
    pub current_player: u8,
    #[pyo3(get, set)]
    pub current_trick_winner: u8,
    #[pyo3(get, set)]
    pub current_trick_best_card: u8,
    #[pyo3(get, set)]
    pub team_scores: [u8; 2],
    #[pyo3(get, set)]
    pub trick_number: u8,
}

#[pymethods]
impl SuecaState {
    #[new]
    #[pyo3(signature = (
        my_seat,
        my_hand,
        played_cards,
        voids,
        target_sizes,
        trump,
        led_suit,
        current_trick,
        current_player,
        current_trick_winner,
        current_trick_best_card,
        team_scores,
        trick_number,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        my_seat: u8,
        my_hand: Vec<u8>,
        played_cards: Vec<u8>,
        voids: [u8; 4],
        target_sizes: [u8; 4],
        trump: u8,
        led_suit: u8,
        current_trick: Vec<u8>,
        current_player: u8,
        current_trick_winner: u8,
        current_trick_best_card: u8,
        team_scores: [u8; 2],
        trick_number: u8,
    ) -> Self {
        Self {
            my_seat,
            my_hand,
            played_cards,
            voids,
            target_sizes,
            trump,
            led_suit,
            current_trick,
            current_player,
            current_trick_winner,
            current_trick_best_card,
            team_scores,
            trick_number,
        }
    }
}

pub fn vec_to_mask(v: &[u8]) -> u64 {
    let mut mask = 0u64;
    for &card in v {
        if card < 40 {
            mask |= 1u64 << card;
        }
    }
    mask
}

#[pyfunction]
#[pyo3(signature = (state, n_worlds=10, search_depth=1, seed=42))]
pub fn solve(state: &SuecaState, n_worlds: usize, search_depth: u8, seed: u64) -> Vec<(u8, f64)> {
    let my_hand_mask = vec_to_mask(&state.my_hand);
    let played_cards_mask = vec_to_mask(&state.played_cards);

    pimc::solve_pimc(
        state.my_seat,
        my_hand_mask,
        played_cards_mask,
        state.voids,
        state.target_sizes,
        state.trump,
        state.led_suit,
        &state.current_trick,
        state.current_player,
        state.current_trick_winner,
        state.current_trick_best_card,
        state.team_scores,
        state.trick_number,
        n_worlds,
        search_depth,
        seed,
    )
}

#[pyfunction]
#[pyo3(signature = (states, n_worlds=10, search_depth=1, seed=42))]
pub fn solve_batch(
    states: Vec<SuecaState>,
    n_worlds: usize,
    search_depth: u8,
    seed: u64,
) -> Vec<Vec<(u8, f64)>> {
    states
        .into_par_iter()
        .enumerate()
        .map(|(idx, state)| {
            let my_hand_mask = vec_to_mask(&state.my_hand);
            let played_cards_mask = vec_to_mask(&state.played_cards);
            let state_seed = seed.wrapping_add(idx as u64 * 1337);

            pimc::solve_pimc(
                state.my_seat,
                my_hand_mask,
                played_cards_mask,
                state.voids,
                state.target_sizes,
                state.trump,
                state.led_suit,
                &state.current_trick,
                state.current_player,
                state.current_trick_winner,
                state.current_trick_best_card,
                state.team_scores,
                state.trick_number,
                n_worlds,
                search_depth,
                state_seed,
            )
        })
        .collect()
}
