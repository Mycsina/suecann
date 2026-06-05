use wasm_bindgen::prelude::*;
use serde::Serialize;
use sueca_solver::simulator::SuecaSimulatorGame;
use sueca_solver::engine::{CARD_SUIT, CARD_RANK, CARD_POINTS};
use sueca_wann::wann_network::RustWannNetwork;
use sueca_wann::genome::JsonGenomeJoint;
use sueca_solver::belief::encode_belief_state;
use sueca_solver::heuristic::resolve_intent;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_pcg::Pcg64;

#[wasm_bindgen]
pub struct WannSuecaGameSession {
    game: SuecaSimulatorGame,
    lead_brain: RustWannNetwork,
    follow_brain: RustWannNetwork,
    rng: Pcg64,
    scratchpad: Vec<f64>,
    sweep_weights: Vec<f64>,
    last_trick: Option<WasmLastTrick>,
    bot_types: [u8; 4],
}

#[derive(Serialize, Clone)]
pub struct WasmLastTrick {
    pub cards: [u8; 4],
    pub seats: [u8; 4],
    pub winner: u8,
    pub points: u8,
}

#[derive(Serialize)]
pub struct WasmGameState {
    pub trump: u8,
    pub player_hand: Vec<u8>,
    pub legal_moves: Vec<u8>,
    pub other_hands_sizes: [usize; 4],
    pub other_hands: [Vec<u8>; 4],
    pub current_trick: Vec<u8>,
    pub current_trick_seats: Vec<u8>,
    pub led_suit: u8,
    pub current_player: u8,
    pub team_02_score: u8,
    pub team_13_score: u8,
    pub trick_number: u8,
    pub voids: [u8; 4],
    pub is_over: bool,
    pub winner_team: Option<u8>,
    pub game_points_02: u8,
    pub game_points_13: u8,
    pub last_trick: Option<WasmLastTrick>,
}

#[wasm_bindgen]
impl WannSuecaGameSession {
    #[wasm_bindgen(constructor)]
    pub fn new(genome_json: &str, seed: u64) -> Result<WannSuecaGameSession, String> {
        // Parse WANN genome
        let genome_joint: JsonGenomeJoint = serde_json::from_str(genome_json)
            .map_err(|e| format!("Failed to parse genome JSON: {}", e))?;
        
        let lead_genome = genome_joint.lead
            .ok_or_else(|| "Lead genome missing in JSON".to_string())?
            .to_genome();
        let follow_genome = genome_joint.follow
            .ok_or_else(|| "Follow genome missing in JSON".to_string())?
            .to_genome();

        let lead_brain = lead_genome.to_rust_wann();
        let follow_brain = follow_genome.to_rust_wann();
        
        let max_nodes = lead_brain.num_nodes.max(follow_brain.num_nodes);
        let scratchpad = vec![0.0f64; max_nodes];

        // Initialize game & RNG
        let mut rng = Pcg64::seed_from_u64(seed);
        
        // Shuffle deck
        let mut deck: Vec<u8> = (0..40).collect();
        deck.shuffle(&mut rng);

        let mut hands = [0u64; 4];
        for player in 0..4 {
            for card_idx in 0..10 {
                let card = deck[player * 10 + card_idx];
                hands[player] |= 1u64 << card;
            }
        }

        // Trump is random suit
        let trump = (seed % 4) as u8;
        // First player is random
        let first_player = ((seed / 4) % 4) as u8;

        let game = SuecaSimulatorGame::new(hands, trump, first_player);
        let sweep_weights = vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0];

        Ok(Self {
            game,
            lead_brain,
            follow_brain,
            rng,
            scratchpad,
            sweep_weights,
            last_trick: None,
            bot_types: [0; 4],
        })
    }

    pub fn set_bot_types(&mut self, bot1: u8, bot2: u8, bot3: u8) {
        self.bot_types[1] = bot1;
        self.bot_types[2] = bot2;
        self.bot_types[3] = bot3;
    }

    pub fn get_state_json(&self) -> String {
        let trump = self.game.state.trump;
        
        // Convert player's hand bitboard to vec of cards
        let mut player_hand = Vec::new();
        let mut temp = self.game.state.hands[0];
        while temp != 0 {
            player_hand.push(temp.trailing_zeros() as u8);
            temp &= temp - 1;
        }
        // Sort player's hand by suit and then rank for nice UI ordering
        player_hand.sort_by(|&a, &b| {
            let suit_a = CARD_SUIT[a as usize];
            let suit_b = CARD_SUIT[b as usize];
            if suit_a != suit_b {
                suit_a.cmp(&suit_b)
            } else {
                CARD_RANK[a as usize].cmp(&CARD_RANK[b as usize])
            }
        });

        // Compute legal moves for the active player (only populated if it's player 0's turn)
        let legal_moves = if self.game.state.current_player == 0 && !self.game.state.is_terminal() {
            let mut moves = Vec::new();
            let mut temp_l = self.game.state.legal_moves();
            while temp_l != 0 {
                moves.push(temp_l.trailing_zeros() as u8);
                temp_l &= temp_l - 1;
            }
            moves
        } else {
            Vec::new()
        };

        let mut other_hands_sizes = [0; 4];
        for i in 0..4 {
            other_hands_sizes[i] = self.game.state.hands[i].count_ones() as usize;
        }

        // Full hands shown at the end
        let mut other_hands = [const { Vec::new() }; 4];
        if self.game.state.is_terminal() {
            for i in 0..4 {
                let mut temp_h = self.game.state.hands[i];
                while temp_h != 0 {
                    other_hands[i].push(temp_h.trailing_zeros() as u8);
                    temp_h &= temp_h - 1;
                }
            }
        }

        let current_trick = self.game.current_trick[0..self.game.current_trick_len].to_vec();
        let current_trick_seats = self.game.current_trick_seats[0..self.game.current_trick_len].to_vec();
        
        let led_suit = if self.game.current_trick_len > 0 {
            CARD_SUIT[self.game.current_trick[0] as usize]
        } else {
            4
        };

        let is_over = self.game.state.is_terminal();
        
        let mut winner_team = None;
        let mut game_points_02 = 0;
        let mut game_points_13 = 0;
        if is_over {
            let s02 = self.game.state.team_02_score;
            let s13 = self.game.state.team_13_score;
            if s02 > s13 {
                winner_team = Some(0);
            } else if s13 > s02 {
                winner_team = Some(1);
            }
            
            let compute_gp = |pts: u8| -> u8 {
                if pts == 120 { 4 }
                else if pts >= 91 { 2 }
                else if pts >= 61 { 1 }
                else { 0 }
            };
            game_points_02 = compute_gp(s02);
            game_points_13 = compute_gp(s13);
        }

        let state = WasmGameState {
            trump,
            player_hand,
            legal_moves,
            other_hands_sizes,
            other_hands,
            current_trick,
            current_trick_seats,
            led_suit,
            current_player: self.game.state.current_player,
            team_02_score: self.game.state.team_02_score,
            team_13_score: self.game.state.team_13_score,
            trick_number: self.game.state.trick_number,
            voids: self.game.voids,
            is_over,
            winner_team,
            game_points_02,
            game_points_13,
            last_trick: self.last_trick.clone(),
        };

        serde_json::to_string(&state).unwrap_or_default()
    }

    pub fn play_player_card(&mut self, card: u8) -> Result<(), String> {
        if self.game.state.is_terminal() {
            return Err("Game is already over".to_string());
        }
        if self.game.state.current_player != 0 {
            return Err("It is not the player's turn".to_string());
        }

        // Verify card is in hand
        let hand_mask = self.game.state.hands[0];
        if (hand_mask & (1u64 << card)) == 0 {
            return Err("Card not in hand".to_string());
        }

        // Verify card is legal
        let legal_mask = self.game.state.legal_moves();
        if (legal_mask & (1u64 << card)) == 0 {
            return Err("Card is not follow-suit legal".to_string());
        }

        self.play_card_and_capture(card);
        Ok(())
    }

    pub fn play_bot_turn(&mut self) -> Result<u8, String> {
        if self.game.state.is_terminal() {
            return Err("Game is already over".to_string());
        }
        let seat = self.game.state.current_player;
        if seat == 0 {
            return Err("Waiting for player move".to_string());
        }

        let card = match self.bot_types[seat as usize] {
            1 => sueca_solver::heuristic::select_card_heuristic_old(&self.game, seat),
            2 => sueca_solver::heuristic::select_card_heuristic(&self.game, seat),
            _ => {
                // WANN Brain selection
                let belief = encode_belief_state(&self.game, seat);
                let network = if (belief[sueca_wann::constants::BeliefFeature::AmILeading as usize] - 1.0).abs() < 1e-9 {
                    &self.lead_brain
                } else {
                    &self.follow_brain
                };

                // Output averaging sweep
                let mut sum_outputs = [0.0f64; sueca_solver::constants::OUTPUT_COUNT];
                for &w in &self.sweep_weights {
                    network.forward(&belief, w, &mut self.scratchpad);
                    for i in 0..sueca_solver::constants::OUTPUT_COUNT {
                        sum_outputs[i] += self.scratchpad[sueca_solver::constants::OUTPUT_START + i];
                    }
                }

                let mut mean_outputs = [0.0f64; sueca_solver::constants::OUTPUT_COUNT];
                for i in 0..sueca_solver::constants::OUTPUT_COUNT {
                    mean_outputs[i] = sum_outputs[i] / (self.sweep_weights.len() as f64);
                }

                // Choose intent
                let mut max_val = mean_outputs[0];
                for i in 1..sueca_solver::constants::OUTPUT_COUNT {
                    if mean_outputs[i] > max_val {
                        max_val = mean_outputs[i];
                    }
                }

                let mut best_intents = [0usize; sueca_solver::constants::OUTPUT_COUNT];
                let mut best_count = 0;
                for i in 0..sueca_solver::constants::OUTPUT_COUNT {
                    if (mean_outputs[i] - max_val).abs() < 1e-9 {
                        best_intents[best_count] = i;
                        best_count += 1;
                    }
                }

                // Random tie break
                let chosen_intent = if best_count == 1 {
                    best_intents[0]
                } else {
                    // Pcg64 random range
                    use rand::Rng;
                    best_intents[self.rng.gen_range(0..best_count)]
                };

                resolve_intent(chosen_intent, &self.game, seat)
            }
        };
        
        self.play_card_and_capture(card);
        Ok(card)
    }

    fn play_card_and_capture(&mut self, card: u8) {
        // If trick is about to complete (it has 3 cards already)
        let is_completing = self.game.current_trick_len == 3;
        
        let mut played_cards = [0u8; 4];
        let mut played_seats = [0u8; 4];
        if is_completing {
            for i in 0..3 {
                played_cards[i] = self.game.current_trick[i];
                played_seats[i] = self.game.current_trick_seats[i];
            }
            played_cards[3] = card;
            played_seats[3] = self.game.state.current_player;
        }

        self.game.play_card(card);

        if is_completing {
            // Trick resolved. The next player is set to the winner.
            let winner = self.game.state.current_player; 
            let pts: u8 = played_cards.iter().map(|&c| CARD_POINTS[c as usize]).sum();
            
            self.last_trick = Some(WasmLastTrick {
                cards: played_cards,
                seats: played_seats,
                winner,
                points: pts,
            });
        }
    }

    pub fn is_game_over(&self) -> bool {
        self.game.state.is_terminal()
    }

    pub fn current_player(&self) -> u8 {
        self.game.state.current_player
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_wasm_game_flow() {
        let genome_path = "../../checkpoints/2026-06-03-2/genomes/best_genome_final.json";
        let genome_json = fs::read_to_string(genome_path)
            .expect("Failed to read test genome JSON");

        // Use a deterministic local LCG or seed-based RNG for the player's choices to keep tests reproducible
        for seed in 0..10000 {
            let mut session = WannSuecaGameSession::new(&genome_json, seed)
                .expect(&format!("Failed to create WannSuecaGameSession on seed {}", seed));

            // Use seed to make a simple local deterministic random generator for the player
            let mut player_rng = seed;

            while !session.is_game_over() {
                let current_player = session.current_player();
                if current_player == 0 {
                    let state_str = session.get_state_json();
                    let state: serde_json::Value = serde_json::from_str(&state_str).unwrap();
                    let legal_moves = state["legal_moves"].as_array().unwrap();
                    assert!(!legal_moves.is_empty(), "Player has no legal moves on seed {}", seed);
                    
                    // Choose a random legal card
                    player_rng = player_rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let card_idx = (player_rng as usize) % legal_moves.len();
                    let card = legal_moves[card_idx].as_u64().unwrap() as u8;
                    
                    session.play_player_card(card).unwrap();
                } else {
                    if let Err(e) = session.play_bot_turn() {
                        panic!("Bot failed on seed {} at player {}: {}", seed, current_player, e);
                    }
                }
            }

            let state_str = session.get_state_json();
            let state: serde_json::Value = serde_json::from_str(&state_str).unwrap();
            assert!(state["is_over"].as_bool().unwrap());
            assert_eq!(state["trick_number"].as_u64().unwrap(), 10);
            let score_02 = state["team_02_score"].as_u64().unwrap() as u8;
            let score_13 = state["team_13_score"].as_u64().unwrap() as u8;
            assert_eq!(score_02 + score_13, 120, "Total points must be 120 on seed {}", seed);
        }
    }
}
