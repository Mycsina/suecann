use crate::checkpoint::{TrainingState, BrainTrainingState};
use crate::config::Config;
use crate::dataset::{load_expert_dataset, ExpertDataset};
use crate::genome::{JsonGenome, JsonGenomeJoint, FIRST_HIDDEN_ID, INPUT_COUNT, OUTPUT_COUNT, OUTPUT_START};
use crate::hall_of_fame::{HallOfFame, JsonHallOfFame, JsonHallOfFameJoint};
use crate::mutations::{InnovationRegistry, TabuVetoList};
use crate::population::Population;
use crate::runtime_data;

use sueca_solver::constants::PHI_FEATURE_COUNT;
use sueca_solver::heuristic::{outputs_to_knobs, resolve_card_phi_utility_ctx};

use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

pub fn generate_deals_rust(
    gen: usize,
    n_deals: usize,
    base_seed: u64,
) -> Vec<crate::evaluator::EvaluatorDeal> {
    let seed = base_seed + gen as u64;
    let mut deals = Vec::new();

    for i in 0..n_deals {
        let deal_seed = seed * 1000 + i as u64;
        let mut rng = Pcg64::seed_from_u64(deal_seed);

        let mut deck: Vec<u8> = (0..40).collect();
        deck.shuffle(&mut rng);

        let mut hands = [0u64; 4];
        for player in 0..4 {
            for card_idx in 0..10 {
                let card = deck[player * 10 + card_idx];
                hands[player] |= 1u64 << card;
            }
        }

        let trump = rng.gen_range(0..4) as u8;
        deals.push(crate::evaluator::EvaluatorDeal {
            hands,
            trump,
            seed: deal_seed,
        });
    }

    deals
}

/// Generate a frozen set of deals for the fixed-yardstick Phase 1 probe.
/// These deals are never regenerated — they provide a stationary benchmark
/// to disentangle opponent-hardening from genuine capability regression.
fn generate_frozen_probe_deals(base_seed: u64) -> Vec<crate::evaluator::EvaluatorDeal> {
    let n_deals = 64;
    let mut deals = Vec::with_capacity(n_deals);
    for i in 0..n_deals {
        let deal_seed = base_seed.wrapping_mul(1000).wrapping_add(i as u64);
        let mut deal_rng = Pcg64::seed_from_u64(deal_seed);
        let mut deck: Vec<u8> = (0..40).collect();
        deck.shuffle(&mut deal_rng);
        let mut hands = [0u64; 4];
        for p in 0..4 {
            for c in 0..10 {
                hands[p] |= 1u64 << deck[p * 10 + c];
            }
        }
        let trump = deal_rng.gen_range(0..4) as u8;
        deals.push(crate::evaluator::EvaluatorDeal { hands, trump, seed: deal_seed });
    }
    deals
}

/// Evaluate a (lead, follow) pair against 4× OldHeuristicBot on frozen deals.
/// Returns mean point delta per game (Team 0-2 score minus baseline).
fn evaluate_fixed_probe(
    lead_brain: &crate::wann_network::RustWannNetwork,
    follow_brain: &crate::wann_network::RustWannNetwork,
    deals: &[crate::evaluator::EvaluatorDeal],
    sweep_weights: &[f64],
    base_seed: u64,
) -> f64 {
    let max_nodes = lead_brain.num_nodes.max(follow_brain.num_nodes);
    let mut total_delta = 0.0f64;
    let n_weights = sweep_weights.len();
    // Evaluate each deal × 4 rotations with the candidate pair at seat 0
    for (d_idx, deal) in deals.iter().enumerate() {
        for rot in 0..4 {
            let rotated = crate::evaluator::rotate_hands(&deal.hands, rot);
            let adj_first = (0u8 + rot as u8) % 4;
            let eval_seed = base_seed ^ ((d_idx as u64) << 32) ^ (rot as u64);

            let candidate_bot = crate::evaluator::SimulatorBot::Wann {
                lead_brain, follow_brain, weights: sweep_weights,
            };
            let baseline_bot = crate::evaluator::SimulatorBot::OldHeuristic;
            let bots = [candidate_bot.clone(), baseline_bot.clone(), baseline_bot.clone(), baseline_bot.clone()];

            let mut scratchpad = vec![0.0f64; max_nodes * n_weights];
            let mut behavior = crate::evaluator::WannBehavior::default();
            let result = crate::evaluator::play_game_sim(
                rotated, deal.trump, adj_first, &bots, eval_seed,
                &mut scratchpad, &mut behavior,
            );
            total_delta += result.team_02_score as f64 - 60.0; // baseline = 60 (expected score)
        }
    }
    total_delta / (deals.len() * 4) as f64
}

// ---------------------------------------------------------------------------
// Phase 0: Supervised card-match accuracy (Stage B fitness)
// ---------------------------------------------------------------------------
// For each state the WANN's 6 sweep-averaged outputs → signed knobs → the φ-
// utility resolver picks a card; fitness is the fraction of states where that
// card is in the teacher's best-cards mask. Decoupled from the action
// representation (works for any φ feature set) and directly optimizes the thing
// we care about (playing strong cards), per the design spec §5.
pub fn evaluate_phase0(
    genomes: &[crate::wann_network::RustWannNetwork],
    dataset: &ExpertDataset,
    sweep_weights: &[f64],
    _use_class_weighting: bool,
) -> Vec<f64> {
    let n_states = dataset.num_states;
    if n_states == 0 {
        return vec![0.0; genomes.len()];
    }

    // Pre-convert flat states into fixed-size arrays + cache PhiCtx per state
    // (one copy, reused across all genomes in the parallel loop).
    let states: Vec<[f64; INPUT_COUNT]> = dataset
        .states
        .chunks_exact(INPUT_COUNT)
        .map(|chunk| {
            let mut arr = [0.0f64; INPUT_COUNT];
            arr.copy_from_slice(chunk);
            arr
        })
        .collect();
    let ctxs: Vec<_> = (0..n_states).map(|i| dataset.ctx(i)).collect();
    let best_cards = &dataset.best_cards;

    genomes
        .into_par_iter()
        .map(|candidate| {
            let n_weights = sweep_weights.len();
            let mut scratchpad = vec![0.0f64; candidate.num_nodes * n_weights];
            let mut correct = 0.0f64;

            for (idx, inputs) in states.iter().enumerate() {
                candidate.forward_batched(inputs, sweep_weights, &mut scratchpad);

                // Average the OUTPUT_COUNT knobs across the weight sweep.
                let mut mean_outputs = [0.0f64; OUTPUT_COUNT];
                for i in 0..OUTPUT_COUNT {
                    let mut s = 0.0f64;
                    for w in 0..n_weights {
                        s += scratchpad[(OUTPUT_START + i) * n_weights + w];
                    }
                    mean_outputs[i] = s / (n_weights as f64);
                }

                let knobs = outputs_to_knobs(&mean_outputs);
                let card = resolve_card_phi_utility_ctx(&knobs, &ctxs[idx]);

                if (best_cards[idx] >> card) & 1 == 1 {
                    correct += 1.0;
                }
            }

            correct / (n_states as f64)
        })
        .collect()
}

// Silence unused-knob-count warning while keeping the dimension documented.
#[allow(dead_code)]
const _PHI_KNOB_COUNT_TRAIN: usize = PHI_FEATURE_COUNT;

// ---------------------------------------------------------------------------
// Phase 1: Self-play with HOF opponents
// ---------------------------------------------------------------------------
#[allow(dead_code, clippy::too_many_arguments)]
pub fn evaluate_phase1(
    genomes: &[crate::wann_network::RustWannNetwork],
    is_lead: bool,
    deals: &[crate::evaluator::EvaluatorDeal],
    reference_brain: &crate::wann_network::RustWannNetwork,
    hof_lead_networks: &[crate::wann_network::RustWannNetwork],
    hof_follow_networks: &[crate::wann_network::RustWannNetwork],
    partner_bot_type: i32,
    opp1_bot_type: i32,
    opp2_bot_type: i32,
    baseline_bot_type: i32,
    sweep_weights: &[f64],
    base_seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<crate::evaluator::WannBehavior>) {
    let max_nodes = genomes
        .iter()
        .map(|g| g.num_nodes)
        .chain(std::iter::once(reference_brain.num_nodes))
        .chain(hof_lead_networks.iter().map(|g| g.num_nodes))
        .chain(hof_follow_networks.iter().map(|g| g.num_nodes))
        .max()
        .unwrap_or(FIRST_HIDDEN_ID);

    let baseline_scores: Vec<f64> = (0..deals.len() * 4)
        .into_par_iter()
        .map(|idx| {
            let deal_idx = idx / 4;
            let rot = idx % 4;
            let deal = &deals[deal_idx];

            let partner = crate::evaluator::get_bot_from_type(partner_bot_type, hof_lead_networks, hof_follow_networks, sweep_weights);
            let opp1 = crate::evaluator::get_bot_from_type(opp1_bot_type, hof_lead_networks, hof_follow_networks, sweep_weights);
            let opp2 = crate::evaluator::get_bot_from_type(opp2_bot_type, hof_lead_networks, hof_follow_networks, sweep_weights);
            let baseline = crate::evaluator::get_bot_from_type(baseline_bot_type, hof_lead_networks, hof_follow_networks, sweep_weights);

            let rotated_hands = crate::evaluator::rotate_hands(&deal.hands, rot);
            let adj_first = (0 + rot as u8) % 4;
            let evaluation_seed = base_seed ^ ((deal_idx as u64) << 32) ^ (rot as u64);

            let bots_baseline = [
                baseline,
                opp1,
                partner,
                opp2,
            ];

            let mut dummy_behavior = crate::evaluator::WannBehavior::default();
            let mut scratchpad = vec![0.0f64; max_nodes * sweep_weights.len()];
            let result_baseline = crate::evaluator::play_game_sim(
                rotated_hands,
                deal.trump,
                adj_first,
                &bots_baseline,
                evaluation_seed,
                &mut scratchpad,
                &mut dummy_behavior,
            );
            result_baseline.team_02_score as f64
        })
        .collect();

    let results: Vec<(f64, crate::evaluator::WannBehavior)> = genomes
        .into_par_iter()
        .map(|candidate| {
            let mut scratchpad = vec![0.0f64; max_nodes * sweep_weights.len()];
            let (candidate_lead, candidate_follow) = if is_lead {
                (candidate, reference_brain)
            } else {
                (reference_brain, candidate)
            };
            crate::evaluator::evaluate_genome_delta(
                candidate_lead,
                candidate_follow,
                partner_bot_type,
                opp1_bot_type,
                opp2_bot_type,
                hof_lead_networks,
                hof_follow_networks,
                sweep_weights,
                deals,
                base_seed,
                &baseline_scores,
                &mut scratchpad,
            )
        })
        .collect();

    let mut fitnesses = Vec::with_capacity(genomes.len());
    let mut deltas = Vec::with_capacity(genomes.len());
    let mut behaviors = Vec::with_capacity(genomes.len());

    for (delta, behavior) in results {
        fitnesses.push(delta);
        deltas.push(delta);
        behaviors.push(behavior);
    }

    (fitnesses, deltas, behaviors)
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_phase1_joint(
    lead_genomes: &[crate::wann_network::RustWannNetwork],
    follow_genomes: &[crate::wann_network::RustWannNetwork],
    deals: &[crate::evaluator::EvaluatorDeal],
    hof_lead_networks: &[crate::wann_network::RustWannNetwork],
    hof_follow_networks: &[crate::wann_network::RustWannNetwork],
    partner_bot_type: i32,
    opp1_bot_type: i32,
    opp2_bot_type: i32,
    baseline_bot_type: i32,
    sweep_weights: &[f64],
    base_seed: u64,
) -> (Vec<f64>, Vec<crate::evaluator::WannBehavior>) {
    let max_nodes = lead_genomes
        .iter()
        .map(|g| g.num_nodes)
        .chain(follow_genomes.iter().map(|g| g.num_nodes))
        .chain(hof_lead_networks.iter().map(|g| g.num_nodes))
        .chain(hof_follow_networks.iter().map(|g| g.num_nodes))
        .max()
        .unwrap_or(FIRST_HIDDEN_ID);

    let baseline_scores: Vec<f64> = (0..deals.len() * 4)
        .into_par_iter()
        .map(|idx| {
            let deal_idx = idx / 4;
            let rot = idx % 4;
            let deal = &deals[deal_idx];

            let partner = crate::evaluator::get_bot_from_type(partner_bot_type, hof_lead_networks, hof_follow_networks, sweep_weights);
            let opp1 = crate::evaluator::get_bot_from_type(opp1_bot_type, hof_lead_networks, hof_follow_networks, sweep_weights);
            let opp2 = crate::evaluator::get_bot_from_type(opp2_bot_type, hof_lead_networks, hof_follow_networks, sweep_weights);
            let baseline = crate::evaluator::get_bot_from_type(baseline_bot_type, hof_lead_networks, hof_follow_networks, sweep_weights);

            let rotated_hands = crate::evaluator::rotate_hands(&deal.hands, rot);
            let adj_first = (0 + rot as u8) % 4;
            let evaluation_seed = base_seed ^ ((deal_idx as u64) << 32) ^ (rot as u64);

            let bots_baseline = [
                baseline,
                opp1,
                partner,
                opp2,
            ];

            let mut dummy_behavior = crate::evaluator::WannBehavior::default();
            let mut scratchpad = vec![0.0f64; max_nodes * sweep_weights.len()];
            let result_baseline = crate::evaluator::play_game_sim(
                rotated_hands,
                deal.trump,
                adj_first,
                &bots_baseline,
                evaluation_seed,
                &mut scratchpad,
                &mut dummy_behavior,
            );
            result_baseline.team_02_score as f64
        })
        .collect();

    let population_size = lead_genomes.len();
    assert_eq!(population_size, follow_genomes.len());

    let results: Vec<(f64, crate::evaluator::WannBehavior)> = (0..population_size)
        .into_par_iter()
        .map(|idx| {
            let mut scratchpad = vec![0.0f64; max_nodes * sweep_weights.len()];
            let candidate_lead = &lead_genomes[idx];
            let candidate_follow = &follow_genomes[idx];
            crate::evaluator::evaluate_genome_delta(
                candidate_lead,
                candidate_follow,
                partner_bot_type,
                opp1_bot_type,
                opp2_bot_type,
                hof_lead_networks,
                hof_follow_networks,
                sweep_weights,
                deals,
                base_seed,
                &baseline_scores,
                &mut scratchpad,
            )
        })
        .collect();

    let mut fitnesses = Vec::with_capacity(population_size);
    let mut behaviors = Vec::with_capacity(population_size);

    for (delta, behavior) in results {
        fitnesses.push(delta);
        behaviors.push(behavior);
    }

    (fitnesses, behaviors)
}


// ---------------------------------------------------------------------------
// Per-generation evaluation dispatchers
// ---------------------------------------------------------------------------
#[allow(dead_code)]
fn run_phase1_generation(
    genomes: &[crate::wann_network::RustWannNetwork],
    is_lead: bool,
    hof_lead: &HallOfFame,
    hof_follow: &HallOfFame,
    map_elites_lead: &crate::map_elites::MapElitesArchive,
    map_elites_follow: &crate::map_elites::MapElitesArchive,
    reference_brain: &crate::wann_network::RustWannNetwork,
    config: &Config,
    gen: usize,
    rng: &mut Pcg64,
) -> (Vec<f64>, Vec<f64>, Vec<crate::evaluator::WannBehavior>) {
    let deals = generate_deals_rust(
        gen,
        config.evaluation.n_deals,
        config.evaluation.seed * 1000,
    );

    let mut hof_lead_nets = Vec::new();
    let mut hof_follow_nets = Vec::new();

    let sample_seat_bot = |rng: &mut Pcg64,
                           h_lead: &HallOfFame,
                           h_follow: &HallOfFame,
                           me_lead: &crate::map_elites::MapElitesArchive,
                           me_follow: &crate::map_elites::MapElitesArchive,
                           hg_lead: &mut Vec<crate::wann_network::RustWannNetwork>,
                           hg_follow: &mut Vec<crate::wann_network::RustWannNetwork>|
     -> i32 {
        let elite_prob = ((gen as f64 - 200.0) / 400.0).clamp(0.0, 0.40);

        if rng.gen_bool(elite_prob) {
            2 // EliteHeuristicBot
        } else if rng.gen_bool(0.5) {
            let use_map_elites = rng.gen_bool(0.5);
            let lead_genome = if use_map_elites {
                me_lead.sample_random(rng)
            } else {
                None
            }.or_else(|| h_lead.sample(rng, 1).first().cloned());

            let follow_genome = if use_map_elites {
                me_follow.sample_random(rng)
            } else {
                None
            }.or_else(|| h_follow.sample(rng, 1).first().cloned());

            if let (Some(g_lead), Some(g_follow)) = (lead_genome, follow_genome) {
                let bot_type = 10 + hg_lead.len() as i32; // WANNs start at 10
                hg_lead.push(g_lead.to_rust_wann());
                hg_follow.push(g_follow.to_rust_wann());
                bot_type
            } else {
                1 // OldHeuristicBot
            }
        } else {
            1 // Old HeuristicBot (Baseline Sanity)
        }
    };

    let mut sample_rng = rng.clone();
    let partner_bot_type: i32 = sample_seat_bot(
        &mut sample_rng,
        hof_lead,
        hof_follow,
        map_elites_lead,
        map_elites_follow,
        &mut hof_lead_nets,
        &mut hof_follow_nets,
    );
    let opp1_bot_type: i32 = sample_seat_bot(
        &mut sample_rng,
        hof_lead,
        hof_follow,
        map_elites_lead,
        map_elites_follow,
        &mut hof_lead_nets,
        &mut hof_follow_nets,
    );
    let opp2_bot_type: i32 = sample_seat_bot(
        &mut sample_rng,
        hof_lead,
        hof_follow,
        map_elites_lead,
        map_elites_follow,
        &mut hof_lead_nets,
        &mut hof_follow_nets,
    );
    *rng = sample_rng;

    evaluate_phase1(
        genomes,
        is_lead,
        &deals,
        reference_brain,
        &hof_lead_nets,
        &hof_follow_nets,
        partner_bot_type,
        opp1_bot_type,
        opp2_bot_type,
        2, // baseline = HeuristicBot
        &config.evaluation.sweep_weights,
        config.evaluation.seed + gen as u64 * 1000,
    )
}

fn run_phase1_generation_joint(
    lead_genomes: &[crate::wann_network::RustWannNetwork],
    follow_genomes: &[crate::wann_network::RustWannNetwork],
    hof_lead: &HallOfFame,
    hof_follow: &HallOfFame,
    map_elites_lead: &crate::map_elites::MapElitesArchive,
    map_elites_follow: &crate::map_elites::MapElitesArchive,
    config: &Config,
    gen: usize,
    rng: &mut Pcg64,
) -> (Vec<f64>, Vec<crate::evaluator::WannBehavior>) {
    let deals = generate_deals_rust(
        gen,
        config.evaluation.n_deals,
        config.evaluation.seed * 1000,
    );

    let mut hof_lead_nets = Vec::new();
    let mut hof_follow_nets = Vec::new();

    let sample_seat_bot = |rng: &mut Pcg64,
                           h_lead: &HallOfFame,
                           h_follow: &HallOfFame,
                           me_lead: &crate::map_elites::MapElitesArchive,
                           me_follow: &crate::map_elites::MapElitesArchive,
                           hg_lead: &mut Vec<crate::wann_network::RustWannNetwork>,
                           hg_follow: &mut Vec<crate::wann_network::RustWannNetwork>|
      -> i32 {
        let elite_prob = ((gen as f64 - 200.0) / 400.0).clamp(0.0, 0.40);

        if rng.gen_bool(elite_prob) {
            2 // EliteHeuristicBot
        } else if rng.gen_bool(0.5) {
            let use_map_elites = rng.gen_bool(0.5);
            let lead_genome = if use_map_elites {
                me_lead.sample_random(rng)
            } else {
                None
            }.or_else(|| h_lead.sample(rng, 1).first().cloned());

            let follow_genome = if use_map_elites {
                me_follow.sample_random(rng)
            } else {
                None
            }.or_else(|| h_follow.sample(rng, 1).first().cloned());

            if let (Some(g_lead), Some(g_follow)) = (lead_genome, follow_genome) {
                let bot_type = 10 + hg_lead.len() as i32; // WANNs start at 10
                hg_lead.push(g_lead.to_rust_wann());
                hg_follow.push(g_follow.to_rust_wann());
                bot_type
            } else {
                1 // OldHeuristicBot
            }
        } else {
            1 // Old HeuristicBot (Baseline Sanity)
        }
    };

    let mut sample_rng = rng.clone();
    let partner_bot_type: i32 = sample_seat_bot(
        &mut sample_rng,
        hof_lead,
        hof_follow,
        map_elites_lead,
        map_elites_follow,
        &mut hof_lead_nets,
        &mut hof_follow_nets,
    );
    let opp1_bot_type: i32 = sample_seat_bot(
        &mut sample_rng,
        hof_lead,
        hof_follow,
        map_elites_lead,
        map_elites_follow,
        &mut hof_lead_nets,
        &mut hof_follow_nets,
    );
    let opp2_bot_type: i32 = sample_seat_bot(
        &mut sample_rng,
        hof_lead,
        hof_follow,
        map_elites_lead,
        map_elites_follow,
        &mut hof_lead_nets,
        &mut hof_follow_nets,
    );
    *rng = sample_rng;

    evaluate_phase1_joint(
        lead_genomes,
        follow_genomes,
        &deals,
        &hof_lead_nets,
        &hof_follow_nets,
        partner_bot_type,
        opp1_bot_type,
        opp2_bot_type,
        2, // baseline = HeuristicBot
        &config.evaluation.sweep_weights,
        config.evaluation.seed + gen as u64 * 1000,
    )
}


fn split_dataset(dataset: &ExpertDataset, holdout_frac: f64)
    -> (ExpertDataset, ExpertDataset, ExpertDataset, ExpertDataset)
{
    // Split by lead/follow (AmILeading flag), then per-split random holdout.
    // No intent stratification under card-match fitness — the best-cards mask
    // already encodes the target and per-intent balancing is obsolete.
    use crate::constants::BeliefFeature;
    let lead = BeliefFeature::AmILeading as usize;

    let mut lead_idxs: Vec<usize> = Vec::new();
    let mut follow_idxs: Vec<usize> = Vec::new();
    for idx in 0..dataset.num_states {
        let is_lead = (dataset.states[idx * INPUT_COUNT + lead] - 1.0).abs() < 1e-9;
        if is_lead { lead_idxs.push(idx); } else { follow_idxs.push(idx); }
    }

    // Deterministic split: first `n_val` indices → validation, rest → train.
    let split_one = |idxs: &[usize], train: &mut AccDataset, val: &mut AccDataset| {
        let n_val = ((idxs.len() as f64) * holdout_frac).ceil() as usize;
        let (val_idxs, train_idxs) = idxs.split_at(n_val.min(idxs.len()));
        for &i in train_idxs { train.push_from(dataset, i); }
        for &i in val_idxs { val.push_from(dataset, i); }
    };

    let mut lead_train = AccDataset::new();
    let mut lead_val = AccDataset::new();
    let mut follow_train = AccDataset::new();
    let mut follow_val = AccDataset::new();
    split_one(&lead_idxs, &mut lead_train, &mut lead_val);
    split_one(&follow_idxs, &mut follow_train, &mut follow_val);

    (
        lead_train.into_dataset(),
        lead_val.into_dataset(),
        follow_train.into_dataset(),
        follow_val.into_dataset(),
    )
}

/// Accumulator for building a sub-dataset from source indices.
struct AccDataset {
    states: Vec<f64>,
    best_cards: Vec<u64>,
    ctx_trump: Vec<u8>,
    ctx_hand: Vec<u64>,
    ctx_trick: Vec<u8>,
    ctx_trick_len: Vec<u8>,
}
impl AccDataset {
    fn new() -> Self {
        Self { states: Vec::new(), best_cards: Vec::new(), ctx_trump: Vec::new(),
               ctx_hand: Vec::new(), ctx_trick: Vec::new(), ctx_trick_len: Vec::new() }
    }
    fn push_from(&mut self, src: &ExpertDataset, i: usize) {
        let off = i * INPUT_COUNT;
        self.states.extend_from_slice(&src.states[off..off + INPUT_COUNT]);
        self.best_cards.push(src.best_cards[i]);
        self.ctx_trump.push(src.ctx_trump[i]);
        self.ctx_hand.push(src.ctx_hand[i]);
        let base = i * 4;
        self.ctx_trick.extend_from_slice(&src.ctx_trick[base..base + 4]);
        self.ctx_trick_len.push(src.ctx_trick_len[i]);
    }
    fn into_dataset(self) -> ExpertDataset {
        let num_states = self.best_cards.len();
        ExpertDataset {
            states: self.states,
            num_states,
            best_cards: self.best_cards,
            ctx_trump: self.ctx_trump,
            ctx_hand: self.ctx_hand,
            ctx_trick: self.ctx_trick,
            ctx_trick_len: self.ctx_trick_len,
        }
    }
}

// ---------------------------------------------------------------------------
// Main Training Loop
// ---------------------------------------------------------------------------
pub fn train(config: Config, resume: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Pcg64::seed_from_u64(config.evaluation.seed);

    // Determine run directory: if resuming, use existing dir; otherwise create dated folder
    let run_dir = if resume {
        let base = std::path::Path::new(&config.output.checkpoint_dir);
        let state_path = base.join("training_state.bin");
        if state_path.exists() {
            config.output.checkpoint_dir.clone()
        } else {
            // Auto-detect latest dated subdirectory
            let mut dated_dirs: Vec<_> = std::fs::read_dir(base)
                .ok()
                .into_iter()
                .flatten()
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .filter(|e| e.path().join("training_state.bin").exists())
                .collect();
            dated_dirs.sort_by_key(|e| e.file_name());
            if let Some(latest) = dated_dirs.last() {
                println!("Resume: auto-detected run directory {}", latest.path().display());
                latest.path().to_string_lossy().to_string()
            } else {
                eprintln!("No checkpoint found in {}. Starting fresh.", base.display());
                config.output.checkpoint_dir.clone()
            }
        }
    } else {
        let today = chrono::Local::now().format("%Y-%m-%d").to_string();
        let base = Path::new(&config.output.checkpoint_dir);
        let mut n = 1;
        let dir = loop {
            let candidate = base.join(format!("{}-{}", today, n));
            if !candidate.exists() {
                break candidate;
            }
            n += 1;
        };
        dir.to_string_lossy().to_string()
    };
    let mut config = config;
    config.output.checkpoint_dir = run_dir.clone();
    config.output.stats_file = format!("{}/training_stats.csv", run_dir);

    fs::create_dir_all(&run_dir)?;
    fs::create_dir_all(Path::new(&run_dir).join("genomes"))?;
    println!("Run directory: {}", run_dir);

    let state_checkpoint_path = Path::new(&run_dir).join("training_state.bin");

    let mut lead_pop: Population;
    let mut lead_hof: HallOfFame;
    let mut lead_map_elites: crate::map_elites::MapElitesArchive;
    let mut lead_registry: Mutex<InnovationRegistry>;
    let lead_tabu = TabuVetoList::new(1000);

    let mut follow_pop: Population;
    let mut follow_hof: HallOfFame;
    let mut follow_map_elites: crate::map_elites::MapElitesArchive;
    let mut follow_registry: Mutex<InnovationRegistry>;
    let follow_tabu = TabuVetoList::new(1000);

    let mut start_gen = 0;
    let mut current_phase = 0;

    if resume && state_checkpoint_path.exists() {
        println!(
            "Resuming training from checkpoint {:?}",
            state_checkpoint_path
        );
        let state = TrainingState::load_from_file(&state_checkpoint_path)?;
        start_gen = state.generation;
        current_phase = state.current_phase;

        let lead_state = state.lead;
        lead_pop = Population {
            config: config.clone(),
            genomes: lead_state.genomes,
            fitnesses: vec![0.0; config.population.pop_size],
            species_list: lead_state.species,
            generation: state.generation,
            next_species_id: lead_state.next_species_id,
            global_best_fitness: lead_state.global_best_fitness,
            global_best_genome: lead_state.global_best_genome,
            generations_since_improvement: lead_state.generations_since_improvement,
        };
        lead_hof = lead_state.hof;
        lead_map_elites = lead_state.map_elites;
        lead_registry = Mutex::new(InnovationRegistry::new(lead_state.next_innovation));

        let follow_state = state.follow;
        follow_pop = Population {
            config: config.clone(),
            genomes: follow_state.genomes,
            fitnesses: vec![0.0; config.population.pop_size],
            species_list: follow_state.species,
            generation: state.generation,
            next_species_id: follow_state.next_species_id,
            global_best_fitness: follow_state.global_best_fitness,
            global_best_genome: follow_state.global_best_genome,
            generations_since_improvement: follow_state.generations_since_improvement,
        };
        follow_hof = follow_state.hof;
        follow_map_elites = follow_state.map_elites;
        follow_registry = Mutex::new(InnovationRegistry::new(follow_state.next_innovation));
    } else {
        lead_registry = Mutex::new(InnovationRegistry::new(0));
        lead_pop = Population::new(config.clone(), &mut rng, &lead_registry);
        lead_hof = HallOfFame::new(config.hall_of_fame.hof_size);
        lead_map_elites = crate::map_elites::MapElitesArchive::new();

        follow_registry = Mutex::new(InnovationRegistry::new(0));
        follow_pop = Population::new(config.clone(), &mut rng, &follow_registry);
        follow_hof = HallOfFame::new(config.hall_of_fame.hof_size);
        follow_map_elites = crate::map_elites::MapElitesArchive::new();
    }

    // Restore runtime data if available (tabu lists, innovation registry mappings)
    let data_dir = Path::new(&run_dir).join("data");
    if let Ok(runtime) = runtime_data::load_runtime_data(&data_dir) {
        if let Some(ref pairs) = runtime.lead_tabu {
            lead_tabu.load_pairs(pairs);
            println!(
                "  >>> Restored lead tabu list: {} vetoed connections",
                pairs.len()
            );
        }
        if let Some(ref pairs) = runtime.follow_tabu {
            follow_tabu.load_pairs(pairs);
            println!(
                "  >>> Restored follow tabu list: {} vetoed connections",
                pairs.len()
            );
        }
        if let Some(state) = runtime.lead_innov {
            let n_pairs = state.pairs.len();
            lead_registry = Mutex::new(InnovationRegistry::from_state(state));
            println!(
                "  >>> Restored lead innovation registry: {} mappings",
                n_pairs
            );
        }
        if let Some(state) = runtime.follow_innov {
            let n_pairs = state.pairs.len();
            follow_registry = Mutex::new(InnovationRegistry::from_state(state));
            println!(
                "  >>> Restored follow innovation registry: {} mappings",
                n_pairs
            );
        }
    }

    // Load dataset for Phase 0
    let dataset = load_expert_dataset(&config.curriculum.phase0_dataset)?;
    let (lead_dataset, lead_val_dataset, follow_dataset, follow_val_dataset) = split_dataset(&dataset, 0.10);
    println!(
        "  >>> Split dataset: {} Lead ({} train / {} val), {} Follow ({} train / {} val).",
        lead_dataset.num_states + lead_val_dataset.num_states,
        lead_dataset.num_states, lead_val_dataset.num_states,
        follow_dataset.num_states + follow_val_dataset.num_states,
        follow_dataset.num_states, follow_val_dataset.num_states
    );

    // CSV Stats
    let stats_path = Path::new(&config.output.stats_file);
    let append = resume && stats_path.exists();
    let mut csv_file = OpenOptions::new()
        .create(true)
        .append(append)
        .write(true)
        .truncate(!append)
        .open(stats_path)?;

    if !resume || csv_file.metadata()?.len() == 0 {
        writeln!(
            csv_file,
            "generation,phase,lead_best_fitness,lead_avg_fitness,follow_best_fitness,follow_avg_fitness,lead_n_species,follow_n_species,lead_n_connections_best,follow_n_connections_best,lead_val_acc,follow_val_acc,fixed_probe_delta,elapsed_sec"
        )?;
    }

    println!(
        "{:>4} {:>2} {:>8} {:>8} {:>8} {:>8} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "Gen", "Ph", "L-Best", "L-Avg", "F-Best", "F-Avg",
        "L-Spec", "F-Spec", "L-Conn", "F-Conn",
        "L-Val", "F-Val", "Prb", "Time"
    );
    println!("{}", "-".repeat(120));

    // Fixed-yardstick probe: frozen deal set for Phase 1 stationary benchmarking.
    // Generated once at Phase transition; same deals used for all probe evaluations.
    let mut frozen_probe_deals: Option<Vec<crate::evaluator::EvaluatorDeal>> = None;

    for gen in start_gen..config.population.generations {
        let t_gen = Instant::now();

        // ── Phase selection ─────────────────────────────────────────────
        let t_phase_start = Instant::now();
        let new_phase = if gen < config.curriculum.phase0_gens {
            0
        } else {
            1
        };
        if new_phase != current_phase {
            println!(
                "  >>> Phase transition: {} -> {} at gen {}",
                current_phase, new_phase, gen
            );
            current_phase = new_phase;
            if current_phase == 1 {
                // 1. Sort and align populations by Phase 0 fitness
                let mut lead_indices: Vec<usize> = (0..lead_pop.genomes.len()).collect();
                lead_indices.sort_by(|&a, &b| lead_pop.fitnesses[b].partial_cmp(&lead_pop.fitnesses[a]).unwrap());

                let mut follow_indices: Vec<usize> = (0..follow_pop.genomes.len()).collect();
                follow_indices.sort_by(|&a, &b| follow_pop.fitnesses[b].partial_cmp(&follow_pop.fitnesses[a]).unwrap());

                let sorted_lead: Vec<crate::genome::Genome> = lead_indices.iter().map(|&idx| lead_pop.genomes[idx].copy()).collect();
                let sorted_follow: Vec<crate::genome::Genome> = follow_indices.iter().map(|&idx| follow_pop.genomes[idx].copy()).collect();

                lead_pop.genomes = sorted_lead;
                follow_pop.genomes = sorted_follow;

                // Reset fitnesses to 0.0
                lead_pop.fitnesses = vec![0.0; lead_pop.genomes.len()];
                follow_pop.fitnesses = vec![0.0; follow_pop.genomes.len()];

                // 2. Pre-populate HOF using top unique Phase 0 genomes.
                //    Uses innovation fingerprint hashing (O(pop·E) total) instead
                //    of pairwise compatibility_distance (O(pop²·E)) for uniqueness.
                println!("  >>> Pre-populating Phase 1 HOF with top unique Phase 0 genomes...");
                lead_hof.clear();
                follow_hof.clear();

                let mut unique_lead = Vec::new();
                let mut seen_fps: HashSet<u64> = HashSet::new();
                for idx in 0..lead_pop.genomes.len() {
                    let fp = lead_pop.genomes[idx].innovation_fingerprint();
                    if seen_fps.insert(fp) {
                        unique_lead.push(lead_pop.genomes[idx].copy());
                        if unique_lead.len() >= 5 {
                            break;
                        }
                    }
                }
                // If we didn't get 5 unique from the top, continue scanning
                if unique_lead.len() < 5 {
                    for idx in 0..lead_pop.genomes.len() {
                        let fp = lead_pop.genomes[idx].innovation_fingerprint();
                        if seen_fps.insert(fp) {
                            unique_lead.push(lead_pop.genomes[idx].copy());
                            if unique_lead.len() >= 5 {
                                break;
                            }
                        }
                    }
                }

                let mut unique_follow = Vec::new();
                seen_fps.clear();
                for idx in 0..follow_pop.genomes.len() {
                    let fp = follow_pop.genomes[idx].innovation_fingerprint();
                    if seen_fps.insert(fp) {
                        unique_follow.push(follow_pop.genomes[idx].copy());
                        if unique_follow.len() >= 5 {
                            break;
                        }
                    }
                }
                if unique_follow.len() < 5 {
                    for idx in 0..follow_pop.genomes.len() {
                        let fp = follow_pop.genomes[idx].innovation_fingerprint();
                        if seen_fps.insert(fp) {
                            unique_follow.push(follow_pop.genomes[idx].copy());
                            if unique_follow.len() >= 5 {
                                break;
                            }
                        }
                    }
                }

                // Evaluate pre-populated Lead and Follow genomes jointly
                let deals = generate_deals_rust(
                    gen,
                    config.evaluation.n_deals,
                    config.evaluation.seed * 1000,
                );

                let min_len = unique_lead.len().min(unique_follow.len());
                let lead_networks: Vec<crate::wann_network::RustWannNetwork> =
                    unique_lead[..min_len].iter().map(|g| g.to_rust_wann()).collect();
                let follow_networks: Vec<crate::wann_network::RustWannNetwork> =
                    unique_follow[..min_len].iter().map(|g| g.to_rust_wann()).collect();

                let (joint_fitnesses, _) = evaluate_phase1_joint(
                    &lead_networks,
                    &follow_networks,
                    &deals,
                    &[],
                    &[],
                    2,   // partner = HeuristicBot
                    2,   // opp1 = HeuristicBot
                    2,   // opp2 = HeuristicBot
                    2,   // baseline = HeuristicBot
                    &config.evaluation.sweep_weights,
                    config.evaluation.seed + gen as u64 * 1000,
                );

                for idx in 0..min_len {
                    lead_hof.add(&unique_lead[idx], joint_fitnesses[idx], gen);
                    follow_hof.add(&unique_follow[idx], joint_fitnesses[idx], gen);
                }

                println!(
                    "  >>> Pre-populated HOF with {} Lead and {} Follow unique genomes from Phase 0",
                    lead_hof.entries.len(),
                    follow_hof.entries.len()
                );

                lead_pop.global_best_fitness = f64::NEG_INFINITY;
                lead_pop.global_best_genome = None;
                lead_pop.generations_since_improvement = 0;

                follow_pop.global_best_fitness = f64::NEG_INFINITY;
                follow_pop.global_best_genome = None;
                follow_pop.generations_since_improvement = 0;

                // Generate frozen probe deals once for the entire Phase 1
                frozen_probe_deals = Some(generate_frozen_probe_deals(config.evaluation.seed + 9999));
            }
        }
        let t_phase = t_phase_start.elapsed();

        // ── Genome → Network conversion (parallel) ──────────────────────
        let t_convert_start = Instant::now();
        let lead_rust_genomes: Vec<crate::wann_network::RustWannNetwork> =
            lead_pop.genomes.par_iter().map(|g| g.to_rust_wann()).collect();
        let follow_rust_genomes: Vec<crate::wann_network::RustWannNetwork> =
            follow_pop.genomes.par_iter().map(|g| g.to_rust_wann()).collect();
        let t_convert = t_convert_start.elapsed();

        // ── Fitness evaluation ──────────────────────────────────────────
        let t_eval_start = Instant::now();
        let mut lead_val_acc = 0.0f64;
        let mut follow_val_acc = 0.0f64;
        let (lead_fitnesses, follow_fitnesses, lead_behaviors, follow_behaviors) = if current_phase == 0 {
            let lead_accs = evaluate_phase0(&lead_rust_genomes, &lead_dataset, &config.evaluation.sweep_weights, config.curriculum.use_class_weighting);
            let follow_accs = evaluate_phase0(&follow_rust_genomes, &follow_dataset, &config.evaluation.sweep_weights, config.curriculum.use_class_weighting);
            // Validation accuracy on holdout (best genome only)
            lead_val_acc = lead_accs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            follow_val_acc = follow_accs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if lead_val_dataset.num_states > 0 {
                let lead_val_genomes: Vec<crate::wann_network::RustWannNetwork> = vec![lead_pop.genomes[lead_accs.iter().enumerate().fold((0, f64::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0].to_rust_wann()];
                lead_val_acc = evaluate_phase0(&lead_val_genomes, &lead_val_dataset, &config.evaluation.sweep_weights, false)[0];
            }
            if follow_val_dataset.num_states > 0 {
                let follow_val_genomes: Vec<crate::wann_network::RustWannNetwork> = vec![follow_pop.genomes[follow_accs.iter().enumerate().fold((0, f64::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0].to_rust_wann()];
                follow_val_acc = evaluate_phase0(&follow_val_genomes, &follow_val_dataset, &config.evaluation.sweep_weights, false)[0];
            }
            (lead_accs.clone(), follow_accs.clone(), Vec::new(), Vec::new())
        } else {
            let (fitnesses, behaviors) = run_phase1_generation_joint(
                &lead_rust_genomes,
                &follow_rust_genomes,
                &lead_hof,
                &follow_hof,
                &lead_map_elites,
                &follow_map_elites,
                &config,
                gen,
                &mut rng,
            );
            (fitnesses.clone(), fitnesses, behaviors.clone(), behaviors)
        };
        let t_eval = t_eval_start.elapsed();

        // ── Stats: tell_fitnesses, MAP-Elites, best candidate, HOF ─────
        let t_stats_start = Instant::now();
        lead_pop.tell_fitnesses(&lead_fitnesses);
        follow_pop.tell_fitnesses(&follow_fitnesses);

        // Add each genome's behavior to MAP-Elites if in Phase 1
        if current_phase == 1 {
            // φ-utility behavioral descriptor: mean "win-preference" knob
            // (PhiFeature::Wins), remapped from [-1,1] → [0,1]. Replaces the old
            // 3-intent `intent_pref` (fraction of EFFICIENT_WIN plays).
            const WINS: usize = sueca_solver::constants::PhiFeature::Wins as usize;
            for (i, behavior) in lead_behaviors.iter().enumerate() {
                let win_pref = ((behavior.knob_sums[WINS] / (behavior.total_actions.max(1) as f64))
                    + 1.0)
                    / 2.0;
                let win_pref = win_pref.clamp(0.0, 1.0);
                let aggression = ((behavior.total_lead_points as f64)
                    / (behavior.count_leads.max(1) as f64 * 10.0))
                    .clamp(0.0, 1.0);
                lead_map_elites.add(&lead_pop.genomes[i], lead_fitnesses[i], gen, win_pref, aggression);
            }
            for (i, behavior) in follow_behaviors.iter().enumerate() {
                let win_pref = ((behavior.knob_sums[WINS] / (behavior.total_actions.max(1) as f64))
                    + 1.0)
                    / 2.0;
                let win_pref = win_pref.clamp(0.0, 1.0);
                let aggression = ((behavior.total_lead_points as f64)
                    / (behavior.count_leads.max(1) as f64 * 10.0))
                    .clamp(0.0, 1.0);
                follow_map_elites.add(&follow_pop.genomes[i], follow_fitnesses[i], gen, win_pref, aggression);
            }
        }

        // Find best candidate
        let mut lead_best_idx = 0;
        let mut lead_max_fit = lead_fitnesses[0];
        for (i, &f) in lead_fitnesses.iter().enumerate() {
            if f > lead_max_fit {
                lead_max_fit = f;
                lead_best_idx = i;
            }
        }
        let lead_best_fit = lead_fitnesses[lead_best_idx];
        let lead_avg_fit = lead_fitnesses.iter().sum::<f64>() / lead_fitnesses.len() as f64;

        let mut follow_best_idx = 0;
        let mut follow_max_fit = follow_fitnesses[0];
        for (i, &f) in follow_fitnesses.iter().enumerate() {
            if f > follow_max_fit {
                follow_max_fit = f;
                follow_best_idx = i;
            }
        }
        let follow_best_fit = follow_fitnesses[follow_best_idx];
        let follow_avg_fit = follow_fitnesses.iter().sum::<f64>() / follow_fitnesses.len() as f64;

        // Add to Hall of Fame
        lead_hof.add(&lead_pop.genomes[lead_best_idx], lead_best_fit, gen);
        follow_hof.add(&follow_pop.genomes[follow_best_idx], follow_best_fit, gen);

        // Snapshot display values BEFORE breeding (breeding replaces the population)
        let lead_species = lead_pop
            .species_list
            .iter()
            .filter(|s| !s.members.is_empty())
            .count();
        let follow_species = follow_pop
            .species_list
            .iter()
            .filter(|s| !s.members.is_empty())
            .count();
        let lead_best_conns = lead_pop.genomes[lead_best_idx].num_enabled();
        let follow_best_conns = follow_pop.genomes[follow_best_idx].num_enabled();
        let t_stats = t_stats_start.elapsed();

        // ── Reseeding (stagnation recovery, rare) ───────────────────────
        let t_reseed_start = Instant::now();
        // Breed next generation
        if current_phase == 1 && lead_pop.generations_since_improvement > 20 {
            if let Some(gb_clone) = lead_pop.global_best_genome.as_ref().map(|g| g.copy()) {
                println!(
                    "  >>> Lead: No improvement for {} generations. Reseeding species with mutations of the global best champion...",
                    lead_pop.generations_since_improvement
                );
                lead_pop.reseed_from_champion(&gb_clone, 0.10, &lead_tabu, &lead_registry, &mut rng);
                lead_pop.generations_since_improvement = 0;
            }
        }
        if current_phase == 1 && follow_pop.generations_since_improvement > 20 {
            if let Some(gb_clone) = follow_pop.global_best_genome.as_ref().map(|g| g.copy()) {
                println!(
                    "  >>> Follow: No improvement for {} generations. Reseeding species with mutations of the global best champion...",
                    follow_pop.generations_since_improvement
                );
                follow_pop.reseed_from_champion(&gb_clone, 0.10, &follow_tabu, &follow_registry, &mut rng);
                follow_pop.generations_since_improvement = 0;
            }
        }
        let t_reseed = t_reseed_start.elapsed();

        // ── Speciation + Breeding ───────────────────────────────────────
        let t_breed_start = Instant::now();
        if current_phase == 0 {
            let lead_eval_data = Some((&lead_dataset, config.evaluation.sweep_weights.as_slice()));
            let follow_eval_data = Some((&follow_dataset, config.evaluation.sweep_weights.as_slice()));

            lead_pop.speciate_and_evolve(
                current_phase,
                config.curriculum.bulking_gens,
                &lead_tabu,
                lead_eval_data,
                &mut rng,
                &lead_registry,
            );

            follow_pop.speciate_and_evolve(
                current_phase,
                config.curriculum.bulking_gens,
                &follow_tabu,
                follow_eval_data,
                &mut rng,
                &follow_registry,
            );
        } else {
            crate::population::speciate_and_evolve_joint(
                &mut lead_pop,
                &mut follow_pop,
                &lead_tabu,
                &follow_tabu,
                &lead_registry,
                &follow_registry,
                &mut rng,
            );
        }
        let t_breed = t_breed_start.elapsed();

        // ── Checkpointing (every 10 gens + final) ───────────────────────
        let t_ckpt_start = Instant::now();
        if (gen + 1) % 10 == 0 || gen == config.population.generations - 1 {
            // Save HOF (Joint)
            let hof_path = Path::new(&config.output.checkpoint_dir)
                .join("genomes")
                .join(format!("hof_gen{:04}.json", gen + 1));
            let json_lead_hof = JsonHallOfFame::from_hof(&lead_hof);
            let json_follow_hof = JsonHallOfFame::from_hof(&follow_hof);
            let joint_hof = JsonHallOfFameJoint {
                lead: json_lead_hof,
                follow: json_follow_hof,
            };
            let hof_file = fs::File::create(hof_path)?;
            serde_json::to_writer_pretty(hof_file, &joint_hof)?;

            // Save global best genome
            let lead_best = lead_pop.global_best_genome.as_ref().map(JsonGenome::from_genome);
            let follow_best = follow_pop.global_best_genome.as_ref().map(JsonGenome::from_genome);
            let joint_best = JsonGenomeJoint {
                lead: lead_best,
                follow: follow_best,
            };
            let best_path = Path::new(&config.output.checkpoint_dir)
                .join("genomes")
                .join(format!("best_genome_gen{:04}.json", gen + 1));
            let best_file = fs::File::create(best_path)?;
            serde_json::to_writer_pretty(best_file, &joint_best)?;

            // Save generation best genome
            let lead_gen_best = Some(JsonGenome::from_genome(&lead_pop.genomes[lead_best_idx]));
            let follow_gen_best = Some(JsonGenome::from_genome(&follow_pop.genomes[follow_best_idx]));
            let joint_gen_best = JsonGenomeJoint {
                lead: lead_gen_best,
                follow: follow_gen_best,
            };
            let gen_best_path = Path::new(&config.output.checkpoint_dir)
                .join("genomes")
                .join(format!("gen_best_genome_gen{:04}.json", gen + 1));
            let gen_best_file = fs::File::create(gen_best_path)?;
            serde_json::to_writer_pretty(gen_best_file, &joint_gen_best)?;

            // Save stateful training checkpoint
            let state = TrainingState {
                generation: gen + 1,
                current_phase,
                lead: BrainTrainingState {
                    genomes: lead_pop.genomes.clone(),
                    species: lead_pop.species_list.clone(),
                    hof: lead_hof.clone(),
                    map_elites: lead_map_elites.clone(),
                    next_species_id: lead_pop.next_species_id,
                    global_best_fitness: lead_pop.global_best_fitness,
                    global_best_genome: lead_pop.global_best_genome.clone(),
                    generations_since_improvement: lead_pop.generations_since_improvement,
                    next_innovation: lead_registry.lock().unwrap().next_innovation,
                },
                follow: BrainTrainingState {
                    genomes: follow_pop.genomes.clone(),
                    species: follow_pop.species_list.clone(),
                    hof: follow_hof.clone(),
                    map_elites: follow_map_elites.clone(),
                    next_species_id: follow_pop.next_species_id,
                    global_best_fitness: follow_pop.global_best_fitness,
                    global_best_genome: follow_pop.global_best_genome.clone(),
                    generations_since_improvement: follow_pop.generations_since_improvement,
                    next_innovation: follow_registry.lock().unwrap().next_innovation,
                },
            };
            state.save_to_file(&state_checkpoint_path)?;

            // --- Runtime data snapshot (non-blocking background write) ---
            let data_dir = Path::new(&run_dir).join("data");
            let snapshot = runtime_data::RuntimeDataSnapshot {
                lead_tabu: lead_tabu.dump_pairs(),
                follow_tabu: follow_tabu.dump_pairs(),
                lead_innov: lead_registry.lock().unwrap().dump_state(),
                follow_innov: follow_registry.lock().unwrap().dump_state(),
                lead_species: runtime_data::extract_species_summary(&lead_pop.species_list),
                follow_species: runtime_data::extract_species_summary(&follow_pop.species_list),
                lead_map_elites: lead_map_elites.dump_grid(),
                follow_map_elites: follow_map_elites.dump_grid(),
                lead_population: runtime_data::extract_population_snapshot(
                    &lead_pop,
                    &lead_pop.species_list,
                ),
                follow_population: runtime_data::extract_population_snapshot(
                    &follow_pop,
                    &follow_pop.species_list,
                ),
            };
            std::thread::spawn(move || {
                if let Err(e) = runtime_data::save_runtime_data(data_dir, &snapshot) {
                    eprintln!("Warning: failed to save runtime data: {e}");
                }
            });
        }

        let t_ckpt = t_ckpt_start.elapsed();

        // ── Fixed-yardstick probe (every 25 Phase 1 gens) ──
        let mut fixed_probe_delta = 0.0f64;
        if current_phase == 1 && gen % 25 == 0 {
            if let Some(ref probe_deals) = frozen_probe_deals {
                let best_lead = lead_pop.genomes[lead_best_idx].to_rust_wann();
                let best_follow = follow_pop.genomes[follow_best_idx].to_rust_wann();
                fixed_probe_delta = evaluate_fixed_probe(
                    &best_lead, &best_follow, probe_deals,
                    &config.evaluation.sweep_weights, config.evaluation.seed + 7777,
                );
            }
        }

        // --- End-of-generation timing and reporting (captures ALL work) ---
        let t_total = t_gen.elapsed().as_secs_f64();

        writeln!(
            csv_file,
            "{},{},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{:.6},{:.6},{:.6},{:.2}",
            gen,
            current_phase,
            lead_best_fit,
            lead_avg_fit,
            follow_best_fit,
            follow_avg_fit,
            lead_species,
            follow_species,
            lead_best_conns,
            follow_best_conns,
            lead_val_acc,
            follow_val_acc,
            fixed_probe_delta,
            t_total
        )?;
        csv_file.flush()?;

        // ── Console output: fitness table + per-phase profiling ─────────
        println!(
            "{:4} {:2} {:8.4} {:8.4} {:8.4} {:8.4} {:6} {:6} {:6} {:6} {:8.4} {:8.4} {:7.1}s",
            gen,
            current_phase,
            lead_best_fit,
            lead_avg_fit,
            follow_best_fit,
            follow_avg_fit,
            lead_species,
            follow_species,
            lead_best_conns,
            follow_best_conns,
            lead_val_acc,
            follow_val_acc,
            t_total
        );
        println!(
            "     phase:{:7.2}s convert:{:7.2}s eval:{:7.2}s stats:{:7.2}s reseed:{:7.2}s breed:{:7.2}s ckpt:{:7.2}s  [O(eval)=P·D·R·|W|·E, O(breed)=P·S·E+P·K·E]",
            t_phase.as_secs_f64(),
            t_convert.as_secs_f64(),
            t_eval.as_secs_f64(),
            t_stats.as_secs_f64(),
            t_reseed.as_secs_f64(),
            t_breed.as_secs_f64(),
            t_ckpt.as_secs_f64(),
        );
    }

    // Save final files
    let hof_path = Path::new(&config.output.checkpoint_dir)
        .join("genomes")
        .join("hof_final.json");
    let json_lead_hof = JsonHallOfFame::from_hof(&lead_hof);
    let json_follow_hof = JsonHallOfFame::from_hof(&follow_hof);
    let joint_hof = JsonHallOfFameJoint {
        lead: json_lead_hof,
        follow: json_follow_hof,
    };
    let hof_file = fs::File::create(hof_path)?;
    serde_json::to_writer_pretty(hof_file, &joint_hof)?;

    let lead_best = lead_pop.global_best_genome.as_ref().map(JsonGenome::from_genome);
    let follow_best = follow_pop.global_best_genome.as_ref().map(JsonGenome::from_genome);
    let joint_best = JsonGenomeJoint {
        lead: lead_best,
        follow: follow_best,
    };
    let best_path = Path::new(&config.output.checkpoint_dir)
        .join("genomes")
        .join("best_genome_final.json");
    let best_file = fs::File::create(best_path)?;
    serde_json::to_writer_pretty(best_file, &joint_best)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::ExpertDataset;
    use crate::genome::Genome;
    use sueca_solver::heuristic::PhiCtx;

    /// Build a synthetic dataset of `n` leading states (trick_len = 0) with the
    /// given per-state `best_cards` mask. Beliefs are zero-filled — irrelevant
    /// to the card-match arithmetic, which only depends on the resolver output
    /// vs the mask. Hands are randomized but always have ≥2 cards.
    fn synthetic_dataset(n: usize, best_cards_fn: impl Fn(&PhiCtx) -> u64) -> ExpertDataset {
        use rand::SeedableRng;
        let mut rng = rand_pcg::Pcg64::seed_from_u64(424242);
        let mut states = Vec::new();
        let mut bc = Vec::new();
        let mut trump_v = Vec::new();
        let mut hand_v = Vec::new();
        let mut trick_v = Vec::new();
        let mut trick_len_v = Vec::new();
        for _ in 0..n {
            // Random hand of 4..=8 cards.
            let mut deck: Vec<u8> = (0..40).collect();
            use rand::seq::SliceRandom;
            deck.shuffle(&mut rng);
            let ncards = rng.gen_range(4..=8);
            let hand: u64 = deck[..ncards].iter().map(|&c| 1u64 << c).fold(0, |a, b| a | b);
            let trump = rng.gen_range(0..4) as u8;
            let ctx = PhiCtx { trump, hand, trick_cards: [40; 4], trick_len: 0 };
            states.extend(std::iter::repeat_n(0.0f64, INPUT_COUNT));
            bc.push(best_cards_fn(&ctx));
            trump_v.push(trump);
            hand_v.push(hand);
            trick_v.extend_from_slice(&[40u8, 40, 40, 40]);
            trick_len_v.push(0);
        }
        ExpertDataset {
            states,
            num_states: n,
            best_cards: bc,
            ctx_trump: trump_v,
            ctx_hand: hand_v,
            ctx_trick: trick_v,
            ctx_trick_len: trick_len_v,
        }
    }

    /// Invariant: if every legal card is "best" (mask = legal set), then any
    /// legal resolver output is correct → Phase-0 fitness must be exactly 1.0.
    /// Uses an empty genome (all-zero outputs → knobs all −1 → resolver picks a
    /// deterministic lowest-φ legal card); the point is the fitness arithmetic,
    /// which must be 1.0 regardless of which legal card is chosen.
    #[test]
    fn phase0_fitness_all_legal_is_one() {
        let dataset = synthetic_dataset(40, |ctx| ctx.legal());
        let nets = vec![Genome::initial().to_rust_wann()];
        for sweep in [&[1.0f64][..], &[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0][..]] {
            let fit = evaluate_phase0(&nets, &dataset, sweep, false);
            assert_eq!(fit.len(), 1);
            assert!(
                (fit[0] - 1.0).abs() < 1e-9,
                "fitness {} != 1.0 when every legal card is best (sweep len {})",
                fit[0],
                sweep.len()
            );
        }
    }

    /// Invariant: a mask with NO legal bits set can never be matched by a legal
    /// resolver output → Phase-0 fitness must be exactly 0.0. Confirms the mask
    /// check genuinely penalizes (not a tautological always-1.0).
    #[test]
    fn phase0_fitness_empty_mask_is_zero() {
        // Mask sets only an ILLEGAL bit (bit 39 with hand excluding 39) → resolver
        // (which always plays legal) can never match → fitness 0.0.
        let dataset = synthetic_dataset(20, |_| 1u64 << 39);
        let nets = vec![Genome::initial().to_rust_wann()];
        let fit = evaluate_phase0(&nets, &dataset, &[1.0], false);
        assert_eq!(fit.len(), 1);
        assert!(
            (fit[0] - 0.0).abs() < 1e-9,
            "fitness {} != 0.0 with an all-illegal mask",
            fit[0]
        );
    }

    /// Range invariant: for realistic single-best-card masks, fitness ∈ [0,1]
    /// and equals the fraction of states where the resolver picks a best card.
    #[test]
    fn phase0_fitness_single_best_card_in_range() {
        // best card = lowest-index legal card. Resolver with empty net + zero
        // knobs picks a deterministic card; fitness is the resulting hit rate.
        let dataset = synthetic_dataset(30, |ctx| {
            let l = ctx.legal();
            1u64 << l.trailing_zeros()
        });
        let nets = vec![Genome::initial().to_rust_wann()];
        let fit = evaluate_phase0(&nets, &dataset, &[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], false);
        assert_eq!(fit.len(), 1);
        assert!(fit[0].is_finite(), "fitness not finite");
        assert!((0.0..=1.0).contains(&fit[0]), "fitness {} out of [0,1]", fit[0]);
    }
}
