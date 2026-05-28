// Evolutionary hyperparameters for the Sueca WANN training pipeline.
// Dimension constants (INPUT_COUNT, OUTPUT_COUNT, etc.) live in sueca_solver::constants.
// These are reference/tuning constants — not all are directly referenced in source.

#![allow(dead_code)]

// Floating-point epsilon for tie-breaking comparisons.
pub const FLOAT_EPSILON: f64 = 1e-9;

// Structural baseline penalty applied to the EQUITY_BUILDER (intent 3) output.
pub const EQUITY_FALLBACK_PENALTY: f64 = 0.25;

// Tournament / breeding hyperparameters.
pub const TOURNAMENT_SIZE: usize = 3;
pub const FITNESS_OFFSET: f64 = 0.1;
pub const STAGNATION_RESEED_LIMIT: usize = 20;

// Champion reseed mutation rates.
pub const RESEED_ADD_NODE_PROB: f64 = 0.05;
pub const RESEED_ADD_CONN_PROB: f64 = 0.10;

// Partner / opponent seat offsets (partner = (seat + 2) % 4).
pub const PARTNER_OFFSET: usize = 2;

// Sueca feature names (indexed by belief-state dimension, 0..30).
pub const FEATURE_NAMES: [&str; 30] = [
    "Has_Led_Suit",
    "Has_Trump",
    "Led_Suit_Power",
    "Trump_Power",
    "Hand_Point_Density",
    "Am_I_Leading",
    "Am_I_Last_To_Play",
    "Is_Partner_Winning",
    "Trick_Point_Value",
    "Has_Trick_Been_Cut",
    "Partner_Void_Led",
    "Partner_Void_Trump",
    "Any_Opp_Void_Led",
    "Any_Opp_Void_Trump",
    "Led_Suit_Ace_Played",
    "Led_Suit_7_Played",
    "Trump_Ace_Played",
    "Game_Pts_Remaining",
    "Trick_Number",
    "Trumps_Remaining",
    "Score_Delta",
    "Side0_Depletion",
    "Side0_Ace_Played",
    "Side0_7_Played",
    "Side1_Depletion",
    "Side1_Ace_Played",
    "Side1_7_Played",
    "Side2_Depletion",
    "Side2_Ace_Played",
    "Side2_7_Played",
];

// Oracle intent names (indexed by output neuron, 0..4).
pub const OUTPUT_NAMES: [&str; 4] = ["MAX_FORCE", "MIN_FORCE", "EFFICIENT_WIN", "EQUITY_BUILDER"];
