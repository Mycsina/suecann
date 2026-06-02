// Evolutionary hyperparameters for the Sueca WANN training pipeline.
// Dimension constants (INPUT_COUNT, OUTPUT_COUNT, etc.) live in sueca_solver::constants.
// These are reference/tuning constants — not all are directly referenced in source.

#![allow(dead_code)]

// Floating-point epsilon for tie-breaking comparisons.
pub const FLOAT_EPSILON: f64 = 1e-9;

// Tournament / breeding hyperparameters.
pub const TOURNAMENT_SIZE: usize = 3;
pub const FITNESS_OFFSET: f64 = 0.1;
pub const STAGNATION_RESEED_LIMIT: usize = 20;

// Champion reseed mutation rates.
pub const RESEED_ADD_NODE_PROB: f64 = 0.05;
pub const RESEED_ADD_CONN_PROB: f64 = 0.10;

// Partner / opponent seat offsets (partner = (seat + 2) % 4).
pub const PARTNER_OFFSET: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum BeliefFeature {
    HasLedSuit = 0,
    HasTrump = 1,
    LedSuitPower = 2,
    TrumpPower = 3,
    HandPointDensity = 4,
    AmILeading = 5,
    AmILastToPlay = 6,
    IsPartnerWinning = 7,
    TrickPointValue = 8,
    HasTrickBeenCut = 9,
    PartnerVoidLed = 10,
    PartnerVoidTrump = 11,
    AnyOppVoidLed = 12,
    AnyOppVoidTrump = 13,
    LedSuitAcePlayed = 14,
    LedSuit7Played = 15,
    TrumpAcePlayed = 16,
    GamePtsRemaining = 17,
    TrickNumber = 18,
    TrumpsRemaining = 19,
    ScoreDelta = 20,
    Side0Depletion = 21,
    Side0AcePlayed = 22,
    Side07Played = 23,
    Side1Depletion = 24,
    Side1AcePlayed = 25,
    Side17Played = 26,
    Side2Depletion = 27,
    Side2AcePlayed = 28,
    Side27Played = 29,
}

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

// Number of worlds to use for the heuristic potential evaluation rollouts.
pub const POTENTIAL_EVAL_WORLDS: usize = 5;
