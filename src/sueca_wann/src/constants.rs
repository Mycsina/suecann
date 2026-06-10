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

// Differential Evolution hyperparameters for weight optimization.
pub const DE_POP_SIZE: usize = 50;
pub const DE_F_SCALING: f64 = 0.5;
pub const DE_CR_CROSSOVER: f64 = 0.7;
pub const DE_WEIGHT_MIN: f64 = -2.0;
pub const DE_WEIGHT_MAX: f64 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum BeliefFeature {
    HasLedSuit = 0,
    HasTrump = 1,
    LedSuitCount = 2,
    TrumpCount = 3,
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
    HoldsBossLed = 17,
    HoldsBossTrump = 18,
    CanBeatWinner = 19,
    MinWinningCost = 20,
    MinSacrificeCost = 21,
    GamePtsRemaining = 22,
    TrickNumber = 23,
    TrumpsRemaining = 24,
    ScoreDelta = 25,
    MyVoidCount = 26,
    LongestSideSuit = 27,
    ShortestSideSuit = 28,
    Side0Depletion = 29,
    Side1Depletion = 30,
    Side2Depletion = 31,
    PointsSecured = 32,
    KnownVoidSuitsCount = 33,
    DepletedSuitsCount = 34,
}

// Sueca feature names (indexed by belief-state dimension, 0..35).
pub const FEATURE_NAMES: [&str; 35] = [
    "Has_Led_Suit",
    "Has_Trump",
    "Led_Suit_Count",
    "Trump_Count",
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
    "Holds_Boss_Led",
    "Holds_Boss_Trump",
    "Can_Beat_Winner",
    "Min_Winning_Cost",
    "Min_Sacrifice_Cost",
    "Game_Pts_Remaining",
    "Trick_Number",
    "Trumps_Remaining",
    "Score_Delta",
    "My_Void_Count",
    "Longest_Side_Suit",
    "Shortest_Side_Suit",
    "Side0_Depletion",
    "Side1_Depletion",
    "Side2_Depletion",
    "Points_Secured_Us",
    "Known_Void_Suits_Count",
    "Depleted_Suits_Count",
];

// Oracle intent names (indexed by output neuron, 0..3).
// MIN_FORCE removed — EFFICIENT_WIN subsumes its useful behavior.
pub const OUTPUT_NAMES: [&str; 3] = ["MAX_FORCE", "EFFICIENT_WIN", "EQUITY_BUILDER"];

// Number of worlds to use for the heuristic potential evaluation rollouts.
pub const POTENTIAL_EVAL_WORLDS: usize = 5;
