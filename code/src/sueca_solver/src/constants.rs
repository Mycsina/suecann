pub const INPUT_START: usize = 0;
pub const INPUT_COUNT: usize = 35;
pub const BIAS_ID: usize = INPUT_START + INPUT_COUNT; // 35
pub const OUTPUT_START: usize = BIAS_ID + 1; // 36
/// Number of φ-utility knobs the WANN emits (one per card-utility feature).
/// Stage B resolver overhaul (2026-06-19): the network output layer is no
/// longer a 3-intent selector but a 6-dimensional continuous knob vector; the
/// resolver picks `argmax_{legal} Σ_k knob_k · φ_k(card, state)`. See
/// `heuristic::resolve_card_phi_utility`.
pub const OUTPUT_COUNT: usize = PHI_FEATURE_COUNT;
pub const FIRST_HIDDEN_ID: usize = OUTPUT_START + OUTPUT_COUNT; // 42

/// Number of hand-designed card-utility features φ(card, state). Must stay in
/// sync with `compute_phi` in `heuristic.rs` and `PHI_FEATURE_NAMES`.
pub const PHI_FEATURE_COUNT: usize = 6;

/// φ(card, state) feature indices. The WANN emits one knob per feature; the
/// resolver forms the card utility `Σ_k knob_k · φ_k(card, state)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum PhiFeature {
    /// Normalized card rank (0=lowest .. 1=Ace). CARD_RANK / 9.0.
    Rank = 0,
    /// Normalized card point value. CARD_POINTS / 11.0.
    Points = 1,
    /// 1.0 if the card is a trump, else 0.0.
    Trump = 2,
    /// 1.0 if this card would beat the current trick winner, else 0.0.
    Wins = 3,
    /// Points captured if this card wins (current trick points / 30.0), else 0.0.
    Captures = 4,
    /// 1.0 if this is the ego player's last card of its suit (builds/keeps a void).
    Void = 5,
}

/// Names of the φ features / WANN output knobs, indexed by `PhiFeature`.
pub const PHI_FEATURE_NAMES: [&str; PHI_FEATURE_COUNT] = [
    "RANK",      // φ0 — prefer high/low rank
    "POINTS",    // φ1 — prefer high/low point value
    "TRUMP",     // φ2 — prefer/avoid trumps
    "WINS",      // φ3 — prefer/avoid winning the trick now
    "CAPTURES",  // φ4 — weight on points captured
    "VOID",      // φ5 — prefer/avoid creating a void
];
