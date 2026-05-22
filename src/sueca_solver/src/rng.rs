/// Deterministic LCG RNG shared across the solver crate.
/// Based on Numerical Recipes multiplier and increment.
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    pub fn gen_range(&mut self, range: std::ops::Range<usize>) -> usize {
        let span = range.end - range.start;
        if span == 0 {
            return range.start;
        }
        range.start + (self.next_u64() as usize % span)
    }
}
