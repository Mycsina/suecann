# Optimization Tasks — Sueca WANN

> Generated from profiling data (2026-06-02). See `docs/OPTIMIZATION_LOG.md` for raw numbers.

## Phase 0: Kill the 34% Dispatch Tax

### Problem
`RustWannNetwork::forward` = 63.86% self. `FnMut::call_mut` = 34.57% self.
The per-sample closure dispatch in the Phase 0 dataset evaluation loop burns a third of CPU time on iterator/closure overhead rather than actual WANN inference.

### Root Cause
The Phase 0 dataset loop processes samples scalar-by-scalar through Rayon parallel iterators with dynamic closure dispatch. Each sample triggers a `FnMut::call_mut` → `forward` call chain. With 14K+ lead states × 400 genomes × 25 generations, this closure is invoked billions of times, and the compiler cannot inline `forward` across the closure boundary.

### Fix
Rewrite `evaluate_single_phase0_sample` (and its callers) to accept a flat contiguous slice `&[f32]` (or `&[[f32; INPUT_COUNT]]`) and run the forward pass in a tight inner loop. Let the compiler see the loop body and inline `forward` directly, eliminating the per-element closure boundary.

```rust
// BEFORE (34% overhead): per-sample closure dispatch
dataset.states.par_iter().map(|state| {
    network.forward(state, weight, scratchpad);
    // ... compute accuracy
})

// AFTER: batched forward over flat slice
fn evaluate_batch(states: &[[f64; INPUT_COUNT]], network: &RustWannNetwork, weight: f64) {
    let mut scratchpad = vec![0.0; network.num_nodes];
    for state in states {
        network.forward(state, weight, &mut scratchpad);
        // ... compute accuracy (inlined, no closure boundary)
    }
}
```

### Expected Gain
Reclaims most of the 34.57% dispatch overhead. Phase 0 per-gen time drops from ~0.8s to ~0.5s.

### Risk
**Low.** Structural refactor only — no algorithm changes, no GPU, no SIMD. Existing tests protect correctness.

---

## Phase 1: Cache the HeuristicBot Baseline with Compound State Keys

### Problem
`select_card_heuristic_old` = 41.67% of Phase 1 CPU. The HeuristicBot baseline is re-evaluated on the same deals for every genome (1000 genomes × 128 deals × 4 seats), but its decisions depend on the full game state — not just the deal ID.

### Cache Key Trap
A naive `(deal_id, seat_id)` cache key is **wrong** because different genomes make different card plays, causing game states to branch. The HeuristicBot at seat 2 in trick 3 sees different cards depending on what the WANN at seat 0 played in tricks 0-2.

### Correct Cache Key
```rust
#[derive(Hash, Eq, PartialEq)]
pub struct BaselineCacheKey {
    pub deal_id: u64,              // Which deal
    pub played_cards_bitmask: u64, // Exactly which cards have been played (40 bits used)
    pub current_trick_bytes: [u8; 4], // Cards in current trick + length (captures lead context)
    pub seat_id: u8,               // Whose turn to play
}
```

### Thread Safety
The evaluator uses Rayon parallel iteration. The cache must be a concurrent map:
- `DashMap<BaselineCacheKey, u8>` for lock-free reads
- Or `Arc<RwLock<HashMap<...>>>` if contention is low

### Expected Gain
- Early tricks (0-3): near 100% hit rate — game states haven't diverged yet
- Late tricks (7-9): lower hit rate due to branching, but fewer cards remain so simulation is cheaper
- Overall: 40-50% reduction in HeuristicBot calls → Phase 1 per-gen time drops from ~5.8s to ~3s

### Risk
**Low-Medium.** The cache key must be exhaustively correct — missing a state component means cache poisoning (wrong card returned). Test by comparing cached vs non-cached baseline scores on a full generation; they must be identical.

---

## GPU Assessment: Killed by Amdahl's Law

| Target | Phase 0 Share | Phase 1 Share | GPU Worth It? |
|--------|---------------|---------------|---------------|
| WANN `forward()` | 63.9% | 2.6% | Only Phase 0 |
| Game simulation | 0% | 23.3% | No (branch-heavy bitboard logic) |
| HeuristicBot | 0% | 49.5% | No (classical rule-based algorithm) |

GPU acceleration of `forward()` for Phase 0 is viable only after the 34% dispatch tax is eliminated. CPU-side batching (flattened loops + auto-vectorization) should be attempted first — it's lower risk and may deliver enough speedup to make GPU unnecessary.

### GPU Decision Gate
After Phase 0 dispatch fix:
- If Phase 0 remains >30% of total training time → evaluate batched CUDA `forward()`
- If Phase 0 drops below 15% of total training time → GPU not worth the complexity

---

## Task Priority

| # | Task | Phase | Est. Effort | Est. Gain | Risk |
|---|------|-------|-------------|-----------|------|
| 1 | Flatten Phase 0 dataset loop (kill dispatch tax) | 0 | 1-2 days | ~34% Phase 0 speedup | Low |
| 2 | Compound-state HeuristicBot cache | 1 | 2-4 days | ~50% Phase 1 speedup | Low-Med |
| 3 | CPU auto-vectorization of batched `forward()` | 0 | 1 week | 2-4× Phase 0 speedup | Med |
| 4 | GPU `forward()` — only if tasks 1+3 insufficient | 0 | 3-4 weeks | 5-10× Phase 0 | High |
| 5 | Compiler flags: LTO + CGU=1 + panic=abort | global | 5 min | 5-15% global | Low |
| 6 | PGO: profile-guided optimization | global | 1 day | 5-20% branch-heavy code | Low |
| 7 | WANN instruction tape (enable SIMD on edges) | 0 | 1-2 weeks | 20-30% of forward() | Med |

---

## Global: Compiler-Level Optimizations (Accepted)

### Problem
No `[profile.release]` section exists in any workspace `Cargo.toml`. Rust defaults apply:
- `lto = false` → cross-crate inlining disabled (`forward()` calls solver bitboard ops across crate boundary)
- `codegen-units = 16` → LLVM sees only 1/16th of the call graph, can't optimize globally
- `panic = "unwind"` → stack-unwinding landing pads in every function, bloating code and icache

### Fix
Add to workspace root `Cargo.toml`:

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
```

### PGO (Separate Step)
```bash
# 1. Build instrumented binary
RUSTFLAGS="-C profile-generate=/tmp/pgo-data" cargo build -p sueca_wann --release

# 2. Run training to collect branch profiles (5 gens each phase is enough)
./target/release/sueca_wann train --config configs/profile_phase0.toml
./target/release/sueca_wann train --config configs/profile_phase1.toml --resume

# 3. Merge profiles and rebuild
llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data/
RUSTFLAGS="-C profile-use=/tmp/pgo-data/merged.profdata" cargo build -p sueca_wann --release
```

### Expected Gain
- LTO: inlines bitboard ops (`leading_zeros`, `trailing_zeros`, `count_ones`) across the `sueca_wann` → `sueca_solver` crate boundary. These are called inside `forward()` and the heuristic — eliminating function call overhead for single-instruction bit ops.
- CGU=1: LLVM can reorder code across the entire binary, placing hot paths contiguous in memory.
- Panic=abort: removes unwind tables from every function, improving icache density.
- PGO: reorders branch layouts in `match` blocks (heuristic card selection has many conditional branches on suit, game phase, etc.)

### Risk
**Low.** LTO increases compile time but has no runtime effect on correctness. Panic=abort means no `catch_unwind` — verify the codebase doesn't use it: `grep -r "catch_unwind" src/` returns empty.

---

## Phase 0: WANN Instruction Tape — SIMD-Ready Forward Pass (Accepted with Caveats)

### Problem
`forward()` = 63.86% of Phase 0. The current CSR-based execution has:
1. Per-node `node_ptrs` index indirection (2 loads per node)
2. Nested loop structure (for node → for edge) that blocks auto-vectorization
3. `scratchpad[src]` reads scattered across the scratchpad — cache-hostile but unavoidable

### What the Instruction Tape Fixes (and What It Doesn't)
- ✅ **Eliminates node_ptrs lookups**: edges are a flat array of opcodes, no per-node index indirection
- ✅ **Enables SIMD**: flat loop over uniform opcodes lets LLVM auto-vectorize 4-8 edge computations at once
- ✅ **Single-level loop**: better instruction prefetch, no nested branch mispredicts
- ❌ **Does NOT fix scratchpad[src] scatter**: source values are still read from arbitrary positions — this is fundamental to the WANN graph structure
- ❌ **Does NOT reduce total operations**: same number of edge computations, just in a different layout

### Implementation Sketch
```rust
#[derive(Clone, Copy)]
pub enum WannOp {
    /// Compute signal from `src` node, accumulate into `dst` node's register
    Edge { src: u16, dst: u16, sign: i8 },
    /// Apply activation to `node`'s accumulated value
    Activate { node: u16, func: u8 },
}

pub struct LinearWann {
    ops: Vec<WannOp>,       // Flat instruction tape (post-compilation)
    registers: Vec<f64>,    // Scratchpad (accumulator + final values)
}

impl LinearWann {
    pub fn forward(&self, inputs: &[f64; INPUT_COUNT], weight: f64) {
        // 1. Load inputs into registers
        // 2. Linear sweep over ops[] — single loop, no nested iteration
        // 3. Read outputs from registers[OUTPUT_START..]
    }
}
```

### Expected Gain
20-30% reduction in `forward()` time (realistic, not the claimed 50%). Translates to 13-19% of Phase 0 total.

### Risk
**Medium.** Requires a new WANN representation alongside the existing CSR format. Affects genome→network conversion and the forward pass. The instruction tape is larger than CSR for networks with many edges per node (each edge is an explicit opcode). Best deployed as an optional compilation step for champion networks that are evaluated millions of times, not for every transient genome during evolution.

---

## Phase 1: Do/Undo for Game State — REJECTED

### Why Rejected
The proposal claims 23.3% of Phase 1 is "deep-cloning overhead." This is incorrect.

**Measured facts:**
- `GameState` = 48 bytes (derives `Copy`)
- `SuecaSimulatorGame` = 72 bytes (derives `Clone`)
- Clone happens once per world: 5 worlds × 72 bytes = **360 bytes** per `evaluate_state_potential` call
- After cloning, each world plays ~40 cards to completion: 5 × 40 = **200 card play operations**
- Each card play involves: bitboard manipulation, trick resolution, void tracking, winner determination

The 23.3% (`play_card_and_resolve` 13.2% + `play_card` 10.1%) is from the 200 card play operations per call, not from copying 360 bytes. A do/undo frame would save the 72-byte clone at the cost of:
1. A push/pop stack per card play (replaces the one-time clone)
2. Architectural complexity (MoveFrame management, ensuring pop matches push)
3. Zero measurable performance improvement (360 bytes vs 200 bitboard operations)

**Verdict:** Clone is not the bottleneck. Game simulation is. If you want to reduce the 23%, you need to optimize the bitboard operations inside `play_card_and_resolve`, not the memcpy outside it.

### Minor Fix
`SuecaSimulatorGame` should derive `Copy` (all fields are `Copy`). Cost: zero. Benefit: `let mut sim_game = game;` instead of `game.clone()`, no functional difference.
