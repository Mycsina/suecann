# Optimization Tasks — Sueca WANN

> Results summary from profiling + implementation (2026-06-02).

## Applied & Measured

| # | Optimization | Phase | Effort | Result |
|---|-------------|-------|--------|--------|
| 1 | Compiler flags: LTO + CGU=1 + panic=abort | global | 5 min | ✅ Applied |
| 2 | `target-cpu=native` | global | 1 min | ✅ Applied. Enables AVX2 auto-vectorization |
| 3 | Phase 0: pre-convert states + unrolled argmax | 0 | 30 min | ✅ **~28% Phase 0 speedup** (22.6s→16.2s wall) |
| 4 | `SuecaSimulatorGame` Copy derive | both | 2 min | ✅ Neutral (code clarity only) |
| 5 | `evaluate_single_phase0_sample` loop flatten | 0 | 15 min | ✅ Applied |
| 9 | PGO (Profile-Guided Optimization) | both | 30 min | ✅ **13.4% wall-time** (21.3s→18.5s). Phase 0: ~20%, Phase 1: ~12% |

## Rejected

| # | Optimization | Reason |
|---|-------------|--------|
| 6 | HeuristicBot compound-state cache | ❌ Memory explosion: 17GB RSS. Game state space too large (400 genomes × 128 deals × 40 plays). Needs per-deal pre-computation approach instead |
| 7 | Do/Undo game state | ❌ Clone is 72 bytes × 5 worlds = 360 bytes. Cost is in 200 card-play bitboard ops, not the memcpy |
| 8 | GPU WANN inference | ❌ Phase 1: `forward()` at 2.6% — Amdahl's Law. Phase 0 only (63.9%) but CPU wins are lower risk |

| 9 | PGO (Profile-Guided Optimization) | ✅ **Applied.** **13.4% wall-time reduction** (21.35s→18.49s) in Phase 0/1 combined benchmark. Phase 0: ~20% per-gen. Phase 1: ~12%. Binary 100KB smaller. Tested 2026-06-02. |

## Pending

| # | Optimization | Notes |
|---|-------------|-------|
| 10 | WANN instruction tape (SIMD) | 20-30% of remaining forward() time. Worth prototyping after PGO |
