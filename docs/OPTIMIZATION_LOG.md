# Optimization Log — Sueca WANN

> Last updated: 2026-06-02 (profiling data from `perf record -g -F 99`)

## Profiling Results (perf, 99Hz sampling)

### Phase 0: Supervised Pretraining (15,342 samples, ~20s runtime)

| Rank | % Self | Samples | Function | Category |
|------|--------|---------|----------|----------|
| 1 | **63.86%** | 9,763 | `RustWannNetwork::forward` | WANN inference |
| 2 | **34.57%** | 5,244 | `FnMut::call_mut` (forward dispatch wrapper) | WANN inference overhead |
| 3 | 0.38% | 71 | `evaluate_single_phase0_sample` | Dataset eval |
| 4 | 0.16% | 31 | `BuildHasher::hash_one` | Hashing |
| 5 | 0.12% | 21 | `compatibility_distance` | Speciation |
| — | <0.10% | — | Everything else combined < 1.0% | — |

**Phase 0 conclusion: 98.4% of CPU is `forward()` + its dispatch wrapper. This is the ONLY meaningful optimization target.**

### Phase 1: Co-evolutionary Self-play (106,508 samples, ~2.5min runtime)

| Rank | % Self | Samples | Function | Category |
|------|--------|---------|----------|----------|
| 1 | **41.67%** | 44,354 | `select_card_heuristic_old` | **HeuristicBot baseline** |
| 2 | **13.23%** | 14,072 | `GameState::play_card_and_resolve` | Game engine |
| 3 | **10.05%** | 10,707 | `SuecaSimulatorGame::play_card` | Simulator |
| 4 | 9.62% | 10,236 | `evaluate_state_potential` | PIMC eval |
| 5 | 7.83% | 8,324 | `select_card_heuristic::{{closure}}` | New heuristic |
| 6 | 7.18% | 7,640 | `SimulatorBot::select_card` | Bot dispatch |
| 7 | **2.63%** | 2,803 | `RustWannNetwork::forward` | **WANN inference** |
| 8 | 2.51% | 2,657 | `encode_belief_state` | Belief encoding |
| 9 | 1.58% | 1,682 | `sample_world` | PIMC sampling |
| 10 | 1.48% | 1,577 | `sample_constraints` | PIMC constraints |
| 11 | 0.55% | 592 | `resolve_intent` | Intent→card |
| — | <0.40% | — | Everything else combined < 4% | — |

**Phase 1 conclusion: The HeuristicBot dominates at 41.7% + 7.8% = 49.5%. The WANN forward() is only 2.6%! Game engine + simulator = 23.3%.**

### Phase Cost Comparison

| Cost Center | Phase 0 | Phase 1 | P1/P0 Ratio |
|-------------|---------|---------|-------------|
| WANN inference (`forward`) | 63.9% | 2.6% | 0.04× |
| HeuristicBot (old + new) | — | 49.5% | — |
| Game engine + simulator | — | 23.3% | — |
| PIMC evaluation | — | 9.6% | — |
| Belief encoding | — | 2.5% | — |
| Per-generation wall time | ~0.8s | ~5.8s | 7.3× |

## CUDA Enhancement Assessment — REVISED WITH PROFILING DATA

### Candidate 1: Batch WANN `forward()` on GPU — **Phase 0 only**

| | Phase 0 Impact | Phase 1 Impact |
|---|---|---|
| % of runtime | 63.9% | 2.6% |
| Max theoretical speedup | 2.7× (Amdahl) | 1.03× (Amdahl) |
| Worth GPU effort? | **Yes** — dominates Phase 0 | **No** — negligible in Phase 1 |

**Recommendation:** If Phase 0 training time is the bottleneck (200 gens with 63.9% in forward), a batched CPU `forward()` (SIMD or multi-threaded within a single genome's evaluation) could give 2-4× wall-clock improvement without GPU complexity. GPU only justified if Phase 0 dominates total training time AND you have the hardware/infrastructure.

### Candidate 2: Optimize HeuristicBot — **Highest Phase 1 ROI**

| | Phase 1 Impact |
|---|---|
| % of runtime | 49.5% (old 41.7% + new 7.8%) |
| Max theoretical speedup | 2.0× (Amdahl) |

**Recommendation:** This is the #1 Phase 1 optimization target. Options:
- Cache HeuristicBot card selections (same deal + same seat → same card, heuristic is deterministic)
- Pre-compute common patterns (leading logic, follow-suit responses)
- Replace `select_card_heuristic_old` with the newer `select_card_heuristic` (which uses intent-based resolution, already 7.8% vs 41.7%)
- The old heuristic does bit-twiddling loops per call (`while temp != 0 { temp &= temp - 1 }`) — these are called 4× per trick per game, hundreds of thousands of times per generation

### Candidate 3: Game Engine Optimization — **Phase 1 only**

| | Phase 1 Impact |
|---|---|
| % of runtime | 23.3% (engine 13.2% + simulator 10.1%) |
| Max theoretical speedup | 1.3× (Amdahl) |

**Recommendation:** Lower priority. The engine uses bitboards which are already efficient. Potential: reduce redundant play_card calls (the same card play is simulated for both WANN and baseline games on the same deal).

### Candidate 4: Cache Baseline Evaluations — **Quick Win**

The HeuristicBot baseline is evaluated on the same deals for EVERY genome. The baseline result is genome-independent — HeuristicBot vs HeuristicBot partners on deal D is always the same result.

**Proposal:** Pre-compute baseline scores for all 128 deals × 4 seats = 512 baseline games once per generation, then reuse across all 1000 genome evaluations.

**Estimated gain:** Eliminates ~50% of game simulations (the baseline half of the delta pair). Could reduce Phase 1 per-gen time from 5.8s to ~3s.
**Risk:** **Low** — the baseline is deterministic (same deal → same HeuristicBot plays).
**Priority:** **Highest ROI** — simple change, large impact, no GPU needed.

### Revised Priority Summary

| Priority | Target | Phase | Est. Gain | Effort | Risk |
|----------|--------|-------|-----------|--------|------|
| **P1** | Cache HeuristicBot baseline evals | 1 | 50% fewer sims | 1-2 days | Low |
| **P2** | Replace old heuristic with optimized new one | 1 | ~30% of Phase 1 time | 2-3 days | Low |
| **P3** | Batch WANN forward() (CPU SIMD) | 0 | 2-4× Phase 0 speed | 1-2 weeks | Med |
| P4 | GPU WANN forward() | 0 | 5-10× Phase 0 speed | 3-4 weeks | High |
| P5 | GPU genome eval | 1 | <3% total gain | 4-8 weeks | High |

## Active Investigations

| Issue | Location | Self % | Phase | Est. Gain | Risk | Status |
|-------|----------|--------|-------|-----------|------|--------|
| HeuristicBot dominates Phase 1 | `heuristic.rs:747` | 41.7% | 1 | ~2× via caching | Low | 🔍 Investigating |
| Baseline re-evaluated per genome | `evaluator.rs:468` | 49.5% (indirect) | 1 | 50% sim reduction | Low | 🔍 Investigating |
| `forward()` dominates Phase 0 | `wann_network.rs:173` | 63.9% | 0 | 2-4× via batched SIMD | Med | 🔍 Investigating |
| Game engine card resolution | `engine.rs` + `simulator.rs` | 23.3% | 1 | 1.3× | Med | ⚠️ Deferred |
| CSV unbuffered write | `train.rs:449` | ~0% | both | 0% | Low | ⚠️ Deferred |

## Architecture: Verified Well-Optimized

- ✅ CSR-format connection storage (cache-locality)
- ✅ Zero-allocation forward pass (scratchpad reuse)
- ✅ Aggregation branch hoisted outside edge loop
- ✅ BufReader/BufWriter on all file I/O paths
- ✅ Bincode binary serialization for checkpoints
- ✅ Rayon parallel genome evaluation
- ✅ Seeded RNG for deterministic reproducibility

## Profiling Data Files

- `perf_phase0.data` — 15,342 samples, Phase 0 supervised pretraining (config: `configs/profile_phase0.toml`)
- `perf_phase1.data` — 106,508 samples, Phase 1 self-play (config: `configs/profile_phase1.toml`)
- To regenerate: `perf report -i perf_phase{0,1}.data --stdio --sort symbol -n --no-children`
