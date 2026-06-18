# WANN Evolution & Sueca AI Roadmap

This document tracks completed milestones and the live roadmap, triaged by impact vs.
engineering effort. **Last updated 2026-06-19.**

---

## Status Snapshot (2026-06-19)

The canonical champion **v6** (`checkpoints/production/2026-06-14-2`) beats the strong
EliteHeuristic baseline **52.1% ± 1.8%** (n=3000) with only 29 hidden gates. Strength is
now **resolver-floored**: every intent resolves to an Elite-flavoured card, so a collapsed
(single-intent) policy merely *ties* Elite, and the realisable headroom above Elite is only
**≈8 points**. The WANN effectively chooses between near-duplicate behaviours.

**Strategic direction (post-presentation):** interpretability is no longer a hard
constraint. The next major push is to **redesign the resolver / action space** so the
network has real expressive power and a much higher ceiling (see §3). Supporting work
(richer belief features, weight fine-tuning) is ranked in §1.

---

## Completed Milestones [IMPLEMENTED]

### 1. Phased Supervised Curriculum (WANN Warmup)
Two-phase training: Phase 0 trains topologies against an offline PIMC expert dataset by
classification accuracy (zero rollout variance, forces structural connectivity); Phase 1
evolves via self-play RL rollouts. Drastically improves initial topology search.

### 2. Legal Intent Masking & Natural Class Distribution
Legal-intent masking integrated into the Rayon accuracy loop (illegal actions masked before
argmax). The original "perfect 10k-per-intent balance" was **superseded** (June 14): under
the styled resolver the decision is naturally near-binary, so we now generate with
`--soft-balance-min-ratio 0.0` and `use_class_weighting = false` (see README dataset notes).

### 3. Adaptive Late-Game Search (Endgame Minimax Switch)
PIMC hard-switches to full minimax once ≤16 cards remain (`trick_number >= 6`), giving 100%
accurate, sub-5ms late-game evaluations.

### 4. Polymorphic Oracle → Styled Resolver (3 Archetypes)
Replaced the original 5 penalty-based intents with always-legal archetypes resolved
contextually. **Superseded June 14** by the *styled resolver*: MIN_FORCE was removed
(subsumed by EFFICIENT_WIN), leaving MAX_FORCE / EFFICIENT_WIN / EQUITY_BUILDER, each a
styled deviation *around* the Elite policy. This raised every intent to ≈Elite individually
(48/47/46% vs 20/37/25%), removing the Phase-1 collapse basin — see §3 for the cost.

### 5. Hot-Path Performance Optimizations (6× Phase 1 Speedup)
13 targeted Rust optimizations (lookup tables, branchless scoring, BinaryHeap toposort, TT
move ordering, two-pointer crossover, scoped innovation locks). Phase 1 gen time 60s → 9.7s.

### 6. Rollout Teacher & v6 Champion (June 14–15)
`solve_pimc_rollout` finishes each determinized world with **Elite playouts** — supra-Elite
labels (62% vs Elite) at ~1000× lower cost than deep alpha-beta. Training on it produced the
v6 champion: **iso-strength with v5 (52.1% vs 52.7%) but 4.5× fewer hidden gates** (29 vs
132). Key finding: a supra-Elite teacher does **not** raise the benchmark because strength is
resolver-floored — it buys a *simpler* champion, not a stronger one.

### 7. Quality-Diversity Archive (MAP-Elites)
10×10 grid archiving behavioral specialists (intent preference × aggression). Non-empty cells
sampled as Phase-1 opponents, preventing mode collapse.

### 8. Co-Evolutionary Opponent Sampling
Phase-1 opponents/partners sampled from a mixed pool (HeuristicBot / HOF / MAP-Elites),
providing a co-evolutionary ladder and preventing baseline overfitting.

### 9. Differential Evolution Weight Optimization (`optimize-weights`)
The **DE half** of the old "Fixed-Topology Fine-Tuning" idea is shipped (`optimize.rs`):
freeze a champion's topology, evolve independent per-connection continuous weights in
[−2, 2] (pop=50, F=0.5, CR=0.7). The CMA-ES + soft-threshold half remains — see §2.

### 10. Structural Tabu Veto List (odNEAT)  ← *was Tier 1*
Two-level tabu filtration: static invariants (no cycles, no input→input) compiled inline;
a dynamic FIFO queue (size 1000) of degraded mutation paths skips redundant evaluations.

### 11. Modular Lead/Follow Brains (L-NEAT)  ← *was Tier 1*
Decision space split into co-evolving Lead and Follow populations, routed per card-play slice
by `BeliefFeature::AmILeading`. Reduces strategic entropy, accelerates search.

### 12. PFS-NEAT Zero-Connection Start
Populations begin with 0 active connections; only structural mutations trigger 2-stage PFS
validation (quick K=25, then full sample for borderline cases). Guards against input noise.

---

## 1. Live Roadmap (impact / effort)

| Tier | Item | Impact | Effort | Status |
|------|------|--------|--------|--------|
| **S** | **Resolver / action-space overhaul** (§3) | High (raises the ceiling) | High — uncharted | **PLANNED** |
| **A** | Probabilistic opponent & partner belief features | Med-High | Medium | Proposed |
| **B** | CMA-ES fixed-topology fine-tune + soft-threshold annealing (§2) | Medium | Medium | Half-done (DE shipped) |
| **B** | SNAP-NEAT — adaptive mutation-operator probabilities | Low-Med | Medium | Proposed |
| **C** | Cascade-NEAT — frozen incremental hidden nodes | Low-Med | Med-High | Backlog |
| **C** | Extended modular brains (split by trick-phase / "can I win") | Low-Med | Medium | Backlog |
| **Skip** | IFSE-NEAT | Low | Medium | Overlaps shipped PFS-NEAT |
| **Skip** | True partner signaling / conventions | Low/uncertain | High | No comms phase; team fitness already rewards coordination — fold into belief features (Tier A) |

---

## 2. Fixed-Topology Fine-Tuning (DE shipped; CMA-ES remains)

The Differential Evolution path is implemented (`optimize-weights`, milestone §9). The
remaining unbuilt half is **CMA-ES with soft-threshold annealing**, which is the stronger
weight optimizer for this rugged, discrete landscape:

- Temporarily replace each hard `THRESHOLD` node with a parameterized steep sigmoid
  `SoftThreshold(x) = 1 / (1 + e^{-k(x-0.5)})`, intercepted in the Rust inference path.
- Initialize `k = 10` (rounds off the cliffs → continuous gradient for CMA-ES to read).
- Run CMA-ES ~30 gens to orient the covariance matrix over the active weights, then anneal
  `k: 10 → 50 → 100` over ~20 gens to drive the sigmoids back to razor-sharp step functions.
- Hard-freeze weights and snap back to raw `THRESHOLD` gates for zero-latency execution.
- Keep mutation step size small (σ ≈ 0.05) so the optimizer doesn't scramble the strategic
  balance by tripping over the digital cliffs.

This is a **cheap squeeze** once a topology stabilizes — most infra exists; only the
soft-threshold node mode and a CMA-ES driver are new.

---

## 3. Resolver / Action-Space Overhaul [PLANNED — Tier S]

**Problem.** `select_card_styled` is `select_card_heuristic` (the full Elite policy) plus a
handful of narrow conditional dials. All three intents are ~95% identical code, so the WANN
chooses between near-duplicates; the oracle ceiling sits at only Elite+≈8. EQUITY_BUILDER is
nearly unused in the follow split, and MAX_FORCE is uniquely best ~0–1% of the time.

**Goal.** Give the network a strong, well-separated action vocabulary whose *oracle envelope*
(best-intent-per-state) is far above Elite, so good context-dependent selection yields large
gains. Interpretability is relaxed; the new north stars are raw strength and user-facing
features (e.g. a trick-end teacher paradigm).

**Key methodology — measure the ceiling before training.** For any candidate action set,
compute the oracle envelope with the rollout teacher (no WANN): in each decision state, score
every primitive's resolved card by PIMC EV and take the best. The envelope's win-rate vs
Elite is the design's ceiling and is computable in minutes. Only train WANNs on designs whose
envelope clears Elite by a wide margin.

**Candidate directions (full plan to follow):**
1. **Richer discrete primitives**, with separate lead-vs-follow vocabularies (leverages the
   existing modular brains).
2. **Continuous control dials** (aggression / risk / trump-preservation / partner-trust)
   parametrizing one flexible resolver.
3. **Search-backed resolver** — the network biases a 1–2 ply rollout/eval rather than picking
   a hand-crafted tactic (strongest, most expensive).
4. **Direct card scoring** — the network scores (state, candidate-card) pairs and we argmax
   over legal cards, dissolving the resolver bottleneck entirely (most invasive, highest
   ceiling, fully drops interpretability).

A detailed, staged plan with hypotheses, kill/go criteria, and effort estimates is being
developed for this milestone.
