# WANN Evolution & Sueca AI Roadmap

This document outlines high-impact ideas, triaged by their mathematical viability and expected return on engineering effort, along with recently completed milestones.

---

## Completed Milestones [IMPLEMENTED]

### 1. Phased Supervised Curriculum (WANN Warmup)
* **Status:** `IMPLEMENTED`
* **Implementation:** Introduced a two-phase training loop. 
  - **Phase 0 (Generations 0 to 100):** Trains WANN topologies against a static, offline expert dataset using classification accuracy. No rollout simulations are performed, dropping fitness evaluation variance to zero and forcing the evolutionary pipeline to establish structural connectivity.
  - **Phase 1 (Generations 100+):** Evolves WANNs via RL rollouts against HeuristicBot and HOF champions to learn game dynamics.
* **Impact:** Drastically improves initial topology search and input/hidden node usage.

### 2. Perfect Class Balancing & Legal Intent Masking
* **Status:** `IMPLEMENTED`
* **Implementation:**
  - Enforced a strict balance of exactly 10,000 states for each of the 5 play intents (total 50,000 states) in the offline dataset generator to avoid simplicity bias (e.g., over-predicting the frequent `DUCK_OR_DUMP` fallback).
  - Saved a `legal_masks` byte vector alongside targets.
  - Integrated legal intent masking directly into the Rust parallel Rayon loop (`evaluate_wann_accuracy`), masking out illegal actions before taking the activation argmax.
* **Impact:** Prevents evolutionary stagnation and ensures fair evaluation of legal strategic intents.

### 3. Adaptive Late-Game Search (Endgame Minimax Switch)
* **Status:** `IMPLEMENTED`
* **Implementation:** Implemented a hard-switch in the Rust PIMC engine (`solve_pimc` in `pimc.rs`): if 16 or fewer cards remain in the game (i.e. `trick_number >= 6`), the engine automatically switches to a full minimax search (40 plies lookup to the terminal states).
* **Impact:** Late-game evaluations are 100% accurate and execute in under 5 milliseconds on a single core, boosting baseline PIMC and heuristic decision accuracy.

### 4. Polymorphic Oracle (4-Archetype Intent System)
* **Status:** `IMPLEMENTED`
* **Implementation:** Replaced the original 5 fixed intents (DUCK_OR_DUMP, TAKE_CHEAPLY, FORCE_HIGH, FEED_PARTNER, CUT_LOW) with 4 always-legal archetypes resolved contextually:
  - **MAX_FORCE** — Aggressive/control: lead high or play max-rank card, chosen by point-value tie-breaking.
  - **MIN_FORCE** — Passive/resource saving: lowest legal card, preserving high cards.
  - **EFFICIENT_WIN** — Tactical exploitation: min card that beats current winner, cheapest cut when void.
  - **EQUITY_BUILDER** — Partnership/voids: feed high-point cards when partner wins, exploit opponent voids, cut when partner is void.
* **Impact:** Eliminates the oracle-tax penalty system entirely — all intents are always legal. The network learns *what* to do, and the resolver handles *how* to do it given the game state.

### 5. Hot-Path Performance Optimizations (6x Phase 1 Speedup)
* **Status:** `IMPLEMENTED`
* **Implementation:** 13 targeted optimizations across the Rust codebase:
  - **Engine layer:** `CARD_SUIT[40]`/`CARD_RANK[40]` lookup tables replacing all division/modulo; branchless team score via `(winner & 1) == 0`.
  - **WANN inference:** BinaryHeap topological sort replacing O(n²) Vec::remove; hoisted `match agg_fn` with three separate SUM/MIN/MAX loops eliminating per-edge branch misprediction.
  - **PIMC:** Pre-built unknown card pool shared across worlds; thread-local accumulator fold/reduce replacing Vec allocations.
  - **Search:** TT best-move lookup for move ordering; compile-time `const fn LCG` Zobrist tables replacing `OnceLock` runtime init.
  - **Genome:** Vec-based sorted node/conn genes with binary search; O(n+m) two-pointer merge crossover replacing HashMap set operations.
  - **Parallelism:** Thread-local innovation registry lock scope (lock held only during `get_or_create`, not entire mutation); precomputed species max-fitness for O(1) sort comparisons.
* **Impact:** Phase 1 generation time dropped from ~60s to ~9.7s (6x). Full 1200-gen production run completed in 4.7 hours (projected 18h+ without optimizations).

### 6. Deep PIMC Expert Dataset (d=4, 80 Worlds)
* **Status:** `IMPLEMENTED`
* **Implementation:** Generated the legacy `expert_states_w80_d4.npz` — 100k states from PIMC with depth 4 search and 80 worlds per state.
* **Impact:** Deeper search produces higher-quality expert labels. Phase 0 accuracy reached 44.37% (up from d=2 baseline). Combined with polymorphic oracle, enabled the network to reach statistical parity with HeuristicBot (49.2% win rate, 59.9 vs 60.1 card points).

### 7. Quality-Diversity Archive (MAP-Elites)
* **Status:** `IMPLEMENTED`
* **Implementation:** 10×10 MAP-Elites grid archiving behavioral specialists during Phase 1:
  - **Dimension 1 (Intent Preference):** Ratio of MIN_FORCE plays to total actions (conservative vs. active).
  - **Dimension 2 (Aggression):** Average point-value of cards played when leading tricks (normalized to [0,1]).
  - Each cell keeps the single best genome by fitness. Non-empty cells sampled at 50% rate when selecting opponent bots during Phase 1 self-play.
* **Impact:** Maintains a diverse pool of strategies (aggressive trumper, defensive ducker, point feeder) as training opponents, preventing mode collapse into a single dominant playstyle.

### 8. Co-Evolutionary Opponent Sampling
* **Status:** `IMPLEMENTED`
* **Implementation:** When evaluating genomes in Phase 1, opponents and partners are sampled from a mixed pool: 50% HeuristicBot, 25% HOF champions, 25% MAP-Elites specialists. This provides a co-evolutionary ladder — as the population improves, so do the opponents.
* **Impact:** Prevents overfitting to a fixed baseline. The WANN must generalize against diverse historical strategies rather than exploiting HeuristicBot-specific weaknesses.

---

## 1. Structural & Architectural Ideas

Tier 1

SNAP-NEAT
odNEAT's structural tabu list
L-NEAT

Tier 2
Cascade-NEAT
IFSE-NEAT

## 2. Delivery (LATER)
### Fixed-Topology Fine-Tuning
When we reach a good champion, freeze the topology (e.g. Gen 849) and evolve the connection weights into independent floats.

The network can fine-tune high-precision thresholds to shift its behavior from conservative to cutthroat, giving you the exact tactical edge needed to break the parity ceiling and dominate HeuristicBot decisively.

If you still want to trim the fat to make your final rule extraction look as pristine as possible, do not do it during evolution. Do it during your upcoming Fixed-Topology Fine-Tuning phase.Once you freeze the Gen 849 topology and assign independent, real-valued float weights ($w_i$) to those 50 connections, your continuous optimizer will naturally drive the weights of minor, marginal connections down toward 0.0.After fine-tuning, you can apply a simple structural threshold filter

When you launch your continuous Genetic Algorithm or CMA-ES pass to tune these weights, keep your initial mutation step size ($\sigma$) small (e.g., $\sigma = 0.05$). If your mutation steps are too large, the optimizer will constantly trip over these deep digital cliffs and scramble the delicate strategic balance of your EQUITY_BUILDER and MAX_FORCE archetypes.

Option 1: The Silver Bullet — Soft-Threshold Smooth TranslationIf you want to use CMA-ES (which you should, because its ability to learn variable correlations is unmatched), you must temporarily transform your network's activation functions into a smooth, optimization-friendly landscape.Before you pass the topology to the optimizer, intercept your THRESHOLD node calls inside your Rust engine. Replace the hard step function with a parameterized, steep sigmoid:$$\text{SoftThreshold}(x) = \frac{1}{1 + e^{-k(x - 0.5)}}$$The Optimization Loop:Initialize the scaling factor at a moderate slope, like $k = 10$. This rounds off the sharp cliffs, turning your flat plateaus into continuous slopes. Now, even a micro-mutation in weights provides a tiny shift in activation output, giving CMA-ES a pristine gradient signal to guide its covariance matrix.Run CMA-ES for 30 generations to let it rapidly scale and orient your 50 active weights.Over the final 20 generations, gradually scale $k$ up ($10 \rightarrow 50 \rightarrow 100$). This drives the sigmoids back into razor-sharp, discrete step functions.Once optimization concludes, hard-freeze the weights and snap the functions back to raw THRESHOLD gates for zero-latency execution.Option 2: Differential Evolution (DE)If you do not want to modify your low-level Rust execution nodes to handle soft sigmoids, skip CMA-ES entirely and deploy Differential Evolution (DE) (specifically the DE/rand/1/bin or DE/best/1/bin variants).Why it works here: Differential Evolution does not build a statistical distribution model like CMA-ES does. Instead, it mutates candidate vectors by taking the direct algebraic difference between other random members of the population:$$v_i = x_{\text{best}} + F \cdot (x_{r1} - x_{r2})$$The Edge: Because its step sizes are dictated by the actual distance between active individuals rather than a localized variance calculation, DE can natively "teleport" right across flat plateaus and jump over threshold cliffs. It handles rugged, discrete, and non-differentiable landscapes exceptionally well.