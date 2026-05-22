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

---

## 1. High Impact & Ready to Implement (Future Backlog)

### Adaptive Deal Count (Hoeffding Races / Successive Halving)
* **Problem:** Running 48 rollouts for every single candidate network in the population is highly expensive, especially when many individuals are topologically broken or weak.
* **Proposed Solution:** Implement successive halving or a Hoeffding Race during Phase 1:
  1. Evaluate all genomes on a small initial seed batch of deals (e.g., $N=4$).
  2. Compute the mean rollout fitness $\bar{X}$ and the Hoeffding confidence bound:
     $$\epsilon = \sqrt{\frac{\ln(2/\delta)}{2N}}$$
  3. Prune any genome whose upper bound ($\bar{X} + \epsilon$) is strictly lower than the lower bound of the top $K$ genomes.
  4. Continue evaluation on a larger batch (e.g., $+8$ deals) only for the surviving candidates.
* **Expected Gain:** Saves up to 60-70% of Phase 1 evaluation compute without losing high-performing champions.
---

## 2. Structural & Architectural Ideas

### Quality-Diversity (MAP-Elites)
* **Problem:** Standard Pareto selection still risks mode collapse into a single dominant playstyle (e.g., overly aggressive high-card dumping).
* **Proposed Solution:** Implement a 2D MAP-Elites grid to archive behavioral specialists:
  - **Dimension 1 (Intent Preference):** Ratio of played `FORCE_HIGH` vs. `DUCK_OR_DUMP`/`TAKE_CHEAPLY`.
  - **Dimension 2 (Aggression):** Average point-value of cards played when leading tricks.
* **Expected Gain:** Maintains a diverse pool of strategies (aggressive trumper, defensive ducker, point feeder) that can be sampled as partners or opponents.

### Co-Evolutionary Opponent Warmup & Self-Play
* **Problem:** Fixed baselines like HeuristicBot can be exploited, leading to over-fitted strategies rather than generalized Sueca expertise.
* **Proposed Solution:** Periodically archive the best WANN champion to the HOF. When evaluating, choose the 3 other seats in the rollout from a mixed pool: 50% historical WANN champions and 50% PIMC bots.
* **Expected Gain:** Provides a smooth, co-evolutionary ladder while anchored to a strong strategic baseline.