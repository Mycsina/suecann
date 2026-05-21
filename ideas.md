This is a highly sophisticated roadmap. You have correctly identified the primary bottlenecks in neuroevolution for imperfect-information games: variance in fitness evaluation, genetic drift, and premature topological convergence.

Here is a critical evaluation of your proposals, triaged by their mathematical viability and expected return on engineering effort.

## 1. High Impact & Ready to Implement

These ideas will yield the highest performance gains for the lowest architectural risk.

### Adaptive Deal Count (with a mathematical upgrade)

Your instinct to evaluate dynamically is correct, but a hardcoded 8-then-16 split is inefficient.

* **Improvement:** Implement a **Hoeffding Race** or **Successive Halving**. Evaluate all genomes on a small batch of deals (e.g., $N=4$). Compute the mean fitness $\bar{X}$ and the Hoeffding confidence bound:

$$\epsilon = \sqrt{\frac{\ln(2/\delta)}{2N}}$$

Drop any genome whose upper bound ($\bar{X} + \epsilon$) is strictly lower than the lower bound of the top $K$ genomes. Continue dealing to the survivors. This mathematically guarantees you don't drop champions while saving up to 60-70% of your evaluation compute.

### Richer Belief State Inputs

The current 18-dim state is starving the WANN of context. Adding trick number and remaining trumps is absolutely critical for late-game Sueca strategy.

* **Improvement:** Neural networks struggle with unbounded or uniquely scaled integers. You must normalize these cleanly to $[0, 1]$:
* `Trick number`: $T / 9.0$
* `Trumps remaining`: $C / 10.0$
* `Score delta`: Instead of raw team score, feed the normalized difference between your team's score and the opponent's score, mapped to $[-1, 1]$.



### Negative Shared Weights

If your weight sweep excludes negative values, your network cannot express **inhibition**—a fundamental property of biological and artificial networks.

* **Improvement:** Implement the full symmetric sweep $W \in \{-2, -1, -0.5, 0.5, 1, 2\}$. Without negative weights, a node cannot easily learn "DO NOT play this card if X is true".

### Adaptive Late-Game Search (PIMC)

* **Improvement:** Because Sueca has only 40 cards, the game tree shrinks exponentially. By Trick 7, each player has 4 cards. Instead of continuing PIMC, implement a hard switch in your Rust engine: `if cards_remaining <= 16: return full_minimax()`. A perfect-information endgame solver running on 16 cards in a Rust bitboard engine will execute in under 5 milliseconds and provide perfect terminal evaluations, instantly raising the ceiling of your baseline bot.

---

## 2. Highly Feasible, but Requires Structural Adjustments

### $\varepsilon$-Lexicase Selection

Lexicase selection is state-of-the-art for preserving specialists, but it has a hidden dependency in card games.

* **The Trap:** If Genome A and Genome B are evaluated on *different* random deals, Lexicase selection falls apart because the "test cases" are not uniform.
* **The Fix:** You must strictly enforce **Common Random Numbers (CRN)**. Before a generation begins, generate a fixed pool of $N$ specific Sueca deals. Every genome in the population must be evaluated on that exact same set of $N$ deals. Only then can you shuffle the deals to perform the Lexicase filtering.

### Adaptive Baseline Bot per Phase

Changing the baseline mid-training breaks the historical fitness delta, meaning a genome with a score of $+2.0$ in Phase 1 might suddenly drop to $-1.5$ in Phase 2, breaking any cross-generational elitism or tracking.

* **The Fix:** Instead of changing the fitness baseline, keep the baseline static for the fitness calculation, but use an **Elo Rating System** for tracking true progress. Periodically match your current WANN champion against a gauntlet of bots (Random, Weak Heuristic, Strong Heuristic, PIMC) to establish its Elo.

### Quality-Diversity (MAP-Elites)

Maintaining an archive of behavioral specialists is a fantastic idea to prevent mode collapse.

* **The Fix:** Your proposed behavioral descriptor (fraction of FORCE_HIGH vs CUT_LOW) is good. You can enhance it by crossing it with a second dimension: **Aggression** (average point value of the cards played when leading). A 2D MAP-Elites grid of `[Intent Preference] x [Aggression]` will yield a highly diverse archive.

# Other ideas

## 1. Node Co-Expression via Structural Cross-Overs

WANN evolution typically relies heavily on structural mutations (`add_node`, `add_connection`), while cross-over (mating two networks) is often discarded because aligning differing topological structures can cause functional disruption. However, in an input-to-output game environment like Sueca, discarding cross-over slows down structural discovery.

* **The Idea:** Implement a tailored topological cross-over based on **historical markings** (similar to NEAT). If Genome A has discovered a solid structural sub-network for detecting when to play a trump card (`CUT_LOW`), and Genome B has a sub-network that tracks if its partner is leading the trick, cross-over allows these two distinct topological "sub-routines" to be combined into a single individual.
* **Why it helps:** It prevents the network from having to discover complex features sequentially through purely random mutations, accelerating structural discovery.

---

## 2. Multi-Objective Feature Isolation

When evaluating a WANN on a single global fitness metric (like total game points or win rate), small topological innovations that don't immediately win games get pruned out. A single connection that correctly calculates a minor strategic nuance will likely get out-performed by the brute-force `FORCE_HIGH` heuristic early on.

* **The Idea:** Evaluate the network on independent, secondary behavioral objectives alongside game wins using your new Lexicographic Pareto selector.
* **Implementation:** Add a secondary objective tracker that counts **Strategic Cleanliness** metrics. For example, track how often the WANN plays a legal card without triggering an invalid fallback, or how often it correctly matches a trick suit when it holds that suit.
* **Why it helps:** This avoids altering the main reward signal (avoiding the pitfalls of reward shaping) while ensuring that individuals who understand the basic mechanics of cardplay are explicitly preserved in the early fronts, providing a structural foundation for advanced strategies.

---

## 3. Weight-Symmetric Activation Pairing

Because a WANN sweeps through a shared weight vector $W$, a connection can change from positive to negative depending on the current generation's sweep value. If you introduce negative shared weights, a single structure must be able to perform a useful action when $W = 1.0$ and its exact inverse action when $W = -1.0$.

* **The Idea:** Enforce structural symmetry in your mutation operator. When an `add_node` mutation occurs, consider occasionally inserting a paired node architecture using complementary activation functions (e.g., an `IDENTITY` node paired with a `NOT` node).
* **Why it helps:** This provides the topology with an implicit mechanism to handle sign changes smoothly during the weight sweep, ensuring the network can remain truly agnostic across the entire range of $W \in \{-2.0, \dots, 2.0\}$.

---

## 4. Co-Evolutionary Opponent Warmup

Training a WANN solely against fixed bots (`RandomBot` or fixed `HeuristicBot`) can lead to structural over-fitting, where the network learns to exploit the specific weaknesses of that baseline rather than learning true Sueca strategy. Conversely, jumping straight into self-play can cause the population to stall because no individual is strong enough to provide a meaningful training signal.

* **The Idea:** Implement a **Co-evolutionary Hall of Fame** mixed with a structured rollout pipeline.
* **Implementation:**
1. Every 20 generations, save the absolute best WANN champion to an archive.
2. When evaluating the current generation, sample their opponents from a mixed pool: 50% from the historical Hall of Fame and 50% from your optimized Rust PIMC baseline.


* **Why it helps:** The historical WANNs provide an accessible, evolving ladder for the population to climb, while the Rust PIMC solver anchors the evaluation to a high-fidelity strategic baseline, preventing the population from drifting into weak cyclical strategies.