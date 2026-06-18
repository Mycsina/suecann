# References

Literature and projects that informed the design of this system.

## Core Architecture

### Weight Agnostic Neural Networks
- **Paper**: Gaier, A. & Ha, D. (2019). *Weight Agnostic Neural Networks*. NeurIPS 2019.
- **URL**: https://weightagnostic.github.io/
- **What we took**:
  - The core WANN concept: evolve topology with a single shared weight, no per-connection weight optimization.
  - Multi-objective Pareto ranking on (performance, complexity) — 80% rank by performance+simplicity, 20% by performance+max_performance. Implemented in `population.py:_pareto_rank()`.
  - Weight sweep evaluation over multiple W values (`{0.5, 1.0, 2.0}`) rather than training on a single weight. Implemented in `fitness.py:WannBotSweep`.
  - Minimal topology initialization with structural mutations (add node, add connection, change activation).

### NEAT (NeuroEvolution of Augmenting Topologies)
- **Paper**: Stanley, K.O. & Miikkulainen, R. (2002). *Evolving Neural Networks through Augmenting Topologies*. Evolutionary Computation, 10(2), 99-127.
- **What we took**:
  - Speciation via compatibility distance to protect structural innovations. Implemented in `species.py`.
  - Innovation numbers for historical gene tracking and crossover alignment.
  - Complexification from minimal topologies.
  - Seeding initial populations with known-good topologies (§4.2) — adapted for Sueca heuristics in `population.py:SEED_STRATEGIES`.

## Fitness & Evaluation

### Potential-Based Reward Shaping
- **Paper**: Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under reward transformations: Theory and application to reward shaping*. In Proceedings of the Sixteenth International Conference on Machine Learning (ICML 1999) (pp. 278-287).
- **What we took**:
  - Replacing raw episodic/per-trick card points with sequential step-potential deltas ($\Delta\Phi = \Phi(s') - \Phi(s)$) to mitigate temporal credit assignment.
  - Policy invariance guarantees under non-discounted settings ($\gamma = 1$), ensuring that maximizing our heuristic potential deltas mathematically aligns with maximizing the terminal game score without introducing reward exploitation loops.

### Imperfect Information World Rollouts (Static Determinization)
- **Paper**: Frank, I., & Basin, D. (1998). *Search in games with incomplete information: A case study using bridge card play*. Artificial Intelligence, 100(1-2), 87-123.
- **What we took**:
  - Generating a frozen world-pool of 10 card distributions matching the initial imperfect information state once at match start.
  - Tracking and masking out played cards from these pre-cached hands to evaluate mid-game state potentials without invoking constraint solvers on every turn, realizing a 99% reduction in solver overhead while preserving imperfect-information constraints.

### AIVAT (Action-Informed Value Assessment Tool)
- **Paper**: Burch, N., Schmid, M., Morav\v{c}\'{i}k, M., Morrill, D., & Bowling, M. (2018). *AIVAT: A New Variance Reduction Technique for Agent Evaluation in Imperfect Information Games*. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence (AAAI 2018) (pp. 949-956).
- **URL**: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16907
- **What we took**:
  - The principle of using **control variates** to reduce evaluation variance in imperfect-information card games. We implemented a simplified version: Common Random Numbers (CRN) — playing the same deal with a baseline bot in the same seat to compute delta-fitness. Implemented in `duplicate_loop.py:evaluate_genome_delta_on_deals()` and `fitness.py:evaluate_genome()`.
  - The insight that raw game outcomes are dominated by card-luck variance, and that comparing against a baseline on the same cards isolates strategic skill.

### Common Random Numbers (CRN)
- **Reference**: Law, A.M. & Kelton, W.D. (2000). *Simulation Modeling and Analysis*, 3rd ed., McGraw-Hill.
- **What we took**:
  - Standard variance reduction technique from stochastic simulation. By using the same random seed (same cards, same opponents) for both the genome and baseline bot, the variance of the difference estimator is dramatically lower than evaluating them independently.

### Duplicate Bridge IMP Scoring
- **Reference**: World Bridge Federation scoring rules.
- **What we took**:
  - The concept of **duplicate evaluation**: comparing agents on the exact same deal configuration to eliminate card-luck. Our seat-rotation system (`duplicate_loop.py:rotate_seats()`) ensures each genome plays all 4 positions on each deal, analogous to how duplicate bridge ensures both partnerships play the same cards.

### Temporal Difference Learning & TD-Gammon
- **Paper**: Tesauro, G. (1995). *Temporal Difference Learning and TD-Gammon*. Communications of the ACM, 38(3), 58-68.
- **What we took**:
  - Informs our rollout-based policy improvement paradigm: using rollouts under baseline policies to estimate positions and train logic network evaluations.

## Selection & Evolution

### Rank-Based Selection
- **References**:
  - Baker, J.E. (1985). *Adaptive Selection Methods for Genetic Algorithms*. In Proceedings of the First International Conference on Genetic Algorithms (ICGA 1985) (pp. 101-111).
  - Whitley, D. (1989). *The GENITOR Algorithm and Selection Pressure: Why Rank-Based Allocation of Reproductive Trials is Best*. In Proceedings of the Third International Conference on Genetic Algorithms (ICGA 1989) (pp. 116-121).
- **What we took**:
  - Converting raw fitness values to normalized ranks before selection. This makes selection robust to noisy fitness — a genome at rank 1 vs rank 50 is always clearly distinguishable even when raw fitnesses differ by 0.01. Implemented in `population.py:_rank_values()`.

### Lexicase Selection (future reference)
- **Paper**: La Cava, W., Helmuth, T., Spector, L., & Moore, J.H. (2019). *A probabilistic and multi-objective analysis of lexicase selection and ε-lexicase selection*. Evolutionary Computation, 27(3), 377-402.
- **Status**: Not yet implemented. Noted in `ideas.md` as a potential upgrade to tournament selection for maintaining behavioral diversity.

## Curriculum Learning

### Adaptive Opponent Scheduling
- **References**:
  - Narvekar, S. et al. (2020). *Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey*. JMLR, 21(181), 1-50.
  - Soltoggio, A., Bullinaria, J. A., Mattiussi, C., Dürr, P., & Floreano, D. (2008). *Evolutionary advantages of neuromodulated plasticity in dynamic, reward-based scenarios*. In Proceedings of the Eleventh International Conference on Artificial Life (ALIFE XI) (pp. 569-576).
- **What we took**:
  - Gating curriculum phase transitions on **population performance** rather than fixed generation counts. The population only advances to harder opponents when it has demonstrated competence against the current difficulty level. Implemented in `train.py:_determine_phase()`.
  - The "flow" principle: opponents should be hard enough to provide selective pressure but not so hard that all genomes score zero.

## Bloat Control

### Parsimony Pressure / Multi-Objective Complexity Control
- **References**:
  - Poli, R. (2003). *A Simple but Theoretically Motivated Method to Control Bloat in Genetic Programming*. EuroGP 2003 (pp. 204-217).
  - Deb, K. (2001). *Multi-Objective Optimization using Evolutionary Algorithms*. Wiley.
- **What we took**:
  - Using network complexity (number of enabled connections) as a second objective alongside fitness, following the WANN paper's approach. Simpler networks that perform equally well are preferred, preventing bloat. Implemented in `population.py:_pareto_rank()`.

## Advanced Evolutionary Search & Feature Selection

### PFS-NEAT (Progressive Feature Selection NEAT)
- **Paper**: Whiteson, S., Stone, P., Stanley, K. O., Miikkulainen, R., & Kohl, N. (2005). *Automatic Feature Selection in Neuroevolution*. In Proceedings of the 2005 conference on Genetic and evolutionary computation (GECCO 2005) (pp. 1225-1232).
- **What we took**:
  - Evolving networks by starting with zero active connections (an empty connection footprint).
  - Selectively introducing connection mutations to input features and verifying performance lift, guarding the network's topology from input noise.

### SNAP-NEAT & Tabu Search in Neuroevolution
- **Papers**: 
  - Glover, F. (1989). *Tabu Search—Part I*. ORSA Journal on Computing, 1(3), 190-206.
  - Silva, F., Urbano, P., Correia, L., & Christensen, A. L. (2015). *odNEAT: An Algorithm for Decentralized Online Evolution of Robotic Controllers*. Evolutionary Computation, 23(3), 421-449.
- **What we took**:
  - Two-level Tabu Veto filtration. Static rules enforce strict invariants (no cycles, no input-to-input connections) at compile-time/inline.
  - A dynamic, rolling FIFO queue stores recently rejected or toxic connection paths that degraded fitness, preventing redundant evaluations of identical bad mutations.

### L-NEAT & Multi-Brain Partitioning
- **Paper**: Reisinger, J., Stanley, K. O., & Miikkulainen, R. (2004). *Evolving Reusable Neural Modules*. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO 2004) (pp. 69-81).
- **What we took**:
  - Splitting a complex decision space into localized, low-entropy game states optimized by specialized sub-networks (modular brains).
  - Implementing dynamic brain routing resolved per card play slice based on the `Am_I_Leading` feature.

## Imperfect-Information Search & EV-Model Quality

Gathered 2026-06-19 while designing the resolver/action-space overhaul (see
`docs/superpowers/specs/2026-06-19-resolver-overhaul-design.md`). These inform whether we
can trust the rollout-PIMC teacher whose EV labels anchor the "oracle envelope" methodology.

### Understanding when PIMC is trustworthy
- **Paper**: Long, J. R., Sturtevant, N. R., Buro, M., & Furtak, T. (2010). *Understanding the
  Success of Perfect Information Monte Carlo Sampling in Game Tree Search*. AAAI 2010.
- **What it says**: three measurable properties predict PIMC quality vs. an optimal player —
  **leaf correlation** (sibling worlds reach similar terminal values → averaging across
  determinizations is reliable; high favors PIMC), **bias** (systematic error from the
  uniform-play / perfect-information-in-each-world assumption; low favors PIMC), and
  **disambiguation factor** (how quickly hidden information is revealed; high favors PIMC).
- **Implication for us**: Sueca is a *favorable* PIMC regime — all plays are public and voids
  reveal fast (high disambiguation, high leaf correlation), so the rollout teacher's error is
  concentrated in the **early tricks**; the late game is near-exact (consistent with our
  ≤16-card minimax switch). We use these as a quantitative per-phase **trust map** for the
  teacher and calibrate against exact endgame minimax where truth is computable.

### Better determinization via policy-based inference (the teacher-quality lever that fits us)
- **Paper**: Rebstock, D., Solinas, C., Buro, M., & Sturtevant, N. R. (2019). *Policy Based
  Inference in Trick-Taking Card Games*. IEEE CoG 2019.
- **What it says**: the reach probability of a world `s` in an information set is the product of
  the *other players'* action probabilities along its history under a policy model,
  `η(s|I) = Π_{h·a ⊑ s} π(h,a)` (our own actions = 1; chance = dealing odds). Sample worlds
  from this posterior instead of uniformly. Vastly improves the True State Sampling Ratio and
  raises Skat strength (+0.6–2.3 tournament pts/game over prior card-location inference), with
  gains mostly on **defense**; costs ~5× more per move.
- **Critical caveat (we must heed)**: better inference is **not** monotonically better play — a
  "cheating" variant placing all mass on the true world played *worse* than ordinary inference
  in non-null games (inference × strategy-fusion interaction). Any inference upgrade must be
  empirically validated, not assumed.
- **What we take**: use **Elite as the policy model** `π` (no human data needed; our rollouts
  already finish under Elite), treat policy-based determinization as an *optional, validated*
  teacher upgrade, and reuse its per-world posterior as a source for opponent/partner belief
  features (the parked opponent-modeling lever).
- **Precursors**: Solinas, C., Rebstock, D., & Buro, M. (2019), *Improving Search with
  Supervised Learning in Trick-Based Card Games*, AAAI 2019 (NN card-location predictions to
  bias sampling; independence assumption that PI removes). Buro, M., Long, J. R., Furtak, T., &
  Sturtevant, N. R. (2009), *Improving State Evaluation, Inference, and Search in Trick-Based
  Card Games*, IJCAI 2009 (expert Skat via learned state eval + inference + search — direct
  prior art for our learned-EV / card-match approach).

### Strategy-fusion fixes — deprioritized for Sueca (with reason)
- **Papers**: Arjonilla, J., Saffidine, A., & Cazenave, T. (2024). *Perfect Information Monte
  Carlo with Postponing Reasoning* (EPIMC), arXiv:2408.02380. Cazenave, T. & Ventos, V.
  (2019/2021). *The αµ Search Algorithm for the Game of Bridge*.
- **What they say**: EPIMC postpones the perfect-information leaf evaluator to depth `d` and
  solves the depth-`d` subgame with a fusion-free infostate algorithm (Information Set Search
  or CFR+); `d=1` recovers PIMC, and increasing `d` provably never increases strategy fusion.
  αµ tackles strategy fusion *and* non-locality and beats PIMC in Bridge.
- **Why deprioritized**: EPIMC's gains concentrate in *private-observation* games; on their
  *public-observation* trick-taking "Card Game" postponing gave ~no benefit across depths.
  Sueca is public-observation / high-disambiguation, so strategy fusion is mild and these
  upgrades are expected to be low-value. Kept only as a last resort if the trust map
  surprises us.

### Frontier EV models — deliberately not taken
- **Papers**: Brown, N. et al. (2020), *Combining Deep RL and Search for Imperfect-Information
  Games* (ReBeL); Moravčík, M. et al. (2017), *DeepStack*; Schmid, M. et al. (2021),
  *Player of Games*.
- **Why not**: these learn value/policy functions over **public belief states** with search and
  converge to Nash — but for **2-player zero-sum** games. Sueca is 4-player 2-v-2 *team* play,
  so the equilibrium machinery does not transfer. Noted as the EV-model frontier and the reason
  we stay with a PIMC-family teacher plus neuroevolution.

