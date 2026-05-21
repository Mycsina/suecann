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

### AIVAT (Action-Informed Value Assessment Tool)
- **Paper**: Burch, N., Schmid, M., Szafron, D., & Bowling, M. (2018). *AIVAT: A New Technique for Agent Evaluation in Imperfect Information Games*. AAAI 2018.
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

## Selection & Evolution

### Rank-Based Selection
- **References**:
  - Baker, J.E. (1985). *Adaptive Selection Methods for Genetic Algorithms*. ICGA 1985.
  - Whitley, D. (1989). *The GENITOR Algorithm and Selection Pressure*. ICGA 1989.
- **What we took**:
  - Converting raw fitness values to normalized ranks before selection. This makes selection robust to noisy fitness — a genome at rank 1 vs rank 50 is always clearly distinguishable even when raw fitnesses differ by 0.01. Implemented in `population.py:_rank_values()`.

### Lexicase Selection (future reference)
- **Paper**: La Cava, W., Helmuth, T., Spector, L., & Moore, J.H. (2019). *A probabilistic and multi-objective analysis of lexicase selection and ε-lexicase selection*. Evolutionary Computation, 27(3).
- **Status**: Not yet implemented. Noted in `ideas.md` as a potential upgrade to tournament selection for maintaining behavioral diversity.

## Curriculum Learning

### Adaptive Opponent Scheduling
- **References**:
  - Narvekar, S. et al. (2020). *Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey*. JMLR, 21(181), 1-50.
  - Silva, F. & Christensen, A.L. (2018). *Evolutionary Advantages of Neuromodulated Synaptic Plasticity in Dynamic, Reward-Based Scenarios*. ALIFE 2018.
- **What we took**:
  - Gating curriculum phase transitions on **population performance** rather than fixed generation counts. The population only advances to harder opponents when it has demonstrated competence against the current difficulty level. Implemented in `train.py:_determine_phase()`.
  - The "flow" principle: opponents should be hard enough to provide selective pressure but not so hard that all genomes score zero.

## Bloat Control

### Parsimony Pressure / Multi-Objective Complexity Control
- **References**:
  - Poli, R. (2003). *A Simple but Theoretically Motivated Method to Control Bloat in Genetic Programming*. EuroGP 2003.
  - Deb, K. (2001). *Multi-Objective Optimization using Evolutionary Algorithms*. Wiley.
- **What we took**:
  - Using network complexity (number of enabled connections) as a second objective alongside fitness, following the WANN paper's approach. Simpler networks that perform equally well are preferred, preventing bloat. Implemented in `population.py:_pareto_rank()`.
