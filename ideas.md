# Ideas — Future Improvements

Ideas for future exploration, roughly ordered by expected impact.

## Fitness & Evaluation

### Adaptive baseline bot per curriculum phase
Currently the delta-fitness baseline is always `RandomBot`. Consider switching
the baseline as the WANN improves:
- **Phase 0–1**: `RandomBot` (measures "better than random")
- **Phase 2–3**: A weak heuristic bot (e.g., a simplified `HeuristicBot` that only follows suit and plays random otherwise)
- **Phase 3+**: Full `HeuristicBot` (measures "better than rules")

This progressively raises the bar, ensuring the delta signal remains informative
even when the WANN has long surpassed random play. The risk is that switching
baselines mid-training changes the fitness scale, so a smooth transition
(e.g., 50% RandomBot + 50% HeuristicBot baseline games) may work better.

### AIVAT-style control variates
Full AIVAT (Burch et al., 2018) goes beyond Common Random Numbers — it uses the
agent's own strategy to compute expected values of chance events, further
reducing variance. This requires knowing the WANN's action probabilities
at each decision point, which we can compute from the output vector. Would
provide the lowest-variance fitness estimates possible.

### Adaptive deal count
Instead of a fixed 16 deals per generation, use adaptive sampling: evaluate
all genomes on 8 deals first, then evaluate the top 30% on 16 additional deals
to confirm their ranking. This halves compute for clearly-bad genomes while
giving high-quality estimates for the ones that matter.

## Population & Selection

### Increase seed genome proportion
Currently 10% of the initial population is seeded with heuristic strategies.
If the population converges too slowly, try 20–30%. If it converges too fast
(everyone becomes a minor variant of the heuristic bot), reduce to 5%.
The NEAT literature suggests 10–30% is the sweet spot.

### ε-Lexicase selection
Instead of rank-based tournament selection, try ε-lexicase: evaluate each
genome on individual deals (not averaged), then select by shuffling deals
and filtering candidates that are within ε of the best on each deal.
This preserves specialists (genomes that excel on specific deal types)
and maintains higher diversity than tournament selection.

### Quality-Diversity (MAP-Elites)
Maintain an archive of genomes indexed by behavioral characteristics (e.g.,
intent distribution: what fraction of plays are FORCE_HIGH vs CUT_LOW).
This prevents the population from collapsing to a single behavioral mode
and provides diverse starting points for further evolution.

## Architecture & Representation

### Negative shared weights in sweep
The original WANN paper uses W ∈ {-2, -1, -0.5, +0.5, +1, +2}. We currently
only sweep positive weights {0.5, 1.0, 2.0}. Adding negative weights would
test whether the topology is truly weight-agnostic or depends on sign.
For our sign-based connection system this may behave differently — worth
investigating.

### Additional activation functions
Currently: IDENTITY, NOT, THRESHOLD. Consider adding:
- **GAUSSIAN**: `exp(-x²)` — useful for "around 0.5" detectors
- **STEP with hysteresis**: prevents oscillation at boundaries
- **ABS**: `|x|` — useful for "either extreme" detection

Each addition expands the expressible rule space but makes rule extraction harder.

### Richer belief state inputs
The 18-dim belief state may be missing important features:
- **Trick number** (0–9): strategy changes between early and late game
- **Team score so far**: risk-taking should depend on the score
- **Number of trumps remaining**: critical for deciding when to cut
- **Position in trick** (0–3 as float): more granular than Am_I_Leading/Am_I_Last

## Training Pipeline

### Multi-population island model
Run 3–5 independent populations with occasional migration (every 20 gens,
swap the top 5% of genomes between populations). This prevents all populations
from converging to the same local optimum and introduces fresh genetic material.

### Fitness shaping rewards
Add small bonuses for desirable intermediate behaviors:
- +0.1 for winning a trick with a high-point card
- +0.05 for feeding partner when they're winning
- -0.1 for playing trump when you have led suit cards

These "reward shaping" signals can bootstrap learning before the WANN
discovers them through game-point outcomes.

### Incremental complexity (phased mutation rates)
In early generations, only allow `add_connection` mutations (no `add_node`).
This forces the population to discover good direct input→output mappings
before building complex hidden-node architectures. After 30 gens, enable
`add_node` mutations.

## Interpretability & Export

### Automatic rule extraction
After training, compile the best genome into IF/THEN rules by backward-chaining
from each output. This is the interpretability payoff of using logical nodes.
Track rule complexity (number of conditions) as a quality metric.

### Strategy heatmap visualization
For each of the 18 belief state features, visualize which intents the WANN
selects as the feature varies from 0 to 1 (holding other features constant).
This shows what the network has learned without needing to extract rules.
