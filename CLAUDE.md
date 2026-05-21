# CLAUDE.md — Agent Instructions for Sueca WANN

## Project Overview

Neurosymbolic AI that evolves Weight-Agnostic Neural Networks (WANNs) to play **Sueca** (Portuguese trick-taking card game). Networks use logical gates instead of traditional activations, output abstract play intents (not cards), and are compiled into human-readable IF/THEN rules.

## Project Documentation

- **`reference.md`** — Literature references with specific ideas taken from each paper.
- **`ideas.md`** — Future improvement paths and untested ideas.
- **`TODO.md`** — Phase-level task checklist.

## Tooling

- **Python 3.13** (stable GIL-enabled), managed by `uv`
- **Testing**: `uv run pytest tests/ -v`
- **Dependencies**: numpy, graphviz, seaborn, matplotlib, pytest

## Sueca Rules (Critical)

- **Deck**: 40 cards (standard 52 minus 8s, 9s, 10s). 4 suits × 10 ranks.
- **Rank order** (high→low): **A > 7 > K > J > Q > 6 > 5 > 4 > 3 > 2**. Note: 7 (manilha) is second-highest.
- **Point values**: A=11, 7=10, K=4, J=3, Q=2, rest=0. Total = **120 per deal**.
- **Teams**: Seats 0 & 2 vs Seats 1 & 3 (partners sit opposite).
- **Turn order**: Counter-clockwise: 0 → 3 → 2 → 1 → 0.
- **Follow-suit rule**: MUST play a card of the led suit if you have one. If void, play anything.
- **Trick winner**: Highest trump wins. If no trump played, highest of led suit wins. Off-suit non-trump cards never win.
- **Game-point tiers**: 61–90 pts = 1 game pt, 91–119 = 2, 120 (sweep) = 4, 60–60 = 0.
- **Void tracking**: When a player doesn't follow suit, all players observe they're void in that suit. This is public information.

## Architecture

```
Belief State (18 floats) → WANN (logical gates) → Oracle Intent (5 outputs) → Legal Subsystem → Card
```

### Belief State (18 inputs, all in [0,1])

| # | Field | Type | Normalization |
|---|-------|------|---------------|
| 0 | Has_Led_Suit | Bool | |
| 1 | Has_Trump | Bool | |
| 2 | Led_Suit_Power | Float | max rank in led suit / 9.0 |
| 3 | Trump_Power | Float | max rank in trump / 9.0 |
| 4 | Hand_Point_Density | Float | hand points / remaining game points |
| 5 | Am_I_Leading | Bool | 1st to play in trick |
| 6 | Am_I_Last_To_Play | Bool | 4th to play |
| 7 | Is_Partner_Winning | Bool | |
| 8 | Trick_Point_Value | Float | trick points / 44 |
| 9 | Has_Trick_Been_Cut | Bool | trump played when led suit ≠ trump |
| 10 | Partner_Void_Led | Bool | |
| 11 | Partner_Void_Trump | Bool | |
| 12 | Any_Opp_Void_Led | Bool | either opponent |
| 13 | Any_Opp_Void_Trump | Bool | either opponent |
| 14 | Led_Suit_Ace_Played | Bool | in previous tricks |
| 15 | Led_Suit_7_Played | Bool | in previous tricks |
| 16 | Trump_Ace_Played | Bool | |
| 17 | Game_Pts_Remaining | Float | unplayed points / 120 |

### Oracle Intents (5 outputs)

| ID | Intent | Action | Illegal When |
|----|--------|--------|-------------|
| 0 | DUCK_OR_DUMP | Lowest legal card | Never (fallback) |
| 1 | TAKE_CHEAPLY | Min card that beats current winner | Can't beat winner |
| 2 | FORCE_HIGH | Highest power card | Never |
| 3 | FEED_PARTNER | Highest point-value card | Never (strategic errors punished by fitness) |
| 4 | CUT_LOW | Lowest trump | Holds cards of led suit |

Illegal intent → fallback to DUCK_OR_DUMP + Oracle Tax penalty.
When WANN outputs tie (e.g. all zeros), a random intent is chosen among the tied maximums (not deterministic argmax).

### WANN Constraints

- **Gene representation**: Connection genes `[5,N]` (innovation, src, dst, sign ∈ {+1,−1}, enabled). Node genes `[4,M]` (id, type, activation_fn, aggregation_fn).
- **Initialization**: 18 input + 1 bias + 5 output nodes. 10% of population seeded with known Sueca heuristic strategies, rest get random connections.
- **Sign-only weights**: Connections carry only a sign (+1 or −1), not a learned weight. A shared weight W is used for evaluation. sign=-1 inverts the signal (1.0 - x) before aggregation.
- **Aggregation functions** (3 only): SUM=0, MIN(AND)=1, MAX(OR)=2. **No MEAN** — it causes float-precision issues at the THRESHOLD boundary.
- **Activation functions** (3 only): IDENTITY=0, NOT=1, THRESHOLD=2. **No SIGMOID** — it breaks IF/THEN rule extraction.
- **All node outputs clamped to [0, 1]**.
- **Shared weight sweep**: Evaluate each topology at W ∈ {0.5, 1.0, 2.0}, average fitness across all three weights for true weight-agnostic evaluation.

### Seed Strategies (Initial Population)

10% of the initial population is seeded with genomes encoding known Sueca heuristics:
- **Aggressive**: BIAS → FORCE_HIGH (always play strongest card)
- **Take Cheaply**: BIAS → TAKE_CHEAPLY
- **Partner Aware**: duck when partner wins, else attack
- **Trump Cutter**: cut when void in led suit and have trump
- **Feeder**: feed points to winning partner
- **Lead Attacker**: force high when leading
- **Last Taker**: take cheaply when last to play
- **Combined**: partner-aware + position-aware

### Evolution

- **Duplicate deals**: 16 deals per generation × 4 seat rotations = 64 games/genome. Deals are **re-seeded each generation** (`seed=gen`) to prevent overfitting.
- **Delta-fitness**: Each genome is compared against a RandomBot baseline on the exact same deal/seat/opponents (Common Random Numbers). Fitness = mean(genome_card_points − baseline_card_points) + Oracle Tax. This eliminates deal-luck variance.
- **Oracle Tax warm-up**: Penalty for illegal intents starts at −0.25 (gen 0), ramps linearly to −3.0 by `curriculum_gens`.
- **Rank-based selection**: Raw fitness converted to normalized ranks before tournament selection for noise robustness.
- **Multi-objective Pareto ranking**: 80% of the time, rank by (performance, simplicity) Pareto front with lexicographic tie-breaking (using Min-Max normalized fitness); 20% by performance only. Prevents bloat while maintaining selection pressure.
- **Adaptive curriculum**: Phase transitions gated by population performance:
  - Phase 0 → 1: median delta > 0.5 (beating RandomBot)
  - Phase 1 → 2: median delta > 0.0 (beating HeuristicBot)
  - Phase 2 → 3: HoF has ≥ 5 entries with positive fitness
- **Hall of Fame**: Frozen champion archive prevents Red Queen Effect.
- **Mutations**: Add node, add connection, toggle connection, flip sign, change activation, change aggregation. No weight mutation.

## Code Conventions

- All source in `src/`, tests in `tests/`.
- Imports use `from src.engine.cards import ...` style.
- Every module gets a corresponding `tests/test_<module>.py`.
- Tests must be thorough — test invariants (e.g., total points = 120), edge cases, and boundary values. Not just happy paths.
- Use `numpy.random.Generator` (not legacy `numpy.random`), pass seeds explicitly for reproducibility.
- Type hints on all function signatures.
- Docstrings on all public functions and classes.

## Common Pitfalls

1. **Never leak opponent hand data** into visible state or belief vector.
2. **Rank ordering is NOT standard** — 7 beats K in Sueca. Use the `Rank` IntEnum values, not card face values.
3. **Partner = (seat + 2) % 4**, not seat ± 1.
4. **Counter-clockwise**: after seat 0, it's seat 3, not seat 1.
5. **Void tracking is per-suit**: a player void in hearts may still have diamonds.
6. **Duplicate deals must differ across generations** — same seed within a gen for fairness, different seed between gens to prevent memorization.
7. **argmax tie-breaking**: When WANN outputs tie, use `rng.choice` among tied maximums, NOT deterministic `np.argmax` (which always picks index 0 = DUCK_OR_DUMP).
8. **Delta-fitness baseline bot must see the same cards**: The baseline plays the exact same seat rotation with the same deal to ensure valid comparison.
