# CLAUDE.md — Agent Instructions for Sueca WANN

## Project Overview

Neurosymbolic AI that evolves Weight-Agnostic Neural Networks (WANNs) to play **Sueca** (Portuguese trick-taking card game). Networks use logical gates instead of traditional activations, output abstract play intents (not cards), and are compiled into human-readable IF/THEN rules.

The training pipeline is a pure-Rust binary (`sueca_wann`) that calls into the Rust game engine (`sueca_solver`). Python is used only for benchmarking, rule extraction, graphing, and the PyO3 FFI layer.

## Project Documentation

- **`reference.md`** — Literature references with specific ideas taken from each paper.
- **`ideas.md`** — Future improvement paths and untested ideas.

## Tooling

- **Python 3.13** (stable GIL-enabled), managed by `uv`
- **Rust workspace**: `src/sueca_solver` (game engine + PyO3 bindings) and `src/sueca_wann` (training binary)
- **Build Rust Module**: `cargo build -p sueca_solver --release && cp target/release/libsueca_solver.so .venv/lib/python3.13/site-packages/sueca_solver/sueca_solver.cpython-313-x86_64-linux-gnu.so`
- **Build Training Binary**: `cargo build -p sueca_wann --release`
- **Testing**: `uv run pytest tests/ -v` (Python), `cargo test --all` (Rust)
- **Linting**: `cargo clippy --all`, `uv run black .`
- **Dependencies**: numpy, pandas, graphviz, seaborn, matplotlib, pytest

## Running Training

```bash
# Build both crates
cargo build -p sueca_solver --release
cargo build -p sueca_wann --release

# Run training (creates checkpoints/YYYY-MM-DD-N/)
./target/release/sueca_wann --config configs/default.toml

# Resume from checkpoint
./target/release/sueca_wann --config configs/default.toml --resume
```

Training creates dated run folders: `checkpoints/2026-05-26-1/`, `checkpoints/2026-05-26-2/`, etc. Each contains `training_stats.csv`, `training_state.bin`, and a `genomes/` subfolder containing `best_genome_final.json`, `hof_final.json`, and snapshots.

## Running Benchmarks

```bash
# Auto-detect latest genome and run tournament
PYTHONPATH=. uv run python src/benchmark.py --deals 200

# Specify genome and output dir
PYTHONPATH=. uv run python src/benchmark.py --deals 200 --genome checkpoints/2026-05-26-1/genomes/best_genome_final.json --output-dir checkpoints/2026-05-26-1
```

## Comparing Training Runs

```bash
# Plot all runs
uv run python scripts/compare_runs.py

# Compare specific runs
uv run python scripts/compare_runs.py --runs 2026-05-26-1 2026-05-26-2
```

Saves `checkpoints/run_comparison.png` with 4 panels: fitness, delta vs HeuristicBot, species diversity, network complexity.

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
Belief State (21 floats) → WANN (logical gates) → Oracle Intent (5 outputs) → Legal Subsystem → Card
```

**Crate dependency**: `sueca_wann` → `sueca_solver`. The solver crate contains the game engine, WANN inference, PIMC search, and PyO3 bindings. The wann crate contains the NEAT evolution loop.

**Key modules in `sueca_solver`**:
- `engine.rs` — Bitboard game state, card logic, beats comparison
- `wann.rs` — CSR-format WANN inference with zero-allocation forward pass
- `evaluator.rs` — Game simulation, bot types, delta-fitness evaluation (re-exports from split modules)
- `simulator.rs` — SuecaSimulatorGame wrapper with void tracking
- `belief.rs` — Belief state encoder (21 floats from game state)
- `heuristic.rs` — Card selection heuristics, intent-to-card resolver
- `pimc.rs` — Perfect Information Monte Carlo solver with late-game minimax switch
- `search.rs` — Alpha-beta search with Zobrist hashing and transposition table
- `rng.rs` — Shared LCG random number generator
- `py_bindings/` — PyO3 FFI layer

**Key modules in `sueca_wann`**:
- `genome.rs` — Genome representation, topological sort, CSR conversion
- `population.rs` — Population management, crossover, Pareto ranking, parallel breeding
- `species.rs` — Compatibility distance, speciation (parallel distance computation)
- `mutations.rs` — NEAT mutation operators, innovation registry
- `train.rs` — Training loop, Phase 0/1 evaluation, HOF transfer
- `hall_of_fame.rs` — HOF management with sampling
- `config.rs` — TOML configuration loading

### Belief State (30 inputs, all in [0,1])

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
| 18 | Trick_Number | Float | current trick index / 9.0 |
| 19 | Trumps_Remaining | Float | unplayed trump cards / 10.0 |
| 20 | Score_Delta | Float | (our_pts − opp_pts + 120) / 240 |
| 21 | Side0_Depletion | Float | played cards of side-suit 0 / 10 |
| 22 | Side0_Ace_Played | Bool | in previous tricks |
| 23 | Side0_7_Played | Bool | in previous tricks |
| 24 | Side1_Depletion | Float | played cards of side-suit 1 / 10 |
| 25 | Side1_Ace_Played | Bool | in previous tricks |
| 26 | Side1_7_Played | Bool | in previous tricks |
| 27 | Side2_Depletion | Float | played cards of side-suit 2 / 10 |
| 28 | Side2_Ace_Played | Bool | in previous tricks |
| 29 | Side2_7_Played | Bool | in previous tricks |

### Oracle Intents (4 outputs)

| ID | Intent | Action | Strategic Meaning |
|----|--------|--------|-------------------|
| 0 | MAX_FORCE | Aggressive / control | Lead high trump/master card or play max-rank card |
| 1 | MIN_FORCE | Passive / resource saving | Lead longest suit or play lowest legal card |
| 2 | EFFICIENT_WIN | Tactical exploitation | Play min card that beats winner, or cut cheaply |
| 3 | EQUITY_BUILDER | Partnership / voids | Lead short suit or load points / cut when partner is void |

All intents are resolved to legal plays contextually by the heuristic resolver, guaranteeing 100% legality.
When WANN outputs tie (e.g. all zeros), a random intent is chosen among the tied maximums (not deterministic argmax).

### WANN Constraints

- **Gene representation**: Connection genes `[5,N]` (innovation, src, dst, sign ∈ {+1,−1}, enabled). Node genes `[4,M]` (id, type, activation_fn, aggregation_fn).
- **Initialization**: 30 input + 1 bias + 4 output nodes. 15% of population seeded with known Sueca heuristic strategies, rest get random connections.
- **Sign-only weights**: Connections carry only a sign (+1 or −1), not a learned weight. A shared weight W is used for evaluation. sign=-1 inverts the signal (1.0 - x) before aggregation.
- **Aggregation functions** (3 only): SUM=0, MIN(AND)=1, MAX(OR)=2. **No MEAN** — it causes float-precision issues at the THRESHOLD boundary.
- **Activation functions** (3 only): IDENTITY=0, NOT=1, THRESHOLD=2. **No SIGMOID** — it breaks IF/THEN rule extraction.
- **All node outputs clamped to [0, 1]**.
- **Shared weight sweep**: Evaluate each topology at W ∈ {-2.0, -1.0, -0.5, 0.5, 1.0, 2.0}, including negative weights for inhibitory rule expression. Average fitness across all six weights for true weight-agnostic evaluation.

### Training Pipeline

**Phase 0 (gens 0–`phase0_gens`): Supervised pretraining.** WANNs are trained to match PIMC expert intents on a pre-generated dataset. Fitness = classification accuracy. No game simulation.

**Phase 1 (gens `phase0_gens`–1000): Self-play evolution.** Fitness = raw game-point delta vs HeuristicBot. Partners/opponents sampled from Hall of Fame and MAP-Elites. Delta computed via Common Random Numbers on the same deals.

**Phase 0→1 HOF transfer**: HOF entries are re-evaluated under Phase 1 fitness at the transition point, preserving knowledge from supervised pretraining.

**Parallelism**: Rayon parallelizes genome→WANN conversion, speciation distance computation, Pareto domination detection, stagnation updates, and offspring generation. Innovation registry uses a Mutex for thread-safe mutation operations.

### Seed Strategies (Initial Population)

15% of the initial population is seeded with genomes encoding known Sueca heuristics:
- **Aggressive**: BIAS → FORCE_HIGH (always play strongest card)
- **Take Cheaply**: BIAS → TAKE_CHEAPLY
- **Partner Aware**: duck when partner wins, else attack
- **Trump Cutter**: cut when void in led suit and have trump
- **Feeder**: feed points to winning partner
- **Lead Attacker**: force high when leading
- **Last Taker**: take cheaply when last to play
- **Combined Basic**: partner-aware + position-aware
- **Late Trump Aggressor**: cut/force when few trumps remain
- **Score Aware**: play safe when ahead, aggressive when behind
- **Trick Point Taker**: take cheaply when trick has value and partner isn't winning
- **Void Exploiter**: force high when opponent is void in led suit
- **Full Strategic**: combines partner, position, score, and trump awareness (7 connections)

### Evolution

- **Duplicate deals**: 48 deals per generation × 4 seat rotations = 192 games/genome. Deals are **re-seeded each generation** (`seed=gen`) to prevent overfitting.
- **Delta-fitness**: Each genome compared against HeuristicBot on the exact same deal/seat/opponents (Common Random Numbers). Eliminates deal-luck variance.
- **Rank-based selection**: Raw fitness converted to normalized ranks before tournament selection for noise robustness.
- **Multi-objective Pareto ranking**: 80% of the time, rank by (performance, simplicity) Pareto front with lexicographic tie-breaking; 20% by performance only. Prevents bloat while maintaining selection pressure.
- **Hall of Fame**: Frozen champion archive (size 30). Sampled as partners/opponents during Phase 1.
- **Mutations**: Add node, add connection, toggle connection, flip sign, change activation, change aggregation. No weight mutation.

## Code Conventions

- All Python source in `src/`, tests in `tests/`.
- Rust source: `src/sueca_solver/src/` (engine + bindings), `src/sueca_wann/src/` (training).
- Python imports use `from src.X import Y` style.
- Tests must be thorough — test invariants (e.g., total points = 120), edge cases, and boundary values.
- Use `numpy.random.Generator` (not legacy `numpy.random`), pass seeds explicitly for reproducibility.
- Type hints on all Python function signatures.
- Rust functions: `#[inline(always)]` on hot-path bitboard/WANN ops.

## Common Pitfalls

1. **Never leak opponent hand data** into visible state or belief vector.
2. **Rank ordering is NOT standard** — 7 beats K in Sueca. Use the `Rank` IntEnum values, not card face values.
3. **Partner = (seat + 2) % 4**, not seat ± 1.
4. **Counter-clockwise**: after seat 0, it's seat 3, not seat 1.
5. **Void tracking is per-suit**: a player void in hearts may still have diamonds.
6. **Duplicate deals must differ across generations** — same seed within a gen for fairness, different seed between gens to prevent memorization.
7. **argmax tie-breaking**: When WANN outputs tie, use random choice among tied maximums, NOT deterministic argmax.
8. **Delta-fitness baseline bot must see the same cards**: The baseline plays the exact same seat rotation with the same deal to ensure valid comparison.
9. **Workspace target dir**: Always build from repo root (`cargo build --release -p sueca_solver`), not `--manifest-path`. The .so lives in `target/release/`, not `src/sueca_solver/target/release/`.
