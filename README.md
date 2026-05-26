# 🃏 Neurosymbolic WANNs for Sueca

A neurosymbolic framework that evolves **Weight-Agnostic Neural Networks (WANNs)** to master **Sueca** (a Portuguese four-player partnership trick-taking card game). Rather than fitting continuous weights to specific tasks, this system evolves discrete network topologies using logical gates and custom aggregations. The resulting networks compile directly into human-readable decision rules (IF/THEN trees) for complete transparency.

---

## Architecture Overview

```
Belief State (30 features) → WANN (Logical Gates) → Oracle Intent (4 outputs) → Legal Subsystem → Card
```

- **Belief State**: 30 normalized floats encoding hand composition, trick state, void tracking, game progress, and side-suit depletion/honor tracking.
- **WANN**: Weight-agnostic network with sign-only connections (±1), logical aggregations (SUM/MIN/MAX), and discrete activations (IDENTITY/NOT/THRESHOLD). Evaluated across a weight sweep W ∈ {-2.0, -1.0, -0.5, 0.5, 1.0, 2.0}.
- **Oracle Intents**: 4 abstract play actions representing strategic card-play archetypes:
  - **MAX_FORCE**: Aggressive/control. If leading, lead high trump or a cash master card; otherwise, play the highest-ranking card.
  - **MIN_FORCE**: Passive/resource saving. If leading, lead from the longest non-trump suit; otherwise, play the lowest legal card of the led suit to preserve high-value cards.
  - **EFFICIENT_WIN**: Tactical exploitation. If leading, lead low trump or longest suit; otherwise, play the cheapest card that beats the current winner, or cut with the cheapest trump card when void.
  - **EQUITY_BUILDER**: Partnership-focused play. If leading, play a short suit to build voids; otherwise, load points when partner is winning, exploit opponent voids, or cut when partner is void.
- **Heuristic Resolver / Legal Subsystem**: Mapped directly to legal cards by the heuristic resolver. Because these 4 polymorphic intents are defined to resolve contextually into a legal move under all game conditions, all intent outputs are always legal. Thus, while the training pipeline maintains an **Oracle Tax** penalty mechanism, the rate of illegal intents remains at `0.0` during rollouts, rendering the tax inactive.

### Belief State Layout (30 inputs, all in [0,1])

| Index | Feature | Type | Normalization / Description |
|:---:|---|---|---|
| 0 | `Has_Led_Suit` | Bool | `1.0` if player holds at least one card of the led suit, `0.0` otherwise. |
| 1 | `Has_Trump` | Bool | `1.0` if player holds at least one trump card, `0.0` otherwise. |
| 2 | `Led_Suit_Power` | Float | Power rank of the highest card held in led suit: `max_rank_held / 9.0`. |
| 3 | `Trump_Power` | Float | Power rank of the highest trump card held: `max_rank_held / 9.0`. |
| 4 | `Hand_Point_Density` | Float | Ratio of points held in player's hand to total remaining points in play: `hand_points / remaining_points`. |
| 5 | `Am_I_Leading` | Bool | `1.0` if player is leading the trick, `0.0` otherwise. |
| 6 | `Am_I_Last_To_Play` | Bool | `1.0` if player is the 4th to play in the trick, `0.0` otherwise. |
| 7 | `Is_Partner_Winning` | Bool | `1.0` if partner is currently winning the trick, `0.0` otherwise. |
| 8 | `Trick_Point_Value` | Float | Ratio of total points currently on the table to maximum possible trick value (44): `trick_points / 44.0`. |
| 9 | `Has_Trick_Been_Cut` | Bool | `1.0` if a trump card has been played when the led suit is not trump, `0.0` otherwise. |
| 10 | `Partner_Void_Led` | Bool | `1.0` if partner is void in the led suit, `0.0` otherwise. |
| 11 | `Partner_Void_Trump` | Bool | `1.0` if partner is void in the trump suit, `0.0` otherwise. |
| 12 | `Any_Opp_Void_Led` | Bool | `1.0` if either opponent is void in the led suit, `0.0` otherwise. |
| 13 | `Any_Opp_Void_Trump` | Bool | `1.0` if either opponent is void in the trump suit, `0.0` otherwise. |
| 14 | `Led_Suit_Ace_Played` | Bool | `1.0` if the Ace of the led suit has already been played in previous tricks, `0.0` otherwise. |
| 15 | `Led_Suit_7_Played` | Bool | `1.0` if the 7 of the led suit has already been played in previous tricks, `0.0` otherwise. |
| 16 | `Trump_Ace_Played` | Bool | `1.0` if the trump Ace has already been played in previous tricks, `0.0` otherwise. |
| 17 | `Game_Pts_Remaining` | Float | Ratio of unplayed card points remaining in the deal to the total deck points (120): `remaining_points / 120.0`. |
| 18 | `Trick_Number` | Float | Index of the current trick: `trick_index / 9.0`. |
| 19 | `Trumps_Remaining` | Float | Ratio of unplayed trump cards remaining in play: `remaining_trumps / 10.0`. |
| 20 | `Score_Delta` | Float | Point delta between our team and opponent team: `(our_points - opp_points + 120) / 240.0`. |
| 21 | `Side0_Depletion` | Float | Played card ratio of side-suit 0 (first non-trump suit ascending): `played_cards / 10.0`. |
| 22 | `Side0_Ace_Played` | Bool | `1.0` if the Ace of side-suit 0 has already been played in previous tricks, `0.0` otherwise. |
| 23 | `Side0_7_Played` | Bool | `1.0` if the 7 of side-suit 0 has already been played in previous tricks, `0.0` otherwise. |
| 24 | `Side1_Depletion` | Float | Played card ratio of side-suit 1 (second non-trump suit ascending): `played_cards / 10.0`. |
| 25 | `Side1_Ace_Played` | Bool | `1.0` if the Ace of side-suit 1 has already been played in previous tricks, `0.0` otherwise. |
| 26 | `Side1_7_Played` | Bool | `1.0` if the 7 of side-suit 1 has already been played in previous tricks, `0.0` otherwise. |
| 27 | `Side2_Depletion` | Float | Played card ratio of side-suit 2 (third non-trump suit ascending): `played_cards / 10.0`. |
| 28 | `Side2_Ace_Played` | Bool | `1.0` if the Ace of side-suit 2 has already been played in previous tricks, `0.0` otherwise. |
| 29 | `Side2_7_Played` | Bool | `1.0` if the 7 of side-suit 2 has already been played in previous tricks, `0.0` otherwise. |


---

## Rust Acceleration

Simulation and evolution run entirely in Rust. The `sueca_solver` crate provides a bitboard game engine, WANN inference in CSR format, PIMC search with late-game minimax, and PyO3 bindings. The `sueca_wann` crate runs the NEAT evolutionary loop with Rayon parallelism across genomes, speciation, and breeding.

### Build Commands

```bash
# Build the PyO3 solver module (for Python scripts)
cargo build -p sueca_solver --release && \
cp target/release/libsueca_solver.so \
  .venv/lib/python3.13/site-packages/sueca_solver/sueca_solver.cpython-313-x86_64-linux-gnu.so

# Build the training binary
cargo build -p sueca_wann --release
```

---

## Quick Start

### Prerequisites
Python 3.13, Rust toolchain (`cargo`), `uv`.

```bash
uv sync                          # Install Python dependencies
cargo build -p sueca_wann --release  # Build training binary
```

### Running Training

```bash
./target/release/sueca_wann --config configs/default.toml
```

Training creates a dated run folder `checkpoints/YYYY-MM-DD-N/` containing `training_stats.csv`, checkpointed genomes, and the Hall of Fame. Resume with `--resume`.

### Running Tests

```bash
cargo test --all                 # Rust tests (engine, WANN, evolution)
uv run pytest tests/ -v          # Python tests (FFI, compiler, benchmark)
```

### Benchmarking a Champion

```bash
# Auto-detect latest genome
PYTHONPATH=. uv run python src/benchmark.py --deals 200

# Specify genome explicitly
PYTHONPATH=. uv run python src/benchmark.py \
  --deals 200 \
  --genome checkpoints/2026-05-26-1/genomes/best_genome_final.json
```

Outputs a tournament heatmap, CSV report, and terminal summary comparing RandomBot, HeuristicBot, PIMCBot, and the WANN champion.

### Comparing Training Runs

```bash
uv run python scripts/compare_runs.py
uv run python scripts/compare_runs.py --runs 2026-05-26-1 2026-05-26-2
```

Generates `checkpoints/run_comparison.png` with four panels: best fitness, delta vs HeuristicBot, species diversity, and network complexity across runs.

### Extracting Rules

```python
from src.export.export_flowchart import load_genome, compile_rules
genome = load_genome("checkpoints/2026-05-26-1/genomes/best_genome_final.json")
print(compile_rules(genome, W=1.0))
```

Outputs human-readable IF/THEN logic with referenced inputs, hidden node computations, and output formulas.

---

## Training Pipeline

### Phase 0: Supervised Pretraining (gens 0–100)
WANNs are trained to match PIMC expert intents on a pre-generated dataset. Fitness = classification accuracy. No game simulation — fast, deterministic evaluation.

### Phase 1: Self-Play Evolution (gens 100–1000)
WANNs play 48 deals × 4 rotations against HeuristicBot with Hall of Fame partners/opponents. Fitness = raw game-point delta vs baseline + Oracle Tax penalty for illegal intents. Tax ramps from −0.25 to −3.0 over 200 curriculum generations.

### Checkpoint Structure

```
checkpoints/
  2026-05-26-1/
    training_stats.csv      # Per-generation metrics
    training_state.bin      # Full state for --resume
    genomes/                # Subfolder containing genome json checkpoints
      best_genome_final.json # Best genome from entire run
      hof_final.json        # Final Hall of Fame (30 entries)
      hof_gen0100.json      # HOF snapshot at gen 100
      best_genome_gen0100.json
      ...
  2026-05-26-2/             # Auto-incremented for same-day runs
    ...
  run_comparison.png        # Generated by compare_runs.py
```

---

## Configuration

Key hyperparameters in `configs/default.toml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| population | pop_size | 400 | Population size |
| population | generations | 1000 | Total generations |
| population | elitism | 8 | Genomes copied verbatim per species |
| evaluation | n_deals | 48 | Deals per generation for Phase 1 |
| evaluation | curriculum_gens | 200 | Oracle tax ramp duration |
| species | compatibility_threshold | 1.2 | Speciation distance threshold |
| species | stagnation_limit | 25 | Gens without improvement before removal |
| mutation | p_add_node | 0.20 | Probability of add-node mutation |
| mutation | p_add_conn | 0.30 | Probability of add-connection mutation |
| mutation | p_crossover | 0.30 | Probability of sexual reproduction |
| curriculum | phase0_gens | 100 | Gens in supervised Phase 0 |
| hall_of_fame | hof_size | 30 | Max HOF entries |

---

## Source Layout

```
src/
  sueca_solver/src/         # Rust game engine + PyO3 bindings
    engine.rs               # Bitboard game state, card logic
    wann.rs                 # CSR-format WANN inference
    evaluator.rs            # Simulation runner, delta-fitness
    simulator.rs            # Game state wrapper with void tracking
    belief.rs               # Belief state encoder (30 floats)
    heuristic.rs            # Card selection, intent resolver
    pimc.rs                 # PIMC solver with late-game minimax
    search.rs               # Alpha-beta with Zobrist transposition table
    rng.rs                  # Shared LCG random number generator
    py_bindings/            # PyO3 FFI (wann, pimc, matchup)
  sueca_wann/src/           # Rust NEAT training binary
    main.rs                 # CLI entry point
    train.rs                # Training loop, Phase 0/1 dispatch
    genome.rs               # Genome representation, topological sort
    population.rs           # Population, crossover, Pareto, parallel breeding
    species.rs              # Compatibility distance, speciation
    mutations.rs            # NEAT mutation operators, innovation registry
    hall_of_fame.rs         # HOF with sampling
    config.rs               # TOML config deserialization
    checkpoint.rs           # Training state save/load
    dataset.rs              # Expert dataset loading
  export/
    export_flowchart.py     # Rule compiler, Graphviz topology export
  benchmark.py              # Tournament benchmarking
  measure_snr.py            # Signal-to-noise ratio measurement
  compat.py                 # Python-Rust data structure bridge
configs/
  default.toml              # Training hyperparameters
scripts/
  compare_runs.py           # Cross-run visualization
  generate_expert_dataset.py
tests/                      # Python integration tests
```
