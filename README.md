# Sueca WANN

Evolves Weight-Agnostic Neural Networks (WANNs) to play Sueca, a Portuguese four-player partnership trick-taking card game. The system evolves discrete network topologies built from logical gates. You can compile the resulting networks into human-readable IF/THEN rules.

## Architecture

```
Belief State (30 features) → WANN (Logical Gates) → Oracle Intent (4 outputs) → Heuristic Resolver → Card
```

**Belief State.** 30 normalized floats encoding hand composition, trick state, void tracking, game progress, and side-suit depletion.

**WANN.** Weight-agnostic network with sign-only connections (±1), logical aggregations (SUM, MIN, MAX), and discrete activations (IDENTITY, NOT, THRESHOLD). Topologies are evaluated across a weight sweep W ∈ {−2.0, −1.0, −0.5, 0.5, 1.0, 2.0}.

**Oracle Intents.** Four abstract play archetypes that map to legal cards in any game state:

- **MAX_FORCE** — Lead high trump or a cash master card. Play the highest-ranking card when following.
- **MIN_FORCE** — Lead from the longest non-trump suit. Play the lowest legal card to preserve high cards.
- **EFFICIENT_WIN** — Lead low trump or longest suit. Play the cheapest card that beats the current winner, or cut with the cheapest trump.
- **EQUITY_BUILDER** — Lead a short suit to build voids. Load points when partner is winning. Exploit opponent voids. Cut when partner is void.

**Heuristic Resolver.** Maps each intent to a legal card contextually. All four intents always resolve to a legal move, so illegal intents never occur during rollouts.

### Belief State Layout (30 inputs, all in [0, 1])

| Index | Feature | Type | Description |
|:---:|---|---|---|
| 0 | `Has_Led_Suit` | Bool | Holds at least one card of the led suit |
| 1 | `Has_Trump` | Bool | Holds at least one trump card |
| 2 | `Led_Suit_Power` | Float | Highest rank held in led suit / 9.0 |
| 3 | `Trump_Power` | Float | Highest rank held in trump / 9.0 |
| 4 | `Hand_Point_Density` | Float | Points in hand / points remaining in game |
| 5 | `Am_I_Leading` | Bool | First to play in the trick |
| 6 | `Am_I_Last_To_Play` | Bool | Fourth to play in the trick |
| 7 | `Is_Partner_Winning` | Bool | Partner currently winning the trick |
| 8 | `Trick_Point_Value` | Float | Points on the table / 44.0 |
| 9 | `Has_Trick_Been_Cut` | Bool | Trump played when led suit is not trump |
| 10 | `Partner_Void_Led` | Bool | Partner is void in the led suit |
| 11 | `Partner_Void_Trump` | Bool | Partner is void in the trump suit |
| 12 | `Any_Opp_Void_Led` | Bool | Either opponent is void in the led suit |
| 13 | `Any_Opp_Void_Trump` | Bool | Either opponent is void in the trump suit |
| 14 | `Led_Suit_Ace_Played` | Bool | Ace of led suit already played |
| 15 | `Led_Suit_7_Played` | Bool | 7 of led suit already played |
| 16 | `Trump_Ace_Played` | Bool | Trump ace already played |
| 17 | `Game_Pts_Remaining` | Float | Unplayed card points / 120.0 |
| 18 | `Trick_Number` | Float | Current trick index / 9.0 |
| 19 | `Trumps_Remaining` | Float | Unplayed trump cards / 10.0 |
| 20 | `Score_Delta` | Float | (our_pts − opp_pts + 120) / 240.0 |
| 21 | `Side0_Depletion` | Float | Played cards of side-suit 0 / 10.0 |
| 22 | `Side0_Ace_Played` | Bool | Ace of side-suit 0 already played |
| 23 | `Side0_7_Played` | Bool | 7 of side-suit 0 already played |
| 24 | `Side1_Depletion` | Float | Played cards of side-suit 1 / 10.0 |
| 25 | `Side1_Ace_Played` | Bool | Ace of side-suit 1 already played |
| 26 | `Side1_7_Played` | Bool | 7 of side-suit 1 already played |
| 27 | `Side2_Depletion` | Float | Played cards of side-suit 2 / 10.0 |
| 28 | `Side2_Ace_Played` | Bool | Ace of side-suit 2 already played |
| 29 | `Side2_7_Played` | Bool | 7 of side-suit 2 already played |

## Benchmark Results

We evaluated the evolved WANN models in a round-robin tournament (200 duplicate matches / 400 games total per matchup).
With the correct duplicate-match win-rate calculation (summing to 100%), the results are:

| Bot | RandomBot | OldHeuristicBot | EliteHeuristicBot | WANN (Champion) | WANN (Optimized) |
|---|---|---|---|---|---|
| **RandomBot** | 50.0% | 10.8% | 6.8% | 7.8% | 13.0% |
| **OldHeuristicBot** | 89.2% | 50.0% | 37.5% | 45.2% | 39.0% |
| **EliteHeuristicBot** | 93.2% | 62.5% | 50.0% | 58.5% | 60.2% |
| **WANN (Champion)** | 92.2% | 54.8% | 41.5% | 50.0% | 49.8% |
| **WANN (Optimized)** | 87.0% | 61.0% | 39.8% | 50.2% | 50.0% |

**Key Takeaways:**
1. **WANN (Champion)** decisively beats **OldHeuristicBot** with a **54.8% win rate** (61.9 vs 58.1 average card points per match).
2. **WANN (Optimized)** (with continuous weights optimized via DE) achieves a **61.0% win rate** against **OldHeuristicBot** (62.6 vs 57.4 average card points per match).
3. Both WANN variants show strong competitive performance against **EliteHeuristicBot** (~40% win rate), outperforming **OldHeuristicBot** which only wins 37.5% of games against **EliteHeuristicBot**.
4. WANN Champion and WANN Optimized are in near-perfect parity, with WANN Optimized winning **50.2%** of their head-to-head matches.

## Rust Crates

Two Rust crates, no Python FFI, no `.so` build step.

- **`sueca_solver`** — Pure game engine (bitboard state, PIMC search with late-game minimax, belief encoding, heuristic intent resolver). rlib only.
- **`sueca_wann`** — WANN inference, NEAT evolution loop, and CLI. Depends on `sueca_solver`.

```bash
cargo build -p sueca_wann --release
```

## Quick Start

```bash
uv sync                              # Python deps (visualization only)
cargo build -p sueca_wann --release  # Build training binary
```

### Training

```bash
./target/release/sueca_wann train --config configs/default.toml
./target/release/sueca_wann train --config configs/default.toml --resume  # resume from checkpoint
```

Creates `checkpoints/YYYY-MM-DD-N/` with training stats, checkpointed genomes, and Hall of Fame.

### Tests

```bash
cargo test --all
```

### Benchmarking

```bash
./target/release/sueca_wann benchmark --deals 200 --genome checkpoints/2026-06-03-2/genomes/best_genome_final.json
```

### Comparing Runs

```bash
uv run python scripts/compare_runs.py
uv run python scripts/compare_runs.py --runs 2026-06-03-2
```

### Extracting Rules

```bash
./target/release/sueca_wann compile-rules --genome checkpoints/2026-06-03-2/genomes/best_genome_final.json --output-dir checkpoints/2026-06-03-2
```

Generates `compiled_rules.txt`, `topology_graph.dot`, and `topology_graph.png`.

### Generating Expert Datasets

```bash
./target/release/sueca_wann generate-dataset \
  --n-worlds 80 --search-depth 4 --target-count 10000 \
  --output expert_states.npz
```

## Training Pipeline

The training pipeline consists of two distinct phases:

### Phase 0: Supervised Pretraining with Split Datasets (gens 0 to 200)
WANNs are pretrained to match Perfect Information Monte Carlo (PIMC) expert intents.
* **Dataset Splitting:** The input expert dataset (such as `expert_states_w20_d2.npz`) is partitioned into two subsets using the `BeliefFeature::AmILeading` flag: `lead_dataset` (low entropy, 87% concentrated on aggressive MAX_FORCE and equity actions) and `follow_dataset` (high entropy, distributed across passive and efficient actions).
* **Zero-Connection Start (PFS-NEAT):** Both `lead_pop` and `follow_pop` populations are initialized with genomes carrying exactly 0 active connections.
* **Online Mutational Filtering:** During connection mutations, candidates are checked against a thread-safe FIFO `TabuVetoList` of size 1000. If the path is not tabued, it is temporarily applied and evaluated on a 1000-state subset. If the mutation degrades accuracy compared to the parent, it is discarded and pushed onto the Tabu queue. Beneficial and neutral mutations are preserved; neutral mutations that fail to find a synergistic partner node are pruned by Pareto complexity domination.
* **Training Output:** Phase 0 finishes when populations independently reach convergence, typically achieving over 60% aggregate accuracy.

### Phase 1: Co-evolutionary Self-Play (gens 200+)
* **Co-Evolution:** Lead Brains and Follow Brains co-evolve. In each duplicate matchup, a candidate Lead Brain is paired with the current champion Follow Brain, and a candidate Follow Brain is paired with the champion Lead Brain.
* **Dynamic Routing:** During gameplay, cards are played seat-by-seat. The evaluator queries a unified `Wann` simulator bot which routes decisions dynamically at each card play slice:
  ```rust
  let network = if belief[BeliefFeature::AmILeading as usize] == 1.0 {
      lead_brain
  } else {
      follow_brain
  };
  ```
* **Duplicate Matching:** Delta-fitness is computed using Common Random Numbers (CRN) over seat rotations on duplicate deals to isolate pure strategic skill from card luck.

## Checkpoint Structure

The training state is statefully saved to a binary file `training_state.bin` using Bincode. Lead and Follow training states are fully encapsulated inside separate fields utilizing `BrainTrainingState`:

```
checkpoints/
  2026-06-03-2/
    training_stats.csv       # Training stats for both brains (lead/follow accuracy & enabled connections)
    training_state.bin       # Stateful binary containing encapsulated Lead and Follow BrainTrainingStates
    genomes/
      best_genome_final.json # Final JsonGenomeJoint containing lead and follow JSON genomes
      hof_final.json         # Final JsonHallOfFameJoint containing HOF entries for both brains
```

## Configuration

Key hyperparameters from `configs/default.toml`:

| Section | Key | Value | Description |
|---------|-----|-------|-------------|
| population | pop_size | 1000 | Population size |
| population | generations | 1200 | Total generations |
| population | elitism | 3 | Genomes copied verbatim per species |
| evaluation | n_deals | 128 | Duplicate deals per generation |
| evaluation | sweep_weights | [1.0] | Weight sweep values |
| species | compatibility_threshold | 1.4 | Speciation distance threshold |
| species | stagnation_limit | 40 | Gens without improvement before removal |
| mutation | p_add_node | 0.20 | Add-node probability |
| mutation | p_add_conn | 0.35 | Add-connection probability |
| mutation | p_crossover | 0.40 | Crossover probability |
| curriculum | phase0_gens | 200 | Gens in supervised Phase 0 |
| hall_of_fame | hof_size | 50 | Max HOF entries |

## Source Layout

```
src/
  sueca_solver/src/         # Pure game engine (rlib)
    engine.rs               # Bitboard state, card logic, beats
    simulator.rs            # Game wrapper with void tracking
    belief.rs               # Belief state encoder (30 floats)
    heuristic.rs            # Card selection, intent resolver
    pimc.rs                 # PIMC solver, late-game minimax switch
    search.rs               # Alpha-beta, Zobrist hashing, TT
    rng.rs                  # Shared LCG RNG
    constants.rs            # WANN dimension constants
  sueca_wann/src/           # Training binary + CLI
    main.rs                 # CLI (train, benchmark, compile-rules, generate-dataset)
    train.rs                # Training loop, Phase 0/1 dispatch
    evaluator.rs            # Bot simulation, delta-fitness evaluation
    wann_network.rs         # CSR-format WANN inference
    genome.rs               # Genome representation, topological sort
    population.rs           # Population, crossover, Pareto ranking, parallel breeding
    species.rs              # Compatibility distance, speciation
    mutations.rs            # NEAT mutation operators, innovation registry
    hall_of_fame.rs         # HOF with sampling
    map_elites.rs           # MAP-Elites quality-diversity archive
    config.rs               # TOML config deserialization
    checkpoint.rs           # Training state save/load
    compile_rules.rs        # Rule compiler, DOT export, PNG rendering
    benchmark.rs            # Tournament benchmarking
    dataset_gen.rs          # PIMC expert dataset generation
    dataset.rs              # Expert dataset loading (NPZ)
    constants.rs            # Evolutionary hyperparameters, feature names
configs/
  default.toml              # Production hyperparameters
  test.toml                 # Test run hyperparameters
scripts/
  compare_runs.py           # Cross-run visualization
```

## Implemented Milestones & Advanced Search

### 1. Linear Input Pruning (PFS-NEAT)
WANNs are evolved from an empty starting footprint (0 active connections). Online Mutational Filtering verifies performance lift on connection proposals, protecting genomes from noisy/redundant inputs. Pareto selection pressure filters out neutral mutations that do not provide synergistic lifts over generations.

### 2. SNAP-NEAT + Tabu search + Multi-Brain Partitioning
* **Two-Level Tabu Veto:** Compiles hardcoded static structural constraints (self-loops, bias/inputs as targets, and cycles) with a dynamic FIFO lock-free queue that stores degraded mutation paths to bypass redundant evaluation.
* **Modular Multi-Brain co-evolution:** Evolve modular Lead Brain (leading hand) and Follow Brain (following hand) populations. Game actions route decisions dynamically per play using `BeliefFeature::AmILeading`. Split brains reduce strategic entropy, accelerating search accuracy.

