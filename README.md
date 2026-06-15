# Sueca WANN

Evolves Weight-Agnostic Neural Networks (WANNs) to play Sueca, a Portuguese four-player partnership trick-taking card game. The system evolves discrete network topologies built from logical gates. You can compile the resulting networks into human-readable IF/THEN rules.

> **Status (2026-06-15).** The canonical champion is **v6** (`checkpoints/production/2026-06-14-2`), which **beats the strong EliteHeuristic baseline 52.1% ± 1.8% (n=3000)** while compiling to a handful of human-readable IF/THEN rules. The benchmarks below are current. See `problems.md` (Chapters 1–3) for the full diagnosis and the strength↔interpretability analysis.

## Architecture

```
Belief State (35 features) → WANN (Logical Gates) → Oracle Intent (3 outputs) → Heuristic Resolver → Card
```

**Belief State.** 35 normalized floats encoding hand composition, trick state, void tracking, tactical affordances (boss detection, "can I win?" evaluation), game progress, side-suit depletion, secured points, and meta features.

**WANN.** Weight-agnostic network with sign-only connections (±1), logical aggregations (SUM, MIN/AND, MAX/OR), and discrete activations (IDENTITY, NOT, THRESHOLD). No MEAN (float-precision issues at THRESHOLD boundary) and no SIGMOID (breaks IF/THEN rule extraction). Topologies are evaluated across a weight sweep W ∈ {−2.0, −1.0, −0.5, 0.5, 1.0, 2.0}.

**Oracle Intents.** Three abstract play archetypes that map to legal cards in any game state (MIN_FORCE removed — subsumed by EFFICIENT_WIN). Each is a **styled deviation around the strong Elite policy**, not a weak standalone tactic (see Resolver below):

- **EFFICIENT_WIN (1)** — The strong default. Delegates exactly to `EliteHeuristicBot` (cheapest winner, else cheap cut, else concede).
- **MAX_FORCE (0)** — Elite + control dials: when trump-long, lead a low trump to *draw* opponents' trumps; otherwise aggressive.
- **EQUITY_BUILDER (2)** — Elite + tempo dials: lead the *shortest* side-suit to build a void to cut from; *duck* cheap tricks (≤2 pts) when not last; *preserve* trump rather than cut a cheap trick.

**Heuristic Resolver (`select_card_styled`).** Maps each intent to a legal card contextually. Because every intent shares Elite's core and only adds dials, each pure intent benchmarks at ≈Elite individually (48/47/46% vs Elite) — so a policy that collapses to one intent merely *ties* Elite, while good situational mixing exceeds it. This raised-floor design (June 14) is what let the WANN finally beat Elite; the prior weak-intent resolver (20/37/25%) created a Phase-1 collapse basin. All three intents always resolve to a legal move. When WANN outputs tie, a random choice is made among tied maximums (not deterministic argmax).

### Belief State Layout (35 inputs, all in [0, 1])

| Index | Feature | Type | Description |
|:---:|---|---|---|
| 0 | `Has_Led_Suit` | Bool | Holds at least one card of the led suit |
| 1 | `Has_Trump` | Bool | Holds at least one trump card |
| 2 | `Led_Suit_Count` | Float | Cards held in led suit / 10.0 |
| 3 | `Trump_Count` | Float | Trumps held / 10.0 |
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
| 15 | `Led_Suit_7_Played` | Bool | 7 (manilha) of led suit already played |
| 16 | `Trump_Ace_Played` | Bool | Trump ace already played |
| 17 | `Holds_Boss_Led` | Bool | Holds the highest unplayed card in led suit |
| 18 | `Holds_Boss_Trump` | Bool | Holds the highest unplayed card in trump |
| 19 | `Can_Beat_Winner` | Bool | Any legal card can beat the current winner |
| 20 | `Min_Winning_Cost` | Float | Points of cheapest winning card / 11.0 |
| 21 | `Min_Sacrifice_Cost` | Float | Points of cheapest legal card / 11.0 |
| 22 | `Game_Pts_Remaining` | Float | Unplayed card points / 120.0 |
| 23 | `Trick_Number` | Float | Current trick index / 9.0 |
| 24 | `Trumps_Remaining` | Float | Unplayed trump cards / 10.0 |
| 25 | `Score_Delta` | Float | (our_pts − opp_pts + 120) / 240.0 |
| 26 | `My_Void_Count` | Float | Suits the player is void in / 3.0 |
| 27 | `Longest_Side_Suit` | Float | Max cards in non-trump, non-led suit / 10.0 |
| 28 | `Shortest_Side_Suit` | Float | Min cards in non-trump, non-led suit / 10.0 |
| 29 | `Side0_Depletion` | Float | Played cards of side-suit 0 / 10.0 |
| 30 | `Side1_Depletion` | Float | Played cards of side-suit 1 / 10.0 |
| 31 | `Side2_Depletion` | Float | Played cards of side-suit 2 / 10.0 |
| 32 | `Points_Secured_Us` | Float | Our team's secured game points / 120.0 |
| 33 | `Known_Void_Suits_Count` | Float | Suits where any player is known void / 4.0 |
| 34 | `Depleted_Suits_Count` | Float | Fully-depleted suits / 4.0 |

### Dataset Pipeline (2026-06 Recovery)

The dataset generator uses PIMC (Perfect Information Monte Carlo) with several quality filters:

- **Mid-trick random walk**: plays 0-7 complete tricks + 0-3 extra cards, ending at a uniform random point within a trick (~25% lead, ~75% follow).
- **Best-of-3-intent labeling** (June 14): each decisive state is labeled with the intent whose *resolved card* has the best PIMC EV among the 3 (statistically-tied intents → uniform multi-label; all 3 tied → reject). The old labeler kept a state only if some intent matched the *global* PIMC-best card — wrong target (the WANN can only choose among the 3 intents) and it discarded ~85% of states under the styled resolver.
- **Degenerate-only pre-filter**: rejects only states where all 3 intents resolve to the *same* card. Keeps every state where the intent choice has any effect.
- **Natural class distribution**: with the styled resolver the decision is effectively binary EFFICIENT-vs-EQUITY (MAX_FORCE is uniquely best ~1% of the time, **0%** in the follow split). Generate with `--soft-balance-min-ratio 0.0` (a nonzero floor makes the balancer hunt the unfillable MAX_FORCE bucket forever) and set `use_class_weighting = false` (inverse-frequency weighting would give the ~1% bucket a huge noise weight).
- **Futility stop**: ambiguous states exit PIMC early when the EV gap cannot plausibly become significant.
- **Exact card equivalence**: zero-point cards in the same suit with no intervening remaining cards are collapsed in alpha-beta search (provably lossless via `mask_between` test).
- **Killer-move heuristic**: 2-slot per-ply killer table in alpha-beta for ~30% more cutoffs.
- **Holdout set**: 10% stratified across all 6 buckets for validation accuracy tracking.
- **Diff mode**: `--diff-mode --fixed-worlds N` for controlled label comparison between pipeline versions.

Generate with (the canonical v6 dataset uses the **rollout teacher** — see below):
```bash
./target/release/sueca_wann generate-dataset \
  --n-worlds 200 --teacher rollout --target-count 15000 \
  --seed 42 --soft-balance-min-ratio 0.0 \
  --output expert_states_v6.npz
```

### Teacher selection (`--teacher`)

The `--teacher` flag chooses the EV oracle that labels each state:

- `--teacher alphabeta` (default) — deep PIMC with alpha-beta + late-game minimax. Its leaf eval returns the raw team score (`state.team_02_score`), which is **myopic**: it ties Elite rather than beating it, so its labels cap imitation at ≈Elite.
- `--teacher rollout` — flat Monte-Carlo PIMC where each determinized world is finished with **Elite playouts** (`solve_pimc_rollout`). By the rollout policy-improvement theorem (1-ply move + Elite playout ≥ Elite), this is **supra-Elite (62% vs Elite)** *and* ~1000× cheaper than deep alpha-beta — the v6 dataset (15k states) generates in ~11s. This is the canonical teacher.

## Benchmark Results (June 15 — canonical champion `checkpoints/production/2026-06-14-2`, "v6")

**The WANN beats EliteHeuristicBot.** Win rates as the row bot vs each column bot, **3000 deals** (seat-rotated, so the margin is statistically pinned):

| Bot | RandomBot | OldHeuristicBot | EliteHeuristicBot | WANN (Champion) |
|---|---|---|---|---|
| **RandomBot** | 50.0% | 10.8% | 4.7% | 5.1% |
| **OldHeuristicBot** | 89.2% | 50.0% | 32.5% | 32.3% |
| **EliteHeuristicBot** | 95.3% | 67.5% | 50.0% | 47.9% |
| **WANN (Champion)** | 95.0% | 67.7% | **52.1% ± 1.8%** | 50.0% |

Head-to-head vs Elite: **52.1% ± 1.8% at n=3000** (95% CI [50.3%, 53.9%], excludes 50% — a significant win), card points 60.2 vs 59.8.

### Why this champion is the canonical one: iso-strength, 4.5× simpler

The June 14 "v5" champion (`2026-06-14-1`) also beat Elite (52.7%), but it was a 132-hidden-gate tangle. The v6 champion — trained on the **rollout teacher** dataset — is **statistically the same strength** but radically smaller, and that is the headline interpretability result:

| Champion | vs Elite (n=3000) | Hidden gates | Enabled connections |
|---|---|---|---|
| v5 (`2026-06-14-1`) | 52.7% ± 1.8% | 132 | 188 |
| **v6 (`2026-06-14-2`)** | **52.1% ± 1.8%** | **29** (4.5× fewer) | **49** (3.8× fewer) |

Both brains compile to a handful of human-readable IF/THEN steps (see `compile-rules` output). The surprising finding (full write-up in `problems.md`, Chapter 2): a **supra-Elite teacher does not raise the benchmark** — strength is *resolver-floored* (every intent resolves to an Elite-flavoured card, so the WANN can only pick *which* small deviation from Elite to apply), and the realisable headroom above the Elite floor is ≈8 points. What the stronger, cleaner teacher buys is a **simpler, genuinely interpretable champion at iso-strength**, not a stronger one. See `problems.md` for the full diagnosis (the project's first Elite-beating champion was 30.2%).

## Rust Crates

Three Rust crates, no Python FFI, no `.so` build step.

- **`sueca_solver`** — Pure game engine (bitboard state, PIMC search with late-game minimax, belief encoding, heuristic intent resolver). rlib only.
- **`sueca_wann`** — WANN inference, NEAT evolution loop, and CLI. Depends on `sueca_solver`.
- **`sueca_wasm`** — WASM bindings for browser-based play. Compiled via wasm-pack, consumed by the React frontend.

```bash
cargo build -p sueca_wann --release
```

## Quick Start

```bash
uv sync                              # Python deps (visualization only)
cargo build -p sueca_wann --release  # Build training binary
git config core.hooksPath .githooks  # Activate WASM pre-commit hook
```

### Training

```bash
./target/release/sueca_wann train --config configs/default.toml
./target/release/sueca_wann train --config configs/default.toml --resume  # resume from checkpoint
```

Creates `code/checkpoints/YYYY-MM-DD-N/` with training stats, checkpointed genomes, and Hall of Fame.

### Tests

```bash
cargo test --all
```

## Reproducing the Results

All commands are run from the repository root. The canonical v6 champion lives at `code/checkpoints/production/2026-06-14-2/genomes/best_genome_final.json`. **`checkpoints/` is gitignored** — the champion and training artifacts live on the training machine, not in the repo.

### Build

```bash
cargo build -p sueca_wann --release
```

### Test

```bash
cargo test --all
```

### Benchmark the Champion

Benchmarks the v6 champion against RandomBot, OldHeuristicBot, and EliteHeuristicBot over 3000 seat-rotated deals. Expect **~52.1% ± 1.8%** vs EliteHeuristicBot.

```bash
./target/release/sueca_wann benchmark \
  --deals 3000 \
  --genome checkpoints/production/2026-06-14-2/genomes/best_genome_final.json
```

### Compile Interpretable Rules

Extracts the champion's IF/THEN rules, DOT topology graph, and PNG rendering.

```bash
./target/release/sueca_wann compile-rules \
  --genome checkpoints/production/2026-06-14-2/genomes/best_genome_final.json \
  --output-dir checkpoints/production/2026-06-14-2
```

Generates `compiled_rules.txt`, `topology_graph.dot`, and `topology_graph.png` in the output directory.

### Regenerate the v6 Expert Dataset

Generates the 15k-state PIMC expert dataset used for Phase 0 pretraining (rollout teacher, supra-Elite labels).

```bash
./target/release/sueca_wann generate-dataset \
  --n-worlds 200 --teacher rollout --target-count 15000 \
  --soft-balance-min-ratio 0.0 \
  --output expert_states_v6.npz
```

### Generate Report Figures

Produces three PDF figures (`training_curve.pdf`, `tournament.pdf`, `complexity.pdf`) from the canonical results, saved into `report/figures/`.

```bash
uv run python scripts/make_report_figures.py
```

### Web Play Interface

To build and run the web play interface (React + WASM game loop):

1. **Build WASM game engine:**
   ```bash
   cd src/sueca_wasm
   RUSTFLAGS="" wasm-pack build --target web --out-dir ../../frontend/src/wasm
   cd ../..
   ```

2. **Run dev server:**
   ```bash
   cd frontend
   bun dev
   ```

Open `http://localhost:5173` in your browser to play Sueca vs WANN/heuristics.


### Benchmarking

```bash
# Full tournament with custom weight sweep
./target/release/sueca_wann benchmark \
  --deals 200 \
  --genome checkpoints/2026-06-03-2/genomes/best_genome_final.json \
  --weights -2.0,-1.0,-0.5,0.5,1.0,2.0 \
  --seed 42

# Auto-detect latest genome, use default weight sweep
./target/release/sueca_wann benchmark --deals 200
```

Optional flags: `--output-dir <dir>` to override the report output directory, `--seed <u64>` for reproducibility. The `--weights` flag accepts a comma-separated list of shared weight values for the WANN sweep.

### Comparing Runs

Cross-run comparison with 4-panel visualization (fitness, delta vs HeuristicBot, species diversity, network complexity):

```bash
uv run python scripts/compare_runs.py                    # all runs
uv run python scripts/compare_runs.py --runs 2026-06-03-2  # specific run
```

Saves `checkpoints/run_comparison.png`.

### Analyzing a Single Run

Per-run training plots (fitness curves, species counts, network complexity over time):

```bash
uv run python scripts/plot_training.py \
  --stats checkpoints/2026-06-03-2/training_stats.csv \
  --out-dir checkpoints/2026-06-03-2
```

### Analyzing Expert Datasets

```bash
uv run python scripts/analyze_dataset.py expert_states.npz      # quick stats with plots
uv run python scripts/dataset_analysis.py expert_states.npz      # comprehensive text report
```

### Batch Rule Compilation

Compile rules for all genomes across all checkpoints:

```bash
uv run python scripts/compile_all.py
```

### Extracting Rules

```bash
# Default weight (1.0)
./target/release/sueca_wann compile-rules \
  --genome checkpoints/2026-06-03-2/genomes/best_genome_final.json \
  --output-dir checkpoints/2026-06-03-2

# Extract at a specific sweep weight (e.g. -1.0 for inhibitory rules)
./target/release/sueca_wann compile-rules \
  --genome checkpoints/2026-06-03-2/genomes/best_genome_final.json \
  --output-dir checkpoints/2026-06-03-2 \
  --weight -1.0
```

Generates `compiled_rules.txt` (IF/THEN logic), `topology_graph.dot`, and `topology_graph.png`.

### Generating Expert Datasets

```bash
./target/release/sueca_wann generate-dataset \
  --n-worlds 80 --search-depth 4 --target-count 10000 \
  --output expert_states.npz
```

### Optimizing Weights

After evolving a topology, optimize independent continuous weights per connection using Differential Evolution:

```bash
./target/release/sueca_wann optimize-weights \
  --genome checkpoints/2026-06-03-2/genomes/best_genome_final.json \
  --deals 200 --generations 50
```

This produces `optimized_weights.json` in the genome's directory. The benchmark command auto-detects this file and adds a "WANN (Optimized)" bot to the tournament.

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

Key hyperparameters from `code/configs/default.toml`:

| Section | Key | Value | Description |
|---------|-----|-------|-------------|
| population | pop_size | 1000 | Population size |
| population | generations | 1200 | Total generations |
| population | elitism | 3 | Genomes copied verbatim per species |
| population | pareto_complexity_prob | 0.50 | Probability of using Pareto (perf+simplicity) ranking |
| evaluation | n_deals | 128 | Duplicate deals per generation |
| evaluation | curriculum_gens | 300 | Gens of curriculum-guided evolution |
| evaluation | sweep_weights | [1.0] | Weight sweep values |
| evaluation | seed | 1337 | Base RNG seed |
| species | compatibility_threshold | 1.4 | Speciation distance threshold |
| species | stagnation_limit | 40 | Gens without improvement before removal |
| species | c_excess | 1.0 | Excess gene coefficient |
| species | c_disjoint | 1.0 | Disjoint gene coefficient |
| species | c_mismatch | 0.5 | Weight mismatch coefficient |
| species | min_species_size | 3 | Minimum genomes per species |
| mutation | p_add_node | 0.20 | Add-node probability |
| mutation | p_add_conn | 0.35 | Add-connection probability |
| mutation | p_toggle_conn | 0.05 | Toggle-connection probability |
| mutation | p_flip_sign | 0.10 | Flip-sign probability |
| mutation | p_change_act | 0.25 | Change-activation probability |
| mutation | p_change_agg | 0.15 | Change-aggregation probability |
| mutation | p_crossover | 0.40 | Crossover probability |
| curriculum | phase0_gens | 200 | Gens in supervised Phase 0 |
| curriculum | bulking_gens | 100 | Gens of connection bulking in Phase 0 |
| curriculum | phase0_dataset | db_w40_d3_mar03.npz | Expert dataset for Phase 0 |
| hall_of_fame | hof_size | 50 | Max HOF entries |

## Source Layout

```
src/
  sueca_solver/src/         # Pure game engine (rlib)
    engine.rs               # Bitboard state, card logic, beats
    simulator.rs            # Game wrapper with void tracking
    belief.rs               # Belief state encoder (33 floats)
    heuristic.rs            # Card selection, intent resolver
    pimc.rs                 # PIMC solver, late-game minimax switch
    search.rs               # Alpha-beta, Zobrist hashing, TT
    rng.rs                  # Shared LCG RNG
    constants.rs            # WANN dimension constants
  sueca_wann/src/           # Training binary + CLI
    main.rs                 # CLI (train, benchmark, compile-rules, generate-dataset, optimize-weights)
    train.rs                # Training loop, Phase 0/1 dispatch
    evaluator.rs            # Bot simulation, delta-fitness evaluation
    wann_network.rs         # CSR-format WANN inference
    genome.rs               # Genome representation, topological sort
    population.rs           # Population, crossover, Pareto ranking, parallel breeding
    species.rs              # Compatibility distance, speciation
    mutations.rs            # NEAT mutation operators, innovation registry
    hall_of_fame.rs         # HOF with sampling
    map_elites.rs           # MAP-Elites quality-diversity archive
    optimize.rs             # Differential Evolution weight optimization
    config.rs               # TOML config deserialization
    checkpoint.rs           # Training state save/load
    compile_rules.rs        # Rule compiler, DOT export, PNG rendering
    benchmark.rs            # Tournament benchmarking
    dataset_gen.rs          # PIMC expert dataset generation
    dataset.rs              # Expert dataset loading (NPZ)
    constants.rs            # Evolutionary hyperparameters, feature names
  sueca_wasm/src/           # WASM bindings for browser play
    lib.rs                  # WASM entry point, game loop for frontend
configs/
  default.toml              # Production hyperparameters
  test.toml                 # Test run hyperparameters
  pgo_bench.toml            # PGO benchmarking config
  pgo_profile.toml          # PGO profiling config
  profile_phase0.toml       # Phase 0 profiling config
  profile_phase1.toml       # Phase 1 profiling config
scripts/
  compare_runs.py           # Cross-run visualization
  plot_training.py          # Single-run training plot
  analyze_dataset.py        # Dataset statistics with plots
  dataset_analysis.py       # Comprehensive dataset text report
  compile_all.py            # Batch rule compilation
```

## Implemented Milestones & Advanced Search

### 1. Linear Input Pruning (PFS-NEAT)
WANNs are evolved from an empty starting footprint (0 active connections). Online Mutational Filtering verifies performance lift on connection proposals, protecting genomes from noisy/redundant inputs. Pareto selection pressure filters out neutral mutations that do not provide synergistic lifts over generations.

### 2. SNAP-NEAT + Tabu search + Multi-Brain Partitioning
* **Two-Level Tabu Veto:** Compiles hardcoded static structural constraints (self-loops, bias/inputs as targets, and cycles) with a dynamic FIFO lock-free queue that stores degraded mutation paths to bypass redundant evaluation.
* **Modular Multi-Brain co-evolution:** Evolve modular Lead Brain (leading hand) and Follow Brain (following hand) populations. Game actions route decisions dynamically per play using `BeliefFeature::AmILeading`. Split brains reduce strategic entropy, accelerating search accuracy.

