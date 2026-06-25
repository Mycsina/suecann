# Sueca WANN

Evolves Weight-Agnostic Neural Networks (WANNs) to play Sueca, a Portuguese four-player partnership trick-taking card game. The system evolves discrete network topologies built from logical gates. You can compile the resulting networks into human-readable IF/THEN rules.

> **Status (2026-06-15).** The canonical champion is **v6** (`checkpoints/production/2026-06-14-2`), which **beats the strong EliteHeuristic baseline 52.1% ± 1.8% (n=3000)** while compiling to a handful of human-readable IF/THEN rules. The benchmarks below are current. See `problems.md` (Chapters 1–3) for the full diagnosis and the strength↔interpretability analysis.

## Architecture

```
Belief State (35 features) → WANN (Logical Gates) → φ-Utility Knobs (6 outputs) → φ-Resolver → Card
```

**Belief State.** 35 normalized floats encoding hand composition, trick state, void tracking, tactical affordances (boss detection, "can I win?" evaluation), game progress, side-suit depletion, secured points, and meta features.

**WANN.** Weight-agnostic network with sign-only connections (±1), logical aggregations (SUM, MIN/AND, MAX/OR), and discrete activations (IDENTITY, NOT, THRESHOLD). No MEAN (float-precision issues at THRESHOLD boundary) and no SIGMOID. Output nodes use THRESHOLD so positive φ-knobs are reachable under the symmetric weight sweep (an IDENTITY output biases knobs ≤ 0 — see CLAUDE.md "Substrate decision"); `change_activation` may target output nodes. Topologies are evaluated across a weight sweep W ∈ {−2.0, −1.0, −0.5, 0.5, 1.0, 2.0}.

**φ-Utility Knobs.** The WANN emits 6 continuous knobs `w ∈ [-1,1]`, one per hand-designed card-utility feature φ(card, state):

| ID | Knob | φ feature | positive knob → prefer… |
|----|------|-----------|-------------------------|
| 0 | RANK | `CARD_RANK/9` | high-rank cards |
| 1 | POINTS | `CARD_POINTS/11` | high-point cards |
| 2 | TRUMP | is trump | trumps |
| 3 | WINS | would beat current winner | winning the trick now |
| 4 | CAPTURES | trick points/30 if wins | capturing the points on the table |
| 5 | VOID | last card of its suit in hand | building/keeping a void |

**φ-Resolver (`resolve_card_phi_utility`).** Plays `argmax_{legal} Σ_k w_k · φ_k(card, state)`. `PhiCtx` (trump, ego hand, trick cards) is the compact context φ is computed from; `PhiCtx::legal()` derives the follow-suit legal set, so the resolver always returns a legal card.

**Why this design (Stage B, 2026-06-19).** A measured ceiling decomposition drove the overhaul: the prior 3-intent action vocabulary capped at ~57% vs Elite while the free-card ceiling is ~81%, and a continuous linear utility over these 6 features reaches ~81% (= free-card). The 3-intent vocabulary was the dominant ~24-point bottleneck; the φ-resolver recovers essentially all of it. (The prior 3-intent resolver `select_card_styled` / `resolve_intent` is retained only for the `OracleEnvelope` ceiling diagnostic.) Full spec: `docs/superpowers/specs/2026-06-19-resolver-overhaul-design.md`.

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
- **Card-match labeling** (Stage B): for each state the rollout teacher returns an EV for every legal card; `best_cards` = the legal cards within one stderr of the max EV (statistical ties are multi-label — the resolver gets full credit for any tied-best card).
- **Signal-preserving pre-filters**: forced (single-legal) and fully-ambiguous (all legal cards tied) states are rejected — they carry no card-selection signal.
- **Futility stop**: ambiguous states exit PIMC early when the EV gap cannot plausibly become significant.
- **Exact card equivalence**: zero-point cards in the same suit with no intervening remaining cards are collapsed in alpha-beta search (provably lossless via `mask_between` test).
- **Killer-move heuristic**: 2-slot per-ply killer table in alpha-beta for ~30% more cutoffs.
- **Holdout set**: 10% per split (lead/follow) for validation accuracy tracking.
- **Diff mode**: `--diff-mode --fixed-worlds N` for controlled label comparison between pipeline versions.

Generate with (the canonical v6 dataset uses the **rollout teacher** — see below):
```bash
code/target/release/sueca_wann generate-dataset \
  --n-worlds 200 --teacher rollout --target-count 15000 \
  --seed 42 --soft-balance-min-ratio 0.0 \
  --output expert_states_v7.npz
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

Both brains compile to human-readable IF/THEN steps over the 6 φ-knobs (see `compile-rules` output). A measured ceiling decomposition (2026-06-19) drove the Stage B resolver overhaul: the prior 3-intent action vocabulary capped at ~57% vs Elite while the **free-card ceiling is ~81%**, and a continuous linear utility over 6 hand-designed card features reaches ~81% (= free-card). The 3-intent vocabulary was the dominant ~24-point bottleneck; the φ-resolver recovers essentially all of it. (Earlier finding that a supra-Elite teacher "does not raise the benchmark" was true *inside* the old 3-intent cage — the real reachable ceiling is ~31 points above Elite.) Full decomposition in the design spec and `ideas.md`; the project's first Elite-beating champion was 30.2%.

## Rust Crates

Three Rust crates, no Python FFI, no `.so` build step.

- **`sueca_solver`** — Pure game engine (bitboard state, PIMC search with late-game minimax, belief encoding, heuristic intent resolver). rlib only.
- **`sueca_wann`** — WANN inference, NEAT evolution loop, and CLI. Depends on `sueca_solver`.
- **`sueca_wasm`** — WASM bindings for browser-based play. Compiled via wasm-pack, consumed by the React frontend.

```bash
cargo build --manifest-path code/Cargo.toml -p sueca_wann --release
```

## Quick Start

```bash
uv sync --project code                              # Python deps (visualization only)
cargo build --manifest-path code/Cargo.toml -p sueca_wann --release  # Build training binary
git config core.hooksPath code/.githooks                   # Activate WASM pre-commit hook
```

### Training

```bash
code/target/release/sueca_wann train --config code/configs/default.toml
code/target/release/sueca_wann train --config code/configs/default.toml --resume  # resume from checkpoint
```

Creates `code/checkpoints/YYYY-MM-DD-N/` with training stats, checkpointed genomes, and Hall of Fame.

### Tests

```bash
cargo test --manifest-path code/Cargo.toml --all
```

## Reproducing the Results

All commands are run from the repository root. The canonical v6 champion lives at `code/checkpoints/production/2026-06-14-2/genomes/best_genome_final.json`. Checkpoints are tracked in the repo (final genomes, training stats, compiled rules, and the production directory; per-generation snapshots and binary training state are excluded via `code/checkpoints/.gitignore`).

### Build

```bash
cargo build --manifest-path code/Cargo.toml -p sueca_wann --release
```

### Test

```bash
cargo test --manifest-path code/Cargo.toml --all
```

### Benchmark the Champion

Benchmarks the v6 champion against RandomBot, OldHeuristicBot, and EliteHeuristicBot in a round-robin tournament. Each matchup plays the same 3000 seat-rotated deals for a statistically-pinned result.

```bash
code/target/release/sueca_wann benchmark \
  --deals 3000 \
  --genome code/checkpoints/production/2026-06-14-2/genomes/best_genome_final.json
```

**Expected output** (takes ~2 minutes on a modern machine):

| Bot | RandomBot | OldHeuristic | EliteHeuristic | WANN (Champion) |
|---|---|---|---|---|
| **RandomBot** | 50.0% | 10.8% | 4.7% | 5.1% |
| **OldHeuristicBot** | 89.2% | 50.0% | 32.5% | 32.3% |
| **EliteHeuristicBot** | 95.3% | 67.5% | 50.0% | 47.9% |
| **WANN (Champion)** | 95.0% | 67.7% | **52.1% ± 1.8%** | 50.0% |

The key number is **52.1% ± 1.8%** — the WANN beats the strong Elite baseline by a statistically significant margin.

**Quick check** (200 deals, ~10 seconds, no `--genome` auto-detects the latest genome in `code/checkpoints/`):

```bash
code/target/release/sueca_wann benchmark --deals 200
```

**Custom weight sweep** (if you've trained your own network):

```bash
code/target/release/sueca_wann benchmark \
  --deals 200 \
  --genome code/checkpoints/YOUR-RUN/genomes/best_genome_final.json \
  --weights -2.0,-1.0,-0.5,0.5,1.0,2.0 \
  --seed 42
```

The report is saved as `tournament_report.csv` alongside the genome file.

### Compile Interpretable Rules

Extracts the champion's IF/THEN rules, DOT topology graph, and PNG rendering.

```bash
code/target/release/sueca_wann compile-rules \
  --genome code/checkpoints/production/2026-06-14-2/genomes/best_genome_final.json \
  --output-dir code/checkpoints/production/2026-06-14-2
```

Generates `compiled_rules.txt`, `topology_graph.dot`, and `topology_graph.png` in the output directory.

### Regenerate the v6 Expert Dataset

Generates the 15k-state PIMC expert dataset used for Phase 0 pretraining (rollout teacher, supra-Elite labels).

```bash
code/target/release/sueca_wann generate-dataset \
  --n-worlds 200 --teacher rollout --target-count 15000 \
  --soft-balance-min-ratio 0.0 \
  --output expert_states_v7.npz
```

### Generate Report Figures

Produces three PDF figures (`training_curve.pdf`, `tournament.pdf`, `complexity.pdf`) from the canonical results, saved into `report/figures/`.

```bash
uv run --project code python code/scripts/make_report_figures.py
```

### Web Play Interface

To build and run the web play interface (React + WASM game loop):

1. **Build WASM game engine:**
   ```bash
   cd code/src/sueca_wasm
   RUSTFLAGS="" wasm-pack build --target web --out-dir ../../frontend/src/wasm
   cd ../..
   ```

2. **Run dev server:**
   ```bash
   cd code/frontend
   bun dev
   ```

Open `http://localhost:5173` in your browser to play Sueca vs WANN/heuristics.


### Benchmarking

Run a round-robin tournament for any genome. If `--genome` is omitted, the latest genome under `code/checkpoints/` is auto-detected. See [Benchmark the Champion](#benchmark-the-champion) above for the canonical champion command and expected output.

| Flag | Default | Description |
|---|---|---|
| `--deals` | 200 | Number of duplicate deals (use 3000 for publishable CIs) |
| `--genome` | auto-detect | Path to a `best_genome_final.json` file |
| `--weights` | `-2.0,-1.0,-0.5,0.5,1.0,2.0` | Comma-separated shared weight sweep |
| `--seed` | 42 | RNG seed for reproducibility |
| `--output-dir` | genome's directory | Where to save `tournament_report.csv` |

### Comparing Runs

Cross-run comparison with 4-panel visualization (fitness, delta vs HeuristicBot, species diversity, network complexity):

```bash
uv run --project code python code/scripts/compare_runs.py                    # all runs
uv run --project code python code/scripts/compare_runs.py --runs 2026-06-03-2  # specific run
```

Saves `code/checkpoints/run_comparison.png`.

### Analyzing a Single Run

Per-run training plots (fitness curves, species counts, network complexity over time):

```bash
uv run --project code python code/scripts/plot_training.py \
  --stats code/checkpoints/2026-06-03-2/training_stats.csv \
  --out-dir code/checkpoints/2026-06-03-2
```

### Analyzing Expert Datasets

```bash
uv run --project code python code/scripts/analyze_dataset.py expert_states.npz      # quick stats with plots
uv run --project code python code/scripts/dataset_analysis.py expert_states.npz      # comprehensive text report
```

### Batch Rule Compilation

Compile rules for all genomes across all checkpoints:

```bash
uv run --project code python code/scripts/compile_all.py
```

### Extracting Rules

```bash
# Default weight (1.0)
code/target/release/sueca_wann compile-rules \
  --genome code/checkpoints/2026-06-03-2/genomes/best_genome_final.json \
  --output-dir code/checkpoints/2026-06-03-2

# Extract at a specific sweep weight (e.g. -1.0 for inhibitory rules)
code/target/release/sueca_wann compile-rules \
  --genome code/checkpoints/2026-06-03-2/genomes/best_genome_final.json \
  --output-dir code/checkpoints/2026-06-03-2 \
  --weight -1.0
```

Generates `compiled_rules.txt` (IF/THEN logic), `topology_graph.dot`, and `topology_graph.png`.

### Generating Expert Datasets

```bash
code/target/release/sueca_wann generate-dataset \
  --n-worlds 80 --search-depth 4 --target-count 10000 \
  --output expert_states.npz
```

### Removed: Optimize Weights

The old `optimize-weights` command was removed. Differential Evolution over independent per-connection continuous weights collapsed the Stage-B THRESHOLD champion (27.9% vs Elite) compared with the sweep-averaged sign-only champion (55.2% vs Elite, 1000 deals). Production evaluation uses the shared weight sweep.

### Pruning Champions

```bash
code/target/release/sueca_wann prune \
  --genome code/checkpoints/stageb/2026-06-25-1/genomes/best_genome_final.json \
  --deals 64 --seed 42 --tolerance 0.0 --passes 2
```

Pruning is game-delta-gated: Lead+Follow are evaluated jointly on fixed duplicate deals against a HeuristicBot baseline. A connection removal is kept only if average game-point delta stays within `--tolerance`; then disabled connections and dead-end hidden nodes are structurally compacted. Writes `<genome>_pruned.json`.

## Training Pipeline

End to end, a champion is produced in four stages: **(0)** generate an offline expert dataset,
**(1)** supervised pretraining (Phase 0), **(2)** Phase 0→1 transfer, **(3)** co-evolutionary
self-play (Phase 1). Two brains — **Lead** and **Follow** — are trained in parallel throughout
and routed at play time by `BeliefFeature::AmILeading`.

### Stage 0 — Expert Dataset Generation (offline, one-time)

`generate-dataset` produces a `.npz` (dataset version 2) of 35-feature belief states
plus a compact `PhiCtx` (trump, ego hand, trick cards) and a `best_cards` u64 mask,
sampled at the *current player's* turn (legal-move / perspective aligned). The
canonical dataset uses the **rollout teacher** (`solve_pimc_rollout_serial`): each
determinized world is finished with Elite playouts, yielding supra-Elite EVs. For
each state the mask holds every legal card within one stderr of the best EV (ties
are multi-label); forced and fully-ambiguous states are rejected. See the
[Dataset Pipeline](#dataset-pipeline-2026-06-recovery) section for the filters and flags.

### Stage 1 — Phase 0: Supervised Pretraining (gens 0 → `phase0_gens`)

WANNs are pretrained by **card-match accuracy** — the fraction of states where the
resolver's card (WANN → 6 knobs → `resolve_card_phi_utility_ctx`) is in the teacher's
`best_cards` mask. No game simulation, so fitness variance is zero and the search
focuses purely on establishing structural connectivity.

* **Dataset split.** The expert dataset is partitioned by the `AmILeading` flag into a
  `lead_dataset` and a `follow_dataset`; the Lead and Follow populations train on their own
  split. Lead states are lower-entropy (concentrated on aggressive/equity leads); follow states
  spread across efficient/equity actions.
* **Zero-connection start (PFS-NEAT).** Both populations begin with genomes carrying exactly
  **0 active connections**. The topology is grown only by mutations that prove their worth.
* **PFS validation (structural mutations only).** Mutations are classified as `Structural`
  (add-node, add-conn) or `NonStructural` (toggle, flip-sign, change-act, change-agg); only
  structural ones trigger validation. Adaptive 2-stage sampling: a quick **K=25** accuracy
  check first, and only borderline cases (within 2% of the parent) run the full
  `pfs_sample_size` (default **100**). Degraded mutations are pushed onto a FIFO
  `TabuVetoList` (size 1000) so the identical bad path is never re-evaluated. A single
  pre-allocated scratchpad is reused per child to avoid hot-path allocations.
* **Bloat control.** Neutral mutations that find no synergistic partner node are pruned by
  Pareto complexity domination.

### Stage 2 — Phase 0→1 Transfer (at `phase0_gens`)

HOF entries from Phase 0 are **re-evaluated under Phase 1 fitness** at the transition, so
supervised knowledge carries into self-play instead of being discarded. Uniqueness filtering
uses O(1) innovation-fingerprint hashing (`Genome::innovation_fingerprint()`), not O(pop²·E)
pairwise compatibility distance.

### Stage 3 — Phase 1: Co-evolutionary Self-Play (gens `phase0_gens` → `generations`)

* **Co-evolution.** Lead and Follow brains co-evolve: candidate Lead WANNs are paired with
  reference Follow champions and vice versa.
* **Dynamic routing.** Games are played card-by-card; a unified `Wann` simulator bot routes
  each decision slice to the correct brain:
  ```rust
  let network = if belief[BeliefFeature::AmILeading as usize] == 1.0 {
      lead_brain
  } else {
      follow_brain
  };
  ```
* **Fitness.** Raw game-point **delta vs HeuristicBot**, computed with Common Random Numbers
  (same deal, seat, and opponents) over seat rotations to isolate strategic skill from card
  luck. Partners/opponents are sampled from a mixed pool (HeuristicBot / HOF / MAP-Elites).
* **Selection.** Rank-based tournament selection; 50% of generations rank by
  (performance, simplicity) Pareto front to prevent bloat. Deals are re-seeded each generation
  to prevent memorization.

### Outputs

Each run writes a dated `code/checkpoints/YYYY-MM-DD-N/` containing `training_stats.csv`,
`training_state.bin` (Bincode, resumable), a `data/` directory of human-readable runtime
snapshots, and `genomes/` with `best_genome_final.json` and `hof_final.json`. See
[Checkpoint Structure](#checkpoint-structure).

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
    main.rs                 # CLI (train, benchmark, compile-rules, generate-dataset, prune)
    train.rs                # Training loop, Phase 0/1 dispatch
    evaluator.rs            # Bot simulation, delta-fitness evaluation
    wann_network.rs         # CSR-format WANN inference
    genome.rs               # Genome representation, topological sort
    population.rs           # Population, crossover, Pareto ranking, parallel breeding
    species.rs              # Compatibility distance, speciation
    mutations.rs            # NEAT mutation operators, innovation registry
    hall_of_fame.rs         # HOF with sampling
    map_elites.rs           # MAP-Elites quality-diversity archive
    prune.rs                # Game-delta-gated + structural genome pruning
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
