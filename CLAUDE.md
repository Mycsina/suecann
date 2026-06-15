# CLAUDE.md — Agent Instructions for Sueca WANN

## Project Overview

Neurosymbolic AI that evolves Weight-Agnostic Neural Networks (WANNs) to play **Sueca** (Portuguese trick-taking card game). Networks use logical gates instead of traditional activations, output abstract play intents (not cards), and are compiled into human-readable IF/THEN rules.

The training pipeline is a pure-Rust binary (`sueca_wann`) that calls into the Rust game engine (`sueca_solver`). Python is used only for cross-run comparison visualization (`scripts/compare_runs.py`).

## Project Documentation

- **`reference.md`** — Literature references with specific ideas taken from each paper.
- **`ideas.md`** — Completed milestones and future improvement paths.

## Tooling

- **Rust workspace**: `src/sueca_solver` (pure game engine, rlib only), `src/sueca_wann` (training binary + CLI) and `src/sueca_wasm` (WASM + interactive frontend in `frontend` folder)

- **Build Training Binary**: `cargo build -p sueca_wann --release`
- **Testing**: `cargo test --all`
- **Linting**: `cargo clippy --all`
- **Python deps**: numpy, pandas, matplotlib, seaborn (visualization only)

### WASM Pre-commit Hook

A `.githooks/pre-commit` hook auto-rebuilds WASM and stages the output whenever Rust source files (`src/sueca_*`) are committed. One-time setup per clone:

```bash
git config core.hooksPath .githooks
```

After this, every commit that touches Rust source automatically runs `wasm-pack build --target web` and `git add`s the regenerated `frontend/src/wasm/` files. No manual WASM rebuild step needed before push.

## Documentation Maintenance

- **Critical**: Whenever you make changes to the codebase that affect architecture, features, configuration, CLI, or module structure, you MUST update BOTH `README.md` and `CLAUDE.md` to reflect those changes. These two files are the project's source of truth and must stay in sync with the actual code.
- After any non-trivial code change, re-verify claims in both files against the codebase and fix any discrepancies found.

## Running Training

For hard computing tasks, you may choose to use arise. It has a RTX 3080, 64GB of ram and a i7-10750H.
To connect you can just use ssh arise, it also has a Projects folder, which you are to always use.

```bash
cargo build -p sueca_wann --release

# Run training (creates checkpoints/YYYY-MM-DD-N/)
./target/release/sueca_wann train --config configs/default.toml

# Resume from checkpoint
./target/release/sueca_wann train --config configs/default.toml --resume
```

Training creates dated run folders containing `training_stats.csv`, `training_state.bin`, a `data/` subdirectory (human-readable runtime snapshots — tabu lists, innovation registries, species summaries, MAP-Elites grid, population snapshots), and a `genomes/` subdirectory with `best_genome_final.json`, `hof_final.json`.

## Running Benchmarks

```bash
./target/release/sueca_wann benchmark --deals 200 --genome checkpoints/2026-06-03-2/genomes/best_genome_final.json
```

## Extracting Rules

```bash
./target/release/sueca_wann compile-rules --genome checkpoints/2026-06-03-2/genomes/best_genome_final.json --output-dir checkpoints/2026-06-03-2
```

Generates `compiled_rules.txt` (IF/THEN logic), `topology_graph.dot`, and `topology_graph.png` (via Graphviz `dot`).

## Generating Expert Dataset

```bash
# Canonical v6 dataset (rollout teacher, supra-Elite labels):
./target/release/sueca_wann generate-dataset \
  --n-worlds 200 --teacher rollout --target-count 15000 \
  --soft-balance-min-ratio 0.0 \
  --output expert_states_v6.npz
```

Generates PIMC expert states for Phase 0 pretraining. Samples only the current player's turn (not all 4 seats) to ensure legal-move / perspective alignment.
Outputs 35-feature belief states with 3-intent soft targets (MAX_FORCE, EFFICIENT_WIN, EQUITY_BUILDER).

**Teacher (`--teacher`, June 15):** `alphabeta` (default) is the deep PIMC with a *myopic* leaf eval
(`state.team_02_score`) — it only ties Elite, capping imitation at ≈Elite. `rollout` uses
`solve_pimc_rollout`: flat Monte-Carlo PIMC finishing each determinized world with **Elite playouts**.
By the rollout policy-improvement theorem (1-ply + Elite playout ≥ Elite) it is **supra-Elite (62% vs
Elite)** and ~1000× cheaper (15k states in ~11s). The canonical v6 dataset uses `--teacher rollout`.

**Labeling (June 14):** each decisive state is labeled with the intent whose *resolved card* has the
best PIMC EV among the 3 (statistically-tied intents → uniform multi-label; all 3 tied → reject).
The pre-filter rejects only fully-degenerate states (all 3 intents resolve to the same card). With
the styled resolver the decision is effectively binary EFFICIENT-vs-EQUITY (MAX_FORCE uniquely best
~1%, **0%** in the follow split), so generate with `--soft-balance-min-ratio 0.0` (a nonzero floor
makes the balancer hunt the unfillable MAX_FORCE bucket forever) and set `use_class_weighting = false`
(inverse-frequency weighting would give the ~1% bucket a huge noise weight).

### Migrating Legacy Datasets

Old datasets with 33-feature states and 4 intents must be regenerated (the feature indices changed).
For intent-only migration (4→3), use:

```bash
python scripts/migrate_intents.py db_w40_d3_mar03.npz --output db_w40_d3_mar03_v2.npz
```

## Optimizing Weights

```bash
./target/release/sueca_wann optimize-weights \
  --genome checkpoints/2026-06-03-2/genomes/best_genome_final.json \
  --deals 200 --generations 50
```

Uses Differential Evolution (pop=50, F=0.5, CR=0.7) to optimize independent per-connection continuous weights within [-2.0, 2.0]. Saves `optimized_weights.json` in the genome's directory. The benchmark command auto-detects this file and adds a WANN (Optimized) entry.

## Comparing Training Runs

```bash
uv run python scripts/compare_runs.py
uv run python scripts/compare_runs.py --runs 2026-06-03-2
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
Belief State (35 floats) → WANN (logical gates) → Oracle Intent (3 outputs) → Legal Subsystem → Card
```

**Crate dependency**: `sueca_wann` → `sueca_solver`. The solver is a pure game engine (rlib only, no PyO3). The wann crate contains WANN inference, evaluator, NEAT evolution, and CLI.

**Key modules in `sueca_solver`** (pure game engine):
- `engine.rs` — Bitboard game state, card logic, beats comparison
- `simulator.rs` — SuecaSimulatorGame wrapper with void tracking
- `belief.rs` — Belief state encoder (33 floats from game state)
- `heuristic.rs` — Card selection heuristics, intent-to-card resolver
- `pimc.rs` — Perfect Information Monte Carlo solver with late-game minimax switch
- `search.rs` — Alpha-beta search with Zobrist hashing and transposition table
- `rng.rs` — Shared LCG random number generator
- `constants.rs` — WANN layout dimension constants (INPUT_COUNT, OUTPUT_COUNT, etc.)

**Key modules in `sueca_wann`**:
- `main.rs` — CLI entry point (train / benchmark / compile-rules / generate-dataset / optimize-weights subcommands)
- `wann_network.rs` — CSR-format WANN inference with zero-allocation forward pass
- `evaluator.rs` — Bot simulation, delta-fitness evaluation
- `train.rs` — Training loop, Phase 0/1 evaluation, HOF transfer
- `genome.rs` — Genome representation, topological sort, CSR conversion
- `population.rs` — Population management, crossover, Pareto ranking, parallel breeding
- `species.rs` — Compatibility distance, speciation
- `mutations.rs` — NEAT mutation operators, innovation registry, tabu veto list, PFS-NEAT mutation classification
- `hall_of_fame.rs` — HOF management with sampling
- `map_elites.rs` — MAP-Elites quality-diversity archive with grid export
- `optimize.rs` — Differential Evolution weight optimization
- `runtime_data.rs` — Runtime state snapshots for checkpoint inspection and resume fidelity
- `constants.rs` — Evolutionary hyperparameters, feature/intent name mappings
- `benchmark.rs` — Tournament benchmarking
- `compile_rules.rs` — Rule compiler, Graphviz DOT export, PNG rendering
- `dataset_gen.rs` — PIMC expert dataset generation with ego-turn synchronization
- `dataset.rs` — Expert dataset loading (NPZ reader; rejects datasets that don't match INPUT_COUNT=35)
- `checkpoint.rs` — Training state serialization (Bincode)
- `config.rs` — TOML configuration loading

### Belief State (35 inputs, all in [0,1])

Redesigned to capture only information a human player can observe — no future-trick knowledge.
Pruned 8 low-signal features (side-suit Ace/7 tracking, raw suit power) in favor of 10 tactical-affordance
features (boss detection, suit counts, "can I win?" evaluation).

| # | Field | Type | Description |
|---|-------|------|-------------|
| 0 | Has_Led_Suit | Bool | Do I hold any card of the led suit? |
| 1 | Has_Trump | Bool | Do I hold any trump? |
| 2 | Led_Suit_Count | Float | Cards held in led suit / 10.0 |
| 3 | Trump_Count | Float | Trumps held / 10.0 |
| 4 | Hand_Point_Density | Float | My hand points / unplayed points |
| 5 | Am_I_Leading | Bool | 1st to play in trick |
| 6 | Am_I_Last_To_Play | Bool | 4th to play |
| 7 | Is_Partner_Winning | Bool | Is partner currently winning the trick? |
| 8 | Trick_Point_Value | Float | Points in current trick so far / 44.0 |
| 9 | Has_Trick_Been_Cut | Bool | Trump played when led suit ≠ trump |
| 10 | Partner_Void_Led | Bool | Partner known void in led suit? |
| 11 | Partner_Void_Trump | Bool | Partner known void in trump? |
| 12 | Any_Opp_Void_Led | Bool | Either opponent void in led suit? |
| 13 | Any_Opp_Void_Trump | Bool | Either opponent void in trump? |
| 14 | Led_Suit_Ace_Played | Bool | Led suit Ace already played? |
| 15 | Led_Suit_7_Played | Bool | Led suit 7 (manilha) already played? |
| 16 | Trump_Ace_Played | Bool | Trump Ace already played? |
| 17 | Holds_Boss_Led | Bool | Do I hold the highest unplayed card in led suit? |
| 18 | Holds_Boss_Trump | Bool | Do I hold the highest unplayed card in trump? |
| 19 | Can_Beat_Winner | Bool | Can any legal card beat the current winner? |
| 20 | Min_Winning_Cost | Float | Points of cheapest winning card / 11.0 (0 if N/A) |
| 21 | Min_Sacrifice_Cost | Float | Points of cheapest legal card / 11.0 |
| 22 | Game_Pts_Remaining | Float | Unplayed points / 120.0 |
| 23 | Trick_Number | Float | Current trick index / 9.0 |
| 24 | Trumps_Remaining | Float | Unplayed trump count / 10.0 |
| 25 | Score_Delta | Float | (our_pts − opp_pts + 120) / 240 |
| 26 | My_Void_Count | Float | Suits I'm void in / 3.0 |
| 27 | Longest_Side_Suit | Float | Max cards in any non-trump, non-led suit / 10.0 |
| 28 | Shortest_Side_Suit | Float | Min cards in any non-trump, non-led suit / 10.0 |
| 29 | Side0_Depletion | Float | Played cards of side-suit 0 / 10 |
| 30 | Side1_Depletion | Float | Played cards of side-suit 1 / 10 |
| 31 | Side2_Depletion | Float | Played cards of side-suit 2 / 10 |
| 32 | Points_Secured_Us | Float | Our team's secured game points / 120.0 |
| 33 | Known_Void_Suits_Count | Float | Suits where any player is known void / 4.0 |
| 34 | Depleted_Suits_Count | Float | Fully-depleted suits / 4.0 |

### Oracle Intents (3 outputs)

MIN_FORCE removed — EFFICIENT_WIN subsumes its useful follow-player behavior
(play cheapest winner, else concede). The resolver remaps WANN outputs 0,1,2
to legacy resolver intents 0,2,3 internally.

| ID | Intent | Action | Strategic Meaning |
|----|--------|--------|-------------------|
| 0 | MAX_FORCE | Elite + control dials | When trump-long, lead a low trump to *draw* trumps; else aggressive |
| 1 | EFFICIENT_WIN | == EliteHeuristicBot | Strong default — delegates exactly to `select_card_heuristic` |
| 2 | EQUITY_BUILDER | Elite + tempo dials | Lead shortest side-suit (build void); duck cheap tricks (≤2pts) when not last; preserve trump |

**Styled resolver (`select_card_styled` in `heuristic.rs`, June 14).** Each intent is a styled
deviation *around* the strong Elite policy, not a weak standalone tactic. `resolve_intent` maps WANN
output 0/1/2 → style 0/1/2 and calls `select_card_styled` (EFFICIENT delegates to Elite; MAX_FORCE/
EQUITY add the dials above). Because every intent shares Elite's core, each pure intent benchmarks
≈Elite individually (48/47/46% vs Elite, up from the old 20/37/25%) — so a policy collapse merely
*ties* Elite while good mixing exceeds it. This raised-floor design removed the Phase-1 collapse basin
and is what let the WANN finally beat Elite. The canonical champion is **v6 (`2026-06-14-2`): 52.1% ±
1.8% vs Elite, n=3000** — iso-strength with the v5 champion (52.7%) but **4.5× fewer hidden gates**
(29 vs 132), trained on the rollout-teacher dataset (prior project champion was 30.2%). The high floor
is double-edged: realisable headroom above Elite is only ≈8 points, so the learned overlay is small.

All intents are resolved to legal plays contextually by the heuristic resolver, guaranteeing 100% legality.
When WANN outputs tie, a random intent is chosen among the tied maximums (not deterministic argmax).

### WANN Constraints

- **Gene representation**: Connection genes `[5,N]` (innovation, src, dst, sign ∈ {+1,−1}, enabled). Node genes `[4,M]` (id, type, activation_fn, aggregation_fn).
- **Initialization**: 35 input + 1 bias + 3 output nodes (BIAS_ID=35, OUTPUT_START=36, FIRST_HIDDEN_ID=39). All genomes start with these base nodes and receive random connections.
- **Sign-only weights**: Connections carry only a sign (+1 or −1), not a learned weight. A shared weight W is used for evaluation. sign=-1 inverts the signal (1.0 - x) before aggregation.
- **Aggregation functions** (3 only): SUM=0, MIN(AND)=1, MAX(OR)=2. **No MEAN** — it causes float-precision issues at the THRESHOLD boundary.
- **Activation functions** (3 only): IDENTITY=0, NOT=1, THRESHOLD=2. **No SIGMOID** — it breaks IF/THEN rule extraction.
- **All node outputs clamped to [0, 1]**.
- **Shared weight sweep**: Evaluate each topology at W ∈ {-2.0, -1.0, -0.5, 0.5, 1.0, 2.0}, including negative weights for inhibitory rule expression. Average fitness across all six weights for true weight-agnostic evaluation.

### Training Pipeline

**Phase 0 (gens 0–`phase0_gens`): Supervised pretraining.**
* **Dataset Split:** Expert PIMC dataset is split into `lead_dataset` and `follow_dataset` using the `BeliefFeature::AmILeading` flag.
* **PFS-NEAT:** Populations start with exactly 0 active connections. Mutations are classified as `Structural` (add_node, add_conn) or `NonStructural` (toggle, flip_sign, change_act, change_agg). Only structural mutations trigger PFS validation. Adaptive 2-stage sampling: quick K=25 check first; only borderline cases (within 2% accuracy) run the full configurable `pfs_sample_size` (default 100, down from original 1000). Degraded mutations are logged into a FIFO `TabuVetoList` of size 1000.
* **Fitness:** Supervised classification accuracy on respective splits. No game simulation.
* **Scratchpad reuse:** PFS evaluations reuse a single pre-allocated scratchpad buffer per child, avoiding repeated allocations in the breeding hot path.

**Phase 1 (gens `phase0_gens`–`generations`): Co-evolutionary Self-play.**
* **Co-Evolution:** Lead and Follow brains co-evolve. Matches pair candidate Lead WANNs with reference Follow WANN champions, and vice versa.
* **Dynamic Routing:** Games are played trick-by-trick and card-by-card. The simulator dynamically routes queries to the Lead or Follow brain on every decision slice based on `belief[BeliefFeature::AmILeading as usize]`.
* **Fitness:** Raw game-point delta vs HeuristicBot. Partners/opponents sampled from HOF and MAP-Elites. Delta computed via Common Random Numbers on the same duplicate deals (seat rotations).

**Phase 0→1 HOF transfer**: HOF entries are re-evaluated under Phase 1 fitness at the transition point, preserving knowledge from supervised pretraining. Uniqueness filtering uses O(1) innovation fingerprint hashing (`Genome::innovation_fingerprint()`) instead of O(pop²·E) pairwise compatibility distance — hashes the sorted list of (innovation, sign) of enabled connections plus hidden node (id, activation, aggregation).

**Speciation**: First-fit (lazy) assignment: each genome is tested against species serially until one accepts it, reducing average cost from O(P · Sp · E) to O(P · Sp_checked · E). Species count capped at `max_species` (default 20) via post-speciation merging of smallest species into closest larger neighbours.

**Parallelism**: Rayon parallelizes genome→WANN conversion, speciation distance computation, Pareto domination detection, stagnation updates, and offspring generation. Innovation registry uses a Mutex for thread-safe mutation operations.

**Per-generation profiling**: The training loop prints per-phase timings each generation: `phase`, `convert`, `eval`, `stats`, `reseed`, `breed`, `ckpt`. The CSV `elapsed_sec` column captures total wall-clock time including speciation and breeding (not just evaluation).

### Evolution

- **Duplicate deals**: Deals per generation × 4 seat rotations. Deals are **re-seeded each generation** (`seed=gen`) to prevent overfitting.
- **Delta-fitness**: Each genome compared against HeuristicBot on the exact same deal/seat/opponents (Common Random Numbers). Eliminates deal-luck variance.
- **Rank-based selection**: Raw fitness converted to normalized ranks before tournament selection for noise robustness.
- **Multi-objective Pareto ranking**: 50% of the time, rank by (performance, simplicity) Pareto front with lexicographic tie-breaking; 50% by performance only. Prevents bloat while maintaining selection pressure.
- **Hall of Fame**: Frozen champion archive (size 50). Sampled as partners/opponents during Phase 1.
- **MAP-Elites**: 10×10 grid archiving behavioral specialists by intent preference and aggression. Sampled as opponents with 50% probability when HOF/MAP-Elites is selected (vs OldHeuristicBot baseline).
- **Mutations**: Add node, add connection, toggle connection, flip sign, change activation, change aggregation. No weight mutation. Classified as `Structural` (add_node, add_conn — triggers PFS) or `NonStructural` (all others — skips PFS).

## Code Conventions

- Rust source: `src/sueca_solver/src/` (engine), `src/sueca_wann/src/` (training + CLI).
- Python is visualization-only: `scripts/compare_runs.py` (cross-run plots).
- Tests must be thorough — test invariants (e.g., total points = 120), edge cases, and boundary values.
- Rust functions: `#[inline(always)]` on hot-path bitboard/WANN ops.

## Common Pitfalls

1. **Never leak opponent hand data** into visible state or belief vector.
2. **Rank ordering is NOT standard** — 7 beats K in Sueca. Use `CARD_RANK` lookups, not card face values.
3. **Partner = (seat + 2) % 4**, not seat ± 1.
4. **Counter-clockwise**: after seat 0, it's seat 3, not seat 1.
5. **Void tracking is per-suit**: a player void in hearts may still have diamonds.
6. **Duplicate deals must differ across generations** — same seed within a gen for fairness, different seed between gens to prevent memorization.
7. **argmax tie-breaking**: When WANN outputs tie, use random choice among tied maximums, NOT deterministic argmax.
8. **Delta-fitness baseline bot must see the same cards**: The baseline plays the exact same seat rotation with the same deal to ensure valid comparison.
9. **Build from repo root**: Always use `cargo build -p sueca_wann --release`. There is no `.so` / FFI build step.
10. **Dataset ego-turn sync**: `legal_moves()` always returns moves for `game.state.current_player`. Never loop over all 4 seats at a frozen game state — the belief, legal mask, and intent resolver must all reference the active player.
11. **Weight Sweep Discrepancy**: Evolving on a single weight (e.g. `[1.0]`) while benchmarking on a multi-weight sweep (e.g. `[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]`) leads to severe performance degradation. Ensure configuration aligns with evaluation goals.