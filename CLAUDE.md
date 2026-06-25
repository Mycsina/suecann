# CLAUDE.md — Agent Instructions for Sueca WANN

## Project Overview

Neurosymbolic AI that evolves Weight-Agnostic Neural Networks (WANNs) to play **Sueca** (Portuguese trick-taking card game). Networks use logical gates instead of traditional activations, output 6 φ-utility knobs (not cards), and are compiled into human-readable IF/THEN rules.

The training pipeline is a pure-Rust binary (`sueca_wann`) that calls into the Rust game engine (`sueca_solver`). Python is used only for cross-run comparison visualization (`code/scripts/compare_runs.py`).

## Project Documentation

- **`reference.md`** — Literature references with specific ideas taken from each paper.
- **`ideas.md`** — Completed milestones and future improvement paths.

## Tooling

- **Rust workspace**: `code/src/sueca_solver` (pure game engine, rlib only), `code/src/sueca_wann` (training binary + CLI) and `code/src/sueca_wasm` (WASM + interactive frontend in `frontend` folder)

- **Build Training Binary**: `cargo build -p sueca_wann --release`
- **Testing**: `cargo test --all`
- **Linting**: `cargo clippy --all`
- **Python deps**: numpy, pandas, matplotlib, seaborn (visualization only)

### WASM Pre-commit Hook

A `code/.githooks/pre-commit` hook auto-rebuilds WASM and stages the output whenever Rust source files (`code/src/sueca_*`) are committed. One-time setup per clone:

```bash
git config core.hooksPath code/.githooks
```

After this, every commit that touches Rust source automatically runs `wasm-pack build --target web` and `git add`s the regenerated `code/frontend/src/wasm/` files. No manual WASM rebuild step needed before push.

## Documentation Maintenance

- **Critical**: Whenever you make changes to the codebase that affect architecture, features, configuration, CLI, or module structure, you MUST update BOTH `README.md` and `CLAUDE.md` to reflect those changes. These two files are the project's source of truth and must stay in sync with the actual code.
- After any non-trivial code change, re-verify claims in both files against the codebase and fix any discrepancies found.

## Running Training

For hard computing tasks, you may choose to use arise. It has a RTX 3080, 64GB of ram and a i7-10750H.
To connect you can just use ssh arise, it also has a Projects folder, which you are to always use.

```bash
cargo build -p sueca_wann --release

# Run training (creates code/checkpoints/YYYY-MM-DD-N/)
./target/release/sueca_wann train --config code/configs/default.toml

# Resume from checkpoint
./target/release/sueca_wann train --config code/configs/default.toml --resume
```

Training creates dated run folders containing `training_stats.csv`, `training_state.bin`, a `data/` subdirectory (human-readable runtime snapshots — tabu lists, innovation registries, species summaries, MAP-Elites grid, population snapshots), and a `genomes/` subdirectory with `best_genome_final.json`, `hof_final.json`.

## Running Benchmarks

```bash
./target/release/sueca_wann benchmark --deals 200 --genome code/checkpoints/2026-06-03-2/genomes/best_genome_final.json
```

## Extracting Rules

```bash
./target/release/sueca_wann compile-rules --genome code/checkpoints/2026-06-03-2/genomes/best_genome_final.json --output-dir code/checkpoints/2026-06-03-2
```

Generates `compiled_rules.txt` (IF/THEN logic), `topology_graph.dot`, and `topology_graph.png` (via Graphviz `dot`).

## Generating Expert Dataset

```bash
# Stage B card-match dataset (rollout teacher → free-card target, ~81% ceiling):
./target/release/sueca_wann generate-dataset \
  --n-worlds 200 --teacher rollout --target-count 15000 \
  --soft-balance-min-ratio 0.0 \
  --output expert_states_v7.npz
```

Generates card-match expert states for Phase 0 pretraining (dataset version 2).
Samples the current player's turn via a mid-trick random walk (so ego-turn
perspective, legal moves, and φ context all stay aligned). Each record stores:
35-feature belief + a compact `PhiCtx` (trump, ego hand, trick cards, trick len)
+ a `best_cards` u64 bitmask of the teacher's best legal cards.

**Teacher (`--teacher`):** `rollout` (canonical) uses `solve_pimc_rollout_serial`:
flat Monte-Carlo PIMC finishing each determinized world with **Elite playouts** —
supra-Elite by the rollout policy-improvement theorem (1-ply + Elite playout ≥
Elite). `alphabeta` is the deep PIMC with a myopic leaf eval (only ties Elite).

**Labeling (Stage B, card-match):** for each state the rollout teacher returns an
EV for every legal card; `best_cards` = the legal cards within one stderr of the
max EV (statistical ties are multi-label — the resolver gets full credit for any
tied-best card). Forced (single-legal) and fully-ambiguous (all cards tied)
states are rejected. Phase-0 fitness is the fraction of states where the
resolver's card is in the mask. `soft_balance_min_ratio` and `use_class_weighting`
are vestigial under card-match fitness (kept for config/CLI compatibility).

### Migrating Legacy Datasets

Old datasets (33-feature states, or the 3-intent soft-label format, or any
version < 2) are **rejected at load time** by the version gate — regenerate.
The old `scripts/migrate_intents.py` flow is obsolete; there is no in-place
3-intent → 6-knob migration path (the target representation changed entirely).

## Optimizing Weights

```bash
./target/release/sueca_wann optimize-weights \
  --genome code/checkpoints/2026-06-03-2/genomes/best_genome_final.json \
  --deals 200 --generations 50
```

Uses Differential Evolution (pop=50, F=0.5, CR=0.7) to optimize independent per-connection continuous weights within [-2.0, 2.0]. Saves `optimized_weights.json` in the genome's directory. The benchmark command auto-detects this file and adds a WANN (Optimized) entry.

> **Caveat (2026-06-25): on the Stage-B THRESHOLD champion this HURTS badly.**
> Per-connection continuous weights fight the weight-agnostic property the
> topology was evolved under (sign-only + shared sweep). DE overfits the 200
> training deals against an all-Heuristic field and the resulting weights
> collapse: WANN (Optimized) benchmarks at 27.9% vs Elite vs 55.2% for the
> sweep-averaged Champion (1000 deals). The sweep-averaged sign-only eval is the
> correct way to run a WANN — `optimize-weights` is retained for experimentation
> but is not recommended for production champions.

## Pruning a Champion

```bash
./target/release/sueca_wann prune \
  --genome code/checkpoints/stageb/2026-06-25-1/genomes/best_genome_final.json \
  --dataset code/expert_states_v7.npz --tolerance 0.0 --passes 2
```

For each brain, iteratively disables enabled connections whose removal keeps
Phase-0 card-match within `--tolerance` (on that brain's `AmILeading` split),
then `Genome::prune_structural` compacts disabled connections + dead-end hidden
nodes (behaviour-preserving). Writes `<genome>_pruned.json`. On the Stage-B
champion at `--tolerance 0.0`: Lead 124→6 conns, Follow 73→10, card-match
exactly preserved, game strength 52.5% vs Elite (from 55.2%) — i.e. a much
smaller, interpretable genome at a ~3-pt strength cost. Card-match is a leaky
proxy for game strength (over-prunes); a game-delta-gated prune is the natural
next improvement. Lower `--tolerance` for safer pruning.

## Comparing Training Runs

```bash
uv run python code/scripts/compare_runs.py
uv run python code/scripts/compare_runs.py --runs 2026-06-03-2
```

Saves `code/checkpoints/run_comparison.png` with 4 panels: fitness, delta vs HeuristicBot, species diversity, network complexity.

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
Belief State (35 floats) → WANN (logical gates) → φ-Utility Knobs (6 outputs) → φ-Resolver → Card
```

**Resolver/action-space overhaul (Stage B, 2026-06-19).** The WANN output layer
was widened from a 3-intent selector to a 6-dimensional continuous **knob**
vector, one per hand-designed card-utility feature φ(card, state). The resolver
plays `argmax_{legal} Σ_k knob_k · φ_k(card, state)` (see
`heuristic::resolve_card_phi_utility`). This was driven by a measured ceiling
decomposition: the 3-intent vocabulary capped at ~57% vs Elite while the free-card
ceiling is ~81%, and a 6-feature linear utility recovers essentially all of that
~24-point vocabulary gap. Design spec: `docs/superpowers/specs/2026-06-19-resolver-overhaul-design.md`.

**Substrate decision (THRESHOLD outputs).** The φ-knobs are produced by
sweep-averaging the **IDENTITY** output nodes and remapping `[0,1]→[-1,1]` via
`outputs_to_knobs` (2·avg−1). This was the original choice and it is **broken**:
the symmetric sweep `{−2,−1,−0.5,0.5,1,2}` clamps the three negative-weight
outputs to 0, dragging the averaged output ≤ 0.5 → knob ≤ 0. A `+1` bias→output
connection yields **knob −0.167**, so the WANN *cannot express a positive knob*
and collapses to a poor all-negative constant (Phase-0 card-match 0.45, below
even the best constant 0.54). Confirmed by Probe B2 (champion: all 6 knobs
≤ 0, max = 0.000) and Probe C (representational probe).

**Fix (Probe C, 2026-06-25): output nodes use THRESHOLD, not IDENTITY.** A
THRESHOLD output driven by bias fires consistently across every sweep weight →
averaged output 1.0 → **knob +1.0 reachable**. This lifts Phase-0 card-match
**0.45 → 0.643** (≈ the 0.65 belief+φ GBM ceiling) in 60 gens; the new champion's
WINS knob is +0.979 (was −0.544) and knobs are now belief-conditioned (the
mean-constant card-match ~0.52 *lags* the WANN's 0.64 — the gap is the belief
signal the old substrate couldn't use). THRESHOLD quantizes knobs to 7 levels;
the quantization cost is ~0.007 (Probe B1). Configured via
`population.output_activation = "threshold"` and
`mutation.allow_output_act_mutation = true` (so `change_activation` can reach
output nodes — previously excluded). The canonical **symmetric** weight sweep is
**preserved** (full WANN weight-agnosticism); only the output activation changes.
The closed activation set remains {IDENTITY, NOT, THRESHOLD}; **no SIGMOID**.

**Alternative substrate (asymmetric sweep) — TESTED INFERIOR (A/B, 2026-06-25).**
Setting `sweep_weights = [0.5,1,2]` (positive-only) with IDENTITY outputs also
unblocks positive knobs (a +1 bias yields knob +0.667, *continuously*) — Phase-0
card-match 0.624 (config `ab_asym_identity.toml`). A full 600-gen A/B run
benchmarked the resulting champion at **38.1% vs Elite / 56.0% vs OldHeuristic
(1000 deals)** vs the THRESHOLD champion's **55.2% / 68.0%** — i.e. the
continuous-knob substrate is ~17 pts worse on game strength despite only ~2 pts
behind on Phase-0 card-match. The "continuous knobs generalize better"
hypothesis did not hold; THRESHOLD's stronger supervised signal wins. Kept as a
config option (`output_activation = "identity"`, `sweep_weights = [0.5,1,2]`)
but not recommended.

**Crate dependency**: `sueca_wann` → `sueca_solver`. The solver is a pure game engine (rlib only, no PyO3). The wann crate contains WANN inference, evaluator, NEAT evolution, and CLI.

**Key modules in `sueca_solver`** (pure game engine):
- `engine.rs` — Bitboard game state, card logic, beats comparison
- `simulator.rs` — SuecaSimulatorGame wrapper with void tracking
- `belief.rs` — Belief state encoder (35 floats from game state)
- `heuristic.rs` — Card selection heuristics, **φ-utility resolver** (`compute_phi`, `PhiCtx`, `resolve_card_phi_utility`); legacy 3-intent styled resolver retained for ceiling diagnostics
- `pimc.rs` — Perfect Information Monte Carlo solver with late-game minimax switch; `solve_pimc_rollout[_serial]` (supra-Elite rollout teacher)
- `search.rs` — Alpha-beta search with Zobrist hashing and transposition table
- `rng.rs` — Shared LCG random number generator
- `constants.rs` — WANN layout dimensions (`INPUT_COUNT=35`, `OUTPUT_COUNT=6` φ-knobs, `PHI_FEATURE_COUNT=6`, `PhiFeature` enum)

**Key modules in `sueca_wann`**:
- `main.rs` — CLI entry point (train / benchmark / compile-rules / generate-dataset / optimize-weights / prune subcommands)
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
- `prune.rs` — Behavioural (card-match-gated) + structural genome pruning
- `runtime_data.rs` — Runtime state snapshots for checkpoint inspection and resume fidelity
- `constants.rs` — Evolutionary hyperparameters, feature/φ-knob name mappings
- `benchmark.rs` — Tournament benchmarking
- `compile_rules.rs` — Rule compiler, Graphviz DOT export, PNG rendering
- `dataset_gen.rs` — Card-match expert dataset generation (rollout teacher → `best_cards` mask + `PhiCtx` per state; ego-turn synchronization)
- `dataset.rs` — Expert dataset loading (NPZ reader; version + INPUT_COUNT=35 gate rejects stale datasets)
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

### φ-Utility Knobs (6 outputs)

The WANN emits 6 knobs `w ∈ [-1,1]^6`, one per card-utility feature
φ(card, state). The resolver plays `argmax_{legal} Σ_k w_k · φ_k(card, state)`
(`heuristic::resolve_card_phi_utility_ctx`). Knobs are produced by sweep-averaging
the THRESHOLD output nodes (each fires 0/1 per sweep weight; the average over the
symmetric sweep lies in [0,1] and is remapped via `2x-1`). THRESHOLD (not IDENTITY)
is required so a positive φ-knob is reachable — see "Substrate decision" above.

| ID | Knob | φ feature | Meaning (positive knob → prefer…) |
|----|------|-----------|-----------------------------------|
| 0 | RANK | `CARD_RANK/9` | high-rank cards |
| 1 | POINTS | `CARD_POINTS/11` | high-point cards |
| 2 | TRUMP | is trump | trumps |
| 3 | WINS | would beat current winner | winning the trick now |
| 4 | CAPTURES | trick points/30 if wins | capturing the points on the table |
| 5 | VOID | last card of its suit in hand | playing the suit's last card (build/keep a void) |

`PhiCtx` (trump, ego hand, trick cards, trick len) is the compact context φ is
computed from — built live during play and serialized into the Phase-0 dataset.
`PhiCtx::legal()` derives the follow-suit legal set, so the resolver always
returns a legal card.

**Why the overhaul.** Measured ceilings vs Elite (n=500–2000): WANN champion 52.3%,
3-intent vocabulary envelope ~57%, free-card ceiling ~81%, and a continuous linear
utility over these 6 features reaches ~81% (= free-card). The 3-intent vocabulary
was the dominant ~24-point bottleneck; the φ-resolver recovers essentially all of
it. Full decomposition in the design spec.

**Legacy 3-intent resolver retained** (`select_card_styled`, `resolve_intent`) for
the `OracleEnvelope` ceiling diagnostic only — it is no longer the production
action vocabulary. The prior canonical champion **v6 (`2026-06-14-2`): 52.1% ± 1.8%
vs Elite** was trained under the old 3-intent system and its genome is **not
loadable** after this refactor (OUTPUT_COUNT and FIRST_HIDDEN_ID changed). The
Stage B system is a fresh training run.

**Canonical Stage-B champion (`stageb/2026-06-25-1`, THRESHOLD substrate):
55.2% ± 3.1% vs Elite, 68.0% ± 2.9% vs OldHeuristic (1000-deal benchmark);
Phase-1 delta +4.6 vs HeuristicBot** (peak). Beats the broken-substrate Stage-B
run (flat −6 delta) and the legacy v6 (52.1% vs Elite).

### WANN Constraints

- **Gene representation**: Connection genes `[5,N]` (innovation, src, dst, sign ∈ {+1,−1}, enabled). Node genes `[4,M]` (id, type, activation_fn, aggregation_fn).
- **Initialization**: 35 input + 1 bias + 6 output nodes (BIAS_ID=35, OUTPUT_START=36, FIRST_HIDDEN_ID=42). All genomes start with these base nodes and receive random connections.
- **Sign-only weights**: Connections carry only a sign (+1 or −1), not a learned weight. A shared weight W is used for evaluation. sign=-1 inverts the signal (1.0 - x) before aggregation.
- **Aggregation functions** (3 only): SUM=0, MIN(AND)=1, MAX(OR)=2. **No MEAN** — it causes float-precision issues at the THRESHOLD boundary.
- **Activation functions** (3 only): IDENTITY=0, NOT=1, THRESHOLD=2. **No SIGMOID needed.** Output nodes use THRESHOLD (Probe C fix) so positive φ-knobs are reachable under the symmetric sweep; `change_activation` may target output nodes (`allow_output_act_mutation`). See "Substrate decision" above.
- **All node outputs clamped to [0, 1]** (output knobs are then remapped to [-1,1]).
- **Shared weight sweep**: Evaluate each topology at W ∈ {-2.0, -1.0, -0.5, 0.5, 1.0, 2.0}, including negative weights for inhibitory rule expression. Average fitness across all six weights for true weight-agnostic evaluation.

### Training Pipeline

End to end: **Stage 0** generate the offline expert dataset (`generate-dataset`, rollout
teacher — see "Generating Expert Dataset" above) → **Stage 1** Phase 0 supervised pretraining
→ **Stage 2** Phase 0→1 HOF transfer → **Stage 3** Phase 1 self-play. Lead and Follow brains
are trained in parallel throughout and routed at play time by `BeliefFeature::AmILeading`.

**Phase 0 (gens 0–`phase0_gens`): Supervised pretraining.**
* **Dataset Split:** Expert PIMC dataset is split into `lead_dataset` and `follow_dataset` using the `BeliefFeature::AmILeading` flag.
* **PFS-NEAT:** Populations start with exactly 0 active connections. Mutations are classified as `Structural` (add_node, add_conn) or `NonStructural` (toggle, flip_sign, change_act, change_agg). Only structural mutations trigger PFS validation. Adaptive 2-stage sampling: quick K=25 check first; only borderline cases (within 2% accuracy) run the full configurable `pfs_sample_size` (default 100, down from original 1000). Degraded mutations are logged into a FIFO `TabuVetoList` of size 1000.
* **Fitness:** Card-match accuracy — the fraction of states where the resolver's card (WANN → 6 knobs → `resolve_card_phi_utility_ctx`) is in the teacher's `best_cards` mask. No game simulation.
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
- **MAP-Elites**: 10×10 grid archiving behavioral specialists by win-preference (mean φ-WINS knob) and aggression. Sampled as opponents with 50% probability when HOF/MAP-Elites is selected (vs OldHeuristicBot baseline).
- **Mutations**: Add node, add connection, toggle connection, flip sign, change activation, change aggregation. No weight mutation. Classified as `Structural` (add_node, add_conn — triggers PFS) or `NonStructural` (all others — skips PFS).

## Code Conventions

- Rust source: `code/src/sueca_solver/src/` (engine), `code/src/sueca_wann/src/` (training + CLI).
- Python is visualization-only: `code/scripts/compare_runs.py` (cross-run plots).
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