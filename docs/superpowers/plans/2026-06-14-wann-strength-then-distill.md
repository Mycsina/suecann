# Sueca WANN — Strength-then-Distill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push the evolved WANN clearly above EliteHeuristicBot (target ≥58% win rate), then distill the strong-but-opaque network into a genuinely human-readable rule set — proving the WANN architecture can both *reach* near-optimal play and *be compressed* into interpretable logic.

**Architecture:** Two decoupled stages. **Stage 1 (Strength):** build a cheap *supra-Elite* teacher (flat Monte-Carlo PIMC with Elite playouts), use it to label a richer dataset, extend Phase-0 supervised bootstrap, then Phase-1 self-play — complexity unconstrained. **Stage 2 (Distill):** compress the strong "reference organism" into a small interpretable network/ruleset via knowledge distillation + logic minimization, measuring the strength↔interpretability trade-off explicitly.

**Tech Stack:** Rust (`sueca_solver` engine, `sueca_wann` training/CLI), Python (sklearn/numpy for surrogate-model probes and analysis only). No FFI. Validation goes through the existing training-free diagnostic harness in `src/sueca_wann/src/evaluator.rs` before any full training run.

---

## Why this plan looks the way it does (read first)

This section is context for an engineer with zero project history. It is not optional reading — the *ordering* and *gates* below exist for measured reasons.

### Three decoupled axes (do not conflate them)
The project has been measuring one number (win% vs Elite) but actually cares about three:
1. **Strength** — win rate vs EliteHeuristicBot in the seat-rotated `benchmark` command. Current champion: **52.7% ± 1.8%** (n=3000).
2. **Fidelity** — for distillation: how often a *student* network reproduces the *teacher's* chosen intent. New metric introduced in Stage 2.
3. **Interpretability** — how few/shallow the compiled logic rules are. Current champion: lead brain = 107 nodes / ~40 active hidden logic gates / 91 enabled connections → effectively a **black box** (verify yourself: `compile-rules` then read `compiled_rules_lead.txt`; `EQUITY_BUILDER` is a 4-term sum over a 40-gate DAG).

The user's key insight, which this plan adopts: **stop trading these off inside one evolutionary run.** Maximize strength first (Stage 1, complexity free), then compress (Stage 2). Pareto/simplicity pressure applied *during* the strength run "almost certainly hurts results"; applied as a *separate distillation objective* it does not compete with strength discovery.

### What a feedforward WANN can and cannot learn (drives feature design)
A WANN here is a *feedforward DAG of discrete logic gates* over a belief snapshot. It has **no memory, no iteration, no search**. Therefore:
- It **cannot** learn multi-step hidden-state inference ("opponent followed suit with a low card under pressure → by assumed-optimal play they're probably out of high cards in suit X → I should…"). That is sequential probabilistic reasoning; it is not expressible as a fixed shallow circuit over raw features.
- It **can** learn simple rules *over features that already summarize* such inference.

**Consequence:** any "extrapolate opponent cards" intelligence must be computed in the **belief encoder** (`src/sueca_solver/src/belief.rs`) as a deterministic, auditable function of *strictly public information* (void observations, played cards, follow-suit failures, count constraints) — the same legal info a human uses — and exposed as a named feature. The WANN then learns thin rules on top. This is identical in status to the existing `Holds_Boss_Led` feature. **Hard constraint (CLAUDE.md Pitfall #1): never read any player's actual hand in belief.rs; only public/observed state.**

This also *helps* interpretability: a feature that captures the right concept lets a 3-gate rule replace a 40-gate nested mess. Good features reduce required network complexity — they serve Stage 1 *and* Stage 2.

### Disagreements with the initial framing (surfaced honestly)
- "Improving Phase 1 gets us closer to rollout-PIMC." *Partly.* Phase-1 self-play with game-outcome fitness discovers Elite-*exploiting reactive* policies, but a feedforward WANN cannot itself learn lookahead/rollout. The "rollout intelligence" must arrive via the teacher's labels (Stage 1) or via lookahead-summarizing features (belief.rs). Phase-1 is necessary but bounded by features+labels; it is not a substitute for a better teacher.
- "Deep PIMC is expensive." *True, and that's why this plan does not use deep PIMC.* The weakness is the **myopic leaf eval** (`search.rs:184` and `:193` both `return state.team_02_score` — points captured *so far*, no positional estimate). The fix is a **shallow flat rollout with the Elite policy as playout**, which is *cheaper* than deep alpha-beta and provably ≥ Elite (rollout policy-improvement theorem). We do **not** patch `alpha_beta` (a prior attempt did, scored only 40% with a naive lowest-rank playout, broke two solver tests, and was reverted). We build a separate solver instead.

### Decision gates (the plan branches on measured results)
- **Gate 1** (end of Stage 1, Task 3): does the rollout teacher beat Elite by a *clear* margin (≥55% in the harness) at acceptable cost? If **no**, stop and reconsider (do not regenerate the dataset). If **yes**, proceed to relabel + retrain.
- **Gate 2** (end of Stage 1, Task 7): does the retrained champion reach ≥56% vs Elite (n≥2000)? If **no**, the bottleneck is features/training, not teacher → pivot to the belief-feature track (Task 8) before Stage 2.
- **Gate 3** (Stage 2): pick the distillation method that gives the best strength-at-fixed-interpretability on the measured trade-off curve.

---

## File Structure (what gets created/modified)

**Stage 1**
- `src/sueca_solver/src/pimc.rs` — ADD `solve_pimc_rollout(...)`: flat MC, Elite playouts, paired-difference early exit. Reuses existing `sample_world` (pimc.rs:102) and `PimcResult` (pimc.rs:173). Does **not** modify `solve_pimc` or `alpha_beta`.
- `src/sueca_solver/src/heuristic.rs` — ADD `rollout_score_gamestate(...)` helper (a `GameState`-level Elite playout to terminal) **iff** the rollout is done on `GameState`; otherwise reuse `SuecaSimulatorGame` + `select_card_heuristic` (preferred — see Task 1).
- `src/sueca_wann/src/evaluator.rs` — EXTEND the `diagnostic_pimc_vs_elite` test (evaluator.rs:751) to add a `RolloutPimc` policy and print its win% vs Elite. Reuses `diag_rollout_elite` (evaluator.rs:635) and `diag_play` (evaluator.rs:684).
- `src/sueca_wann/src/dataset_gen.rs` — SWITCH the labeler from `solve_pimc` to `solve_pimc_rollout` behind a CLI flag `--teacher rollout|alphabeta` (default keep `alphabeta` until Gate 1 passes).
- `src/sueca_wann/src/main.rs` — ADD the `--teacher` flag to the `generate-dataset` subcommand.
- `configs/default.toml` — bump `phase0_gens` (extend Phase 0) **only after** Gate 1 + better labels exist.

**Stage 2** (gated; detailed as candidate experiments, see Stage 2 section)
- `src/sueca_wann/src/distill.rs` — NEW module: teacher→student intent-label dump and student-fitness = teacher-agreement.
- `scripts/extract_rules_tree.py` — NEW: decision-tree / rule-list surrogate of the teacher (interpretability upper bound).
- `src/sueca_wann/src/compile_rules.rs` — EXTEND: constant-fold dead gates (`hidden_x = 0.0`), report rule-complexity metrics (gate count, max depth, literal count).

---

## STAGE 1 — Strength (cheap supra-Elite teacher → relabel → extend Phase 0 → Phase 1)

### Task 1: Flat-rollout PIMC solver (`solve_pimc_rollout`)

**Files:**
- Modify: `src/sueca_solver/src/pimc.rs` (add new public fn; reuse `sample_world` at :102, `PimcResult` at :173, `constraints_feasible`)
- Test: `src/sueca_solver/src/pimc.rs` (inline `#[cfg(test)]` module)

**Design (why this shape):** The Elite policy `select_card_heuristic(game: &SuecaSimulatorGame, seat)` runs on `SuecaSimulatorGame`, not `GameState`. The cleanest rollout therefore builds a determinized `SuecaSimulatorGame` per world (all four hands known from `sample_world`) and plays it to terminal with Elite for all seats — reusing the exact primitive already proven in `diag_rollout_elite` (evaluator.rs:635-643). No `alpha_beta` change → no risk to the search engine.

The value of a candidate card in a world = Team 0&2 final score after: play the card, then Elite plays every seat to the end. EV per card = mean over worlds. This is **1-ply + Elite-rollout ≥ Elite** by the policy-improvement theorem.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod rollout_tests {
    use super::*;

    // A forced-move position must return exactly that move with no panic.
    #[test]
    fn test_rollout_forced_move_returns_single() {
        // my_hand has exactly one card of the led suit -> legal_count == 1
        let my_seat = 0u8;
        let my_hand = 1u64 << 0;          // single card (suit 0, rank 0)
        let played = 0u64;
        let voids = [0u8; 4];
        let target = [1u8, 0, 0, 0];      // only I have a card left
        let res = solve_pimc_rollout(
            my_seat, my_hand, played, voids, target,
            /*trump*/0, /*led_suit*/0, &[], /*current_player*/0,
            /*winner*/0, /*best_card*/0, /*team_scores*/[0,0],
            /*trick_number*/9, /*n_worlds*/8, /*seed*/42,
        );
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].card, 0);
    }

    // With >1 legal move, every legal move must get an EV in [0,120] and the
    // returned vector must cover all legal moves.
    #[test]
    fn test_rollout_evs_bounded_and_complete() {
        let my_seat = 0u8;
        // two cards in suit 0: ranks 0 and 1
        let my_hand = (1u64 << 0) | (1u64 << 1);
        let played = 0u64;
        let voids = [0u8; 4];
        let target = [2u8, 1, 1, 1];
        let res = solve_pimc_rollout(
            my_seat, my_hand, played, voids, target,
            0, 4 /*led_suit=4 means "no led suit yet" (leading)*/, &[],
            0, 0, 0, [0,0], 0, 32, 7,
        );
        assert_eq!(res.len(), 2, "both legal moves must be scored");
        for r in &res {
            assert!(r.ev >= 0.0 && r.ev <= 120.0, "EV out of range: {}", r.ev);
        }
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p sueca_solver rollout_tests -- --nocapture`
Expected: FAIL — `cannot find function solve_pimc_rollout in this scope`.

- [ ] **Step 3: Implement `solve_pimc_rollout`**

Add to `src/sueca_solver/src/pimc.rs`. Mirror `solve_pimc`'s signature minus `search_depth`/`diff_mode`/`fixed_worlds`. Reuse `sample_world`. For the playout, construct a `SuecaSimulatorGame` from the determinized `local_hands` and the mid-trick state, then loop `select_card_heuristic` until terminal.

```rust
/// Flat Monte-Carlo PIMC with Elite (HeuristicBot) playouts.
/// EV(card) = mean over `n_worlds` determinized worlds of Team-0&2 final score
/// after playing `card` then letting Elite play all seats to terminal.
/// Provably >= Elite by the rollout policy-improvement theorem; cheaper than
/// deep alpha-beta because each playout is O(remaining cards) bitboard ops.
#[allow(clippy::too_many_arguments)]
pub fn solve_pimc_rollout(
    my_seat: u8,
    my_hand: u64,
    played_cards: u64,
    voids: [u8; 4],
    target_sizes: [u8; 4],
    trump: u8,
    led_suit: u8,
    current_trick_cards: &[u8],
    current_player: u8,
    current_trick_winner: u8,
    current_trick_best_card: u8,
    team_scores: [u8; 2],
    trick_number: u8,
    n_worlds: usize,
    seed: u64,
) -> Vec<PimcResult> {
    use crate::heuristic::select_card_heuristic;
    use crate::simulator::SuecaSimulatorGame;

    // 1. Legal moves (identical logic to solve_pimc).
    let suit_mask = 0x3FFu64 << (led_suit * 10);
    let suited = if led_suit < 4 { my_hand & suit_mask } else { 0 };
    let moves_mask = if suited != 0 { suited } else { my_hand };
    let mut legal_moves = [0u8; 10];
    let mut legal_count = 0;
    let mut temp = moves_mask;
    while temp != 0 {
        legal_moves[legal_count] = temp.trailing_zeros() as u8;
        legal_count += 1;
        temp &= temp - 1;
    }
    if legal_count == 0 { return Vec::new(); }
    if legal_count == 1 {
        return vec![PimcResult { card: legal_moves[0], ev: 0.0, std_error: 0.0 }];
    }

    // 2. Unknown pool + feasibility (reuse helpers).
    let known_mask = my_hand | played_cards;
    let mut unknown_cards = [0u8; 40];
    let mut num_unknowns = 0usize;
    for c in 0..40 {
        if (known_mask & (1u64 << c)) == 0 {
            unknown_cards[num_unknowns] = c; num_unknowns += 1;
        }
    }
    let unknown_slice = &unknown_cards[..num_unknowns];
    if !constraints_feasible(unknown_slice, voids, target_sizes, my_seat) {
        return Vec::new();
    }

    // 3. Per-move Welford over worlds (parallel).
    let stats: Vec<MoveWelford> = (0..n_worlds)
        .into_par_iter()
        .fold(|| [MoveWelford::default(); 40], |mut acc, world_idx| {
            let mut local_hands = [0u64; 4];
            let mut local_seed =
                seed.wrapping_add((world_idx as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));
            let mut ok = false;
            for _ in 0..10 {
                if sample_world(my_seat, my_hand, unknown_slice, voids,
                                target_sizes, &mut local_hands, &mut local_seed) {
                    ok = true; break;
                }
            }
            if !ok { return acc; }

            for i in 0..legal_count {
                let m = legal_moves[i];
                // Build a determinized sim game at the current mid-trick state.
                // NOTE TO IMPLEMENTER: verify the exact SuecaSimulatorGame
                // constructor/setters in src/sueca_solver/src/simulator.rs.
                // It must set: hands=local_hands, trump, current_player,
                // led_suit, current trick cards/winner/best_card, team_scores,
                // trick_number. Mirror how base_game is built in solve_pimc
                // (pimc.rs:393-400) but at the SuecaSimulatorGame layer.
                let mut g = SuecaSimulatorGame::from_determinized(
                    local_hands, trump, current_player, led_suit,
                    current_trick_cards, current_trick_winner,
                    current_trick_best_card, team_scores, trick_number,
                );
                g.apply_card(current_player, m);
                while !g.is_terminal() {
                    let s = g.current_player();
                    let c = select_card_heuristic(&g, s);
                    g.apply_card(s, c);
                }
                let (t02, _t13) = g.team_scores();
                acc[m as usize].update(t02 as f64);
            }
            acc
        })
        .reduce(|| [MoveWelford::default(); 40], |mut a, b| {
            for i in 0..40 { a[i].merge_parallel(&b[i]); }
            a
        });

    // 4. Emit PimcResult per legal move.
    let mut out = Vec::with_capacity(legal_count);
    for i in 0..legal_count {
        let m = legal_moves[i];
        let w = &stats[m as usize];
        let n = w.count.max(1) as f64;
        let var = if w.count > 1 { w.m2 / (w.count as f64 - 1.0) } else { 0.0 };
        out.push(PimcResult { card: m, ev: w.mean, std_error: (var / n).sqrt() });
    }
    out
}
```

> **IMPLEMENTER NOTE:** The `SuecaSimulatorGame::from_determinized`, `apply_card`, `current_player`, `is_terminal`, `team_scores` names above are *intended* APIs. Before writing, open `src/sueca_solver/src/simulator.rs` and `diag_rollout_elite` (evaluator.rs:635-643) to find the real method names (the harness already plays a sim game to terminal with Elite — copy that exact pattern). If no mid-trick determinized constructor exists, add one that mirrors `GameState` setup at pimc.rs:393-400. Adjust the calls to match; keep the algorithm identical.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p sueca_solver rollout_tests -- --nocapture`
Expected: PASS (2 tests).

- [ ] **Step 5: Full solver test sweep (no regressions)**

Run: `cargo test -p sueca_solver`
Expected: all pass (the existing `solve_pimc`/`alpha_beta` tests are untouched).

- [ ] **Step 6: Commit**

```bash
git add src/sueca_solver/src/pimc.rs
git commit -m "feat(pimc): flat Monte-Carlo rollout solver with Elite playouts"
```

---

### Task 2: Add `RolloutPimc` to the diagnostic harness

**Files:**
- Modify: `src/sueca_wann/src/evaluator.rs` — `DiagPolicy` enum (evaluator.rs:627), `diag_pick` (evaluator.rs:645), and `diagnostic_pimc_vs_elite` (evaluator.rs:751)

- [ ] **Step 1: Extend the policy enum and picker**

In the `DiagPolicy` enum, add a variant:

```rust
    enum DiagPolicy {
        Elite,
        Intent(usize),
        MimicOracle,
        RolloutOracle,
        RolloutPimc { worlds: usize }, // NEW
    }
```

In `diag_pick`, handle it by calling the new solver on the *current* (perspective-correct) game state. Build the `solve_pimc_rollout` arguments from the `SuecaSimulatorGame` exactly as the existing `solve_pimc` call site does (evaluator.rs:226 region — copy the argument extraction: my_seat, my_hand, played, voids, target sizes, trump, led_suit, current trick cards/winner/best card, team scores, trick number). Then pick `argmax ev`:

```rust
            DiagPolicy::RolloutPimc { worlds } => {
                let res = sueca_solver::pimc::solve_pimc_rollout(
                    /* ...extract from `game` exactly like the solve_pimc call at evaluator.rs:226... */
                    seat, /*my_hand*/ game.hand(seat), /*...*/ *worlds, 0xC0FFEE,
                );
                res.iter().copied()
                    .max_by(|a, b| a.ev.partial_cmp(&b.ev).unwrap())
                    .map(|r| r.card)
                    .unwrap_or_else(|| select_card_heuristic(game, seat))
            }
```

- [ ] **Step 2: Print RolloutPimc vs Elite in the diagnostic**

In `diagnostic_pimc_vs_elite`, after the existing PIMC-vs-Elite measurement, add a head-to-head where seats {0,2} use `RolloutPimc { worlds: nw }` and seats {1,3} use `Elite`, using the same `diag_play` loop already in the test. Print `Rollout-PIMC vs Elite: XX.X%` and average team points.

- [ ] **Step 3: Run the diagnostic (this is Gate 1 evidence)**

Run: `PIMC_DEALS=200 PIMC_WORLDS=100 cargo test -p sueca_wann --release diagnostic_pimc_vs_elite -- --nocapture --test-threads=1`
Expected output includes a line like `Rollout-PIMC vs Elite: NN.N%`. Record it.

- [ ] **Step 4: Commit**

```bash
git add src/sueca_wann/src/evaluator.rs
git commit -m "test(diag): measure flat-rollout PIMC vs Elite"
```

---

### Task 3: GATE 1 — decide whether the teacher is supra-Elite

**No code.** Read the number from Task 2 Step 3 and the wall-clock.

- [ ] Record `Rollout-PIMC vs Elite` win% and per-decision latency.
- [ ] **Decision:**
  - **≥55% and latency acceptable for dataset gen** (target: a 5k-state dataset in ≲1 h on arise) → proceed to Task 4.
  - **50–55%** → marginal. Try `PIMC_WORLDS=200`; if still <55%, the rollout policy (Elite) is too weak a playout to exceed itself meaningfully → **skip to Task 8** (belief features) — a better teacher is not the lever.
  - **<50%** → the implementation is wrong (rollout cannot be below its own base policy with enough worlds). Debug determinization/seating before proceeding.

> The positional baseline in this harness is Elite-vs-Elite ≈47% / 58.8 pts (Team 0&2 disadvantage, no seat rotation). Judge "supra-Elite" relative to that 47%, not 50%.

---

### Task 4: Wire the rollout teacher into dataset generation (flagged)

**Files:**
- Modify: `src/sueca_wann/src/dataset_gen.rs` (the labeling call that today invokes `solve_pimc`)
- Modify: `src/sueca_wann/src/main.rs` (`generate-dataset` subcommand args)

- [ ] **Step 1: Add `--teacher` CLI flag**

In `main.rs`, add to the `generate-dataset` arg parser an enum-like string flag `--teacher` with values `alphabeta` (default) and `rollout`, plumbed into the dataset-gen entry function as a parameter.

- [ ] **Step 2: Branch the labeler**

In `dataset_gen.rs`, where each decisive state currently calls `solve_pimc(...)` to get per-intent-card EV (best-of-3-by-PIMC-EV labeling), branch on the flag: when `rollout`, call `solve_pimc_rollout(...)` with the same state arguments (drop `search_depth`/`diff_mode`/`fixed_worlds`). Keep the *labeling rule identical* (best of the 3 intent-resolved cards by EV; ties → multi-label; all-3-tied → reject). Only the EV source changes.

- [ ] **Step 3: Smoke test — small dataset compiles and yields states**

Run: `cargo build -p sueca_wann --release && ./target/release/sueca_wann generate-dataset --n-worlds 100 --teacher rollout --target-count 300 --soft-balance-min-ratio 0.0 --output /tmp/teacher_smoke.npz 2>&1 | tail -5`
Expected: completes, writes `/tmp/teacher_smoke.npz`, prints accepted-state count > 100.

- [ ] **Step 4: Verify label sanity (Python)**

Run: `uv run python scripts/analyze_dataset.py /tmp/teacher_smoke.npz` (this script already exists per README "Analyzing Expert Datasets").
Expected: belief features in [0,1]; intent distribution is a sane EFFICIENT/EQUITY mix (not 100% one class).

- [ ] **Step 5: Commit**

```bash
git add src/sueca_wann/src/dataset_gen.rs src/sueca_wann/src/main.rs
git commit -m "feat(dataset): rollout teacher behind --teacher flag"
```

---

### Task 5: Generate the production teacher dataset (on arise)

**No new code.** Run on arise (`ssh arise`, project at `/home/andre/Projects/CAA/project`, build with `cargo build -p sueca_wann --release`).

- [ ] Generate a larger dataset than v5 (the rollout teacher is cheaper, and Stage-1 extends Phase 0 so more data helps):

```bash
./target/release/sueca_wann generate-dataset \
  --n-worlds 150 --teacher rollout --target-count 15000 \
  --soft-balance-min-ratio 0.0 --output expert_states_v6.npz
```

- [ ] scp `expert_states_v6.npz` back to the repo root and `git add` it (datasets are tracked here; see `expert_states_v5.npz`).
- [ ] Commit: `git commit -m "data: rollout-teacher dataset expert_states_v6"`

---

### Task 6: Extend Phase 0 and point training at the new dataset

**Files:**
- Modify: `configs/default.toml`

- [ ] Set `phase0_dataset = "expert_states_v6.npz"`.
- [ ] Increase `phase0_gens` (extend the supervised bootstrap — now worthwhile because labels are supra-Elite, not capped at Elite). Suggested: `phase0_gens = 150` (from 100). Keep `use_class_weighting = false`.
- [ ] Commit: `git commit -m "config: extend phase0, use rollout-teacher dataset"`

---

### Task 7: Full retrain + benchmark + GATE 2

**No new code.** On arise.

- [ ] `cargo build -p sueca_wann --release`
- [ ] `./target/release/sueca_wann train --config configs/default.toml` (creates a new dated run dir under `checkpoints/production/`)
- [ ] Benchmark the champion at n=2000 and n=3000:

```bash
./target/release/sueca_wann benchmark --deals 3000 \
  --genome checkpoints/production/<NEW-DATE-N>/genomes/best_genome_final.json
```

- [ ] Record win% vs Elite (the `[6/6] EliteHeuristicBot vs WANN` block). Update `problems.md` and `README.md` benchmark table.
- [ ] **GATE 2 decision:**
  - **≥56% vs Elite (n≥2000, CI excludes 50%)** → Stage 1 succeeded → proceed to **Stage 2 (Distill)**.
  - **52–56%** → real but modest gain; the teacher helped but features/training cap it → do **Task 8** (belief features) then re-run Task 7 before Stage 2.
  - **≤ current 52.7%** → teacher labels did not transfer; investigate Phase-0 val accuracy and the Phase-0→1 transfer before spending more.

---

### Task 8: (Conditional) Belief inference features — raise the feature ceiling

Do this **only if** Gate 1 or Gate 2 indicates features are the bottleneck. RandomForest probes measured the *current* feature ceiling at LEAD 62.6% / FOLLOW 70.8% — not 90% — so new features are needed to exceed ~58% vs Elite.

**Files:**
- Modify: `src/sueca_solver/src/belief.rs` (add features; bump `INPUT_COUNT` in `src/sueca_solver/src/constants.rs`)
- Modify: `src/sueca_wann/src/dataset.rs` (the loader asserts `INPUT_COUNT`; it will reject old datasets — regenerate)
- Modify: README/CLAUDE belief tables

**Candidate features (all computable from public info only — verify no hand leakage):**
1. `Opp_Min_Trumps_Lower_Bound` — from void observations + played trumps + count constraints, the provable minimum trumps an opponent could still hold / 10.0. (Captures "are opponents out of trump?")
2. `Led_Suit_Likely_Exhausted` — fraction of led-suit cards accounted for (played + in my hand) / 10.0. (Captures "will this suit come back around?")
3. `Partner_Trump_Strength_Estimate` — expected trump count for partner given public constraints / 10.0.
4. `Cards_That_Beat_My_Boss_Remaining` — count of unseen cards that could beat my best card in a suit / 10.0.

**Method (measure before training — your own "diagnose first" rule):**
- [ ] Add features to belief.rs guarded so the existing dataset still loads for A/B, OR regenerate a labeled probe set.
- [ ] Dump (new belief, teacher intent) pairs via the harness; run a RandomForest + per-feature importance in a small Python script (mirror the existing RF probe described in the memory note) to measure each feature's marginal lift on FOLLOW EFFICIENT-vs-EQUITY accuracy.
- [ ] **Keep only features with ≥+1.5pp RF lift.** Drop the rest (YAGNI; each feature is a node the WANN must wire and a thing to interpret later).
- [ ] Regenerate dataset (Task 5) with the new belief, retrain (Task 7).

> **Interpretability note:** these features move inference complexity *out of the WANN into auditable belief code* — they help Stage 2, not hurt it. But each one is itself a heuristic that must be documented in README/CLAUDE belief tables.

---

## STAGE 2 — Distillation (strong reference organism → interpretable artifact)

**Entry condition:** Stage 1 produced a strong "reference organism" (the teacher: either the strong WANN champion, or the rollout-PIMC itself). Stage 2 compresses its *policy* into something human-readable, measuring the strength↔interpretability trade-off explicitly. These are **candidate experiments**, gated on Stage-1 results; run them cheapest-first and keep what wins on the trade-off curve. (Per the writing-plans scope rule, each becomes its own detailed sub-plan once Stage 1 fixes the teacher.)

### The distillation toolbox (recommendations, ranked)

**2a. Knowledge distillation into a small WANN student (RECOMMENDED — keeps the WANN identity).**
- Teacher = the strong network (or rollout-PIMC). Dump intent distributions for a large set of belief states sampled from real games (dense, noise-free, supra-Elite labels — far better than PIMC's sparse expensive ones).
- Evolve a *student* WANN with fitness = **agreement with teacher intent** (fidelity), under **hard Pareto/simplicity pressure** (heavily weight the simplicity objective the codebase already supports via MAP-Elites complexity axis + Pareto ranking).
- Output: an evolved interpretable WANN + a measured **strength-vs-size Pareto curve** ("X% of champion strength at Y connections"). This curve is the paper-worthy result: it *proves* the architecture compresses.
- New module `src/sueca_wann/src/distill.rs`; reuses Phase-0 supervised machinery (fitness = classification agreement instead of dataset labels).

**2b. Decision-tree / rule-list surrogate (RECOMMENDED as the interpretability *upper bound* + deployable ruleset).**
- Fit a shallow `DecisionTreeClassifier` (or RIPPER/CN2 rule list) in `scripts/extract_rules_tree.py` on (belief → teacher intent argmax) pairs from real games.
- Sweep `max_depth ∈ {3,4,5,6}`; for each, measure **fidelity** (agreement with teacher) and **strength** (plug the tree in as the intent policy via a tiny Rust/!PyO3-free export, or evaluate fidelity-only and accept the tree as documentation).
- This abandons the WANN representation for the *artifact* but yields the most readable rules and a hard ceiling on "how simple can the policy be." Frame: "the WANN proves features+intents suffice; the tree is the human-readable distillate."

**2c. Post-hoc network minification of the champion (cheap, do first as a quick win).**
- Constant-fold dead gates already visible in `compiled_rules_lead.txt` (`hidden_43 = 0.0`, `hidden_79 = 0.0`) — extend `compile_rules.rs` to eliminate constant/unreachable nodes and report complexity metrics (gate count, max DAG depth, literal count).
- Greedy connection pruning guided by the diagnostic harness: remove the connection whose deletion least drops win% vs Elite; repeat until win% drops below a tolerance. Cheap, no retrain, immediately shrinks the existing champion.

**2d. Boolean/logic minimization (highest interpretability, more effort).**
- The network restricted to the *realized belief distribution* is a threshold/logic circuit. Binarize inputs at their THRESHOLD points, enumerate each intent output over sampled states, and minimize with Espresso/Quine–McCluskey or a BDD to a small sum-of-products. Yields canonical minimal rules. Risk: float inputs and the on-data distribution must be handled carefully (it's an *approximation on the data manifold*, not the full input cube — which is fine and actually tighter).

### Suggested Stage-2 sequence
1. **2c first** (cheap, shrinks today's champion, gives a baseline interpretability number).
2. **2b** (fast Python, establishes the fidelity/strength/interpretability frontier and a readable ruleset).
3. **2a** (the headline result: a compressed *WANN* on the Pareto curve, distilled from dense teacher labels).
4. **2d** only if 2a/2b leave the artifact still too large to read.

### Stage-2 metrics (report all three for every candidate)
- **Strength:** win% vs Elite (n≥2000, seat-rotated benchmark).
- **Fidelity:** % intent agreement with the teacher on a held-out belief set.
- **Interpretability:** enabled connections, active hidden gates, max DAG depth, total literals in compiled rules. (Add these to `compile_rules.rs` output in 2c.)

---

## Self-Review (against the spec / conversation)

- **Strength lever covered?** Yes — Stage 1 Tasks 1–7 (rollout teacher → relabel → extend Phase 0 → retrain), gated.
- **Cheap-PIMC insight honored?** Yes — flat rollout, no deep alpha-beta, no `alpha_beta` edit (avoids the prior revert).
- **Fog-of-war / human-info constraint?** Yes — Task 8 features are public-info-only with an explicit no-leak check; "WANN can't learn inference, belief.rs must" is stated in the framing.
- **"Can a WANN learn opponent extrapolation?" answered?** Yes — no (feedforward, no memory/search); inference belongs in belief.rs as auditable features.
- **Interpretability / versioning?** Yes — the strength↔interpretability conflict is resolved by the two-stage split; Stage 2 gives four ranked distillation methods + three metrics + a measured Pareto curve.
- **Phase-1 framing disagreement surfaced?** Yes — in "Disagreements."
- **No placeholders in Stage 1?** Stage 1 steps carry real code, real paths, real commands. The one explicit `IMPLEMENTER NOTE` flags a genuine API-name verification (simulator constructor) rather than inventing a signature — this is honest, not a placeholder. Stage 2 is intentionally candidate-level (gated on Stage-1 results) per the writing-plans multi-subsystem rule; each method becomes its own sub-plan when reached.

---

## First action for the executing agent
Start at **Stage 1, Task 1, Step 1**. Do not touch `alpha_beta` or `solve_pimc`. Before writing `solve_pimc_rollout`'s playout, open `src/sueca_solver/src/simulator.rs` and `diag_rollout_elite` (evaluator.rs:635) to copy the real `SuecaSimulatorGame` play-to-terminal pattern. Everything downstream is gated on Task 3 (Gate 1) — do not regenerate datasets or retrain before that number says the teacher is supra-Elite.
