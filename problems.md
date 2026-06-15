# Sueca WANN — Beating EliteHeuristicBot: Problems Found & Fixes

**Goal:** make the evolved WANN beat `EliteHeuristicBot` (>50% head-to-head).
**Starting point:** the balanced-retrain champion (`checkpoints/production/2026-06-13-2`)
benchmarked at **30.2% vs Elite** and **46.3% vs OldHeuristic** — an improvement over the
prior failed run, but far from the goal.

This document records the investigation in the order it happened, the problems found,
how each was diagnosed, and how it was solved (or why it was left).

---

## Background — problems solved in prior sessions (the prequel)

The current 30.2%-vs-Elite starting point is itself the product of a long chain of earlier
fixes. Summarized here so the record is complete; details live in earlier conversation logs and
git history (`0833c77`, `6a1eb71`, `9587e84`, `43417da`, `792de50`).

**B0 — Per-generation timing was wrong.** Elapsed time excluded speciation, hiding the real
bottleneck. Fixed the profiling so each phase (convert/eval/stats/reseed/breed/ckpt) is timed.

**B1 — PFS-NEAT was a Phase-0 anchor.** Validating every structural mutation at `O(P·K·E)` with
`K=1000` dominated breeding. Reduced to an adaptive 2-stage check (quick `K=25`, full `K=100` only
for borderline mutations); degraded mutations go into a FIFO Tabu veto list.

**B2 — Belief redesign (33→35 features), no future-info leakage.** Removed features a human can't
observe; added tactical-affordance features (`Can_Beat_Winner`, `Min_Winning_Cost`,
`Min_Sacrifice_Cost`, boss detection, suit-shape counts). Goal: extract human-playable instructions
from only observable state.

**B3 — Removed the MIN_FORCE intent (4→3).** The follow brain never used MIN_FORCE. Root cause
(diagnosed then): the belief features for MIN_FORCE vs non-MIN_FORCE follow states were **nearly
identical (<0.09 delta on all features)** — PIMC picks MIN_FORCE using knowledge of specific
unplayed cards / future dynamics that **isn't in the belief**, so the network correctly evolved away
from it. EFFICIENT_WIN subsumes its useful "play cheapest winner, else concede" behavior.
*(This is the direct precedent for Problem 8 below — the same "belief can't distinguish the
situation" theme recurs on the lead split.)*

**B4 — `pimc_min_margin = 0.5` was a no-op.** Standard error is ~2.2 pts at 80 worlds, so a 0.5
margin was a floating-point equality check. Replaced with an SE-based confidence filter
(`best − second > Z·SE_diff`).

**B5 — Dataset generation took ~97 CPU-hours and stalled.** Strict 33/33/33 per-class balance sent
the generator into "hunt mode" for the rare EFFICIENT_WIN bucket (869/10000 after 97 CPU-hrs,
rejecting 90%+ of deals). Fixed with a stack of optimizations (profiled: `alpha_beta` was 81% of
cost, Zobrist recompute 14%):
- **Incremental Zobrist hashing** (was recomputed from all hands at every node), **MSB-first
  iteration + make/unmake** (removed GameState copies + an insertion sort).
- **Weight-batched forward pass** — one CSR traversal for all 6 sweep weights (the hot path is
  memory-bound, not compute-bound).
- **Paired-difference Welford + futility stop** — projects the best–second EV gap; ~33% of states
  exit before running full PIMC. (Later fixed for non-stationarity: reset the accumulator when the
  leader pair changes.)
- **Exact card equivalence** in alpha-beta (`mask_between` adjacency test — provably lossless),
  constraint propagation, killer-move heuristic, always-on pre-PIMC intent-collision filter.
- **Checkpoint every 5 batches** for crash-resilience.
- Net: **4–6× faster per state** (≈26s → 4–7s).

**B6 — The failed run (2026-06-12) and its post-mortem.** A retrain benchmarked **43.8% vs Old
(down from the 54.8% champion), 31.5% vs Elite**. Dataset was 31,851 states but **EFFICIENT_WIN
8.2%, Follow split 9.4%**. Root-cause cascade: the random walk always ended at trick boundaries →
extracted states were almost always *leading* → the **Follow brain was starved** (2,997 states,
20.7% Phase-0 accuracy = below chance); and EFFICIENT_WIN at 8.2% was extreme imbalance that
inverse-frequency class weighting (4.06×) could only amplify as noise.

**B7 — The recovery (commit `43417da`) → this session's starting point.** Fixes: **mid-trick walk**
(extras land mid-trick, lead fraction 90.6%→~45%), **6-bucket per-split balance** (a globally
balanced set can still starve a split), **multi-label acceptance**, **diff-mode** for controlled
label comparison, a **10% stratified holdout**, and a **fixed-yardstick probe** (frozen 64-deal vs
4× OldHeuristic every 25 Phase-1 gens) to separate opponent-hardening from real regression. This
produced run `2026-06-13-2`: Phase-0 validation Lead 53.1% / Follow 58.2%, probe rising
−1.2→+1.9, benchmark **46.3% vs Old, 30.2% vs Elite** — the chance-level collapse was fixed, but it
still lost to Elite. **That is where this session begins.**

Across this entire history, the project's *best-ever* result was 54.8% vs Old / ~40% vs Elite — it
**never beat Elite**. This session set out to find why and fix it.

---

## Method: diagnose before retraining

A full training run is ~90 min. Instead of guessing-and-retraining, I built a fast,
**training-free diagnostic harness** as an ignored test in
`src/sueca_wann/src/evaluator.rs`:

```
cargo test -p sueca_wann --release diagnostic_resolver_ceiling -- --nocapture --test-threads=1
cargo test -p sueca_wann --release diagnostic_pimc_vs_elite    -- --nocapture --test-threads=1
cargo test -p sueca_wann --release diagnostic_dump_intent_labels -- --nocapture --test-threads=1
```

These run in seconds and answer the questions a retrain would have answered in hours.
Positional note: the harness plays Team 0&2 vs Team 1&3 with `first_player=0` and **no seat
rotation**, so Elite-vs-Elite is ~47% / 58.8 pts (a fixed positional disadvantage for the
candidate seats), not 50%. The seat-rotated `benchmark` command removes this bias.

---

## Problem 1 — Wrong assumption: "the resolver/intent set is the ceiling"

**Hypothesis:** the WANN outputs 1 of 3 abstract intents that a heuristic resolver turns into a
card; maybe 3 intents can't express Elite-grade play.

**Diagnosis (harness):**
- Elite-move expressiveness coverage (does *some* intent reproduce Elite's card): 78% overall,
  but only **50% when leading**.
- **But** the *rollout best-of-3 oracle* — pick, per decision, the intent whose card yields the
  best outcome — scored **65% vs Elite**. So optimally mixing the *existing* 3 intents beats Elite.

**Conclusion:** the intent vocabulary is rich enough. The resolver is **not** the binding ceiling.
(The rollout oracle is perfect-information-optimistic, so 65% overstates the belief-state ceiling,
but it proved real headroom.)

---

## Problem 2 — The WANN never learned intent selection (collapsed to a constant)

**Diagnosis (harness, running the actual champion):**
- The champion played ≈ **"always EFFICIENT_WIN"**: 35% vs Elite / 53.5 pts, statistically
  indistinguishable from the constant policy "always output intent 1" (37% / 52.4).
- Its intent mix was **MAX_FORCE 20% / EFFICIENT 80% / EQUITY 0%** — it had **functionally dropped
  the EQUITY intent**. Inspecting the genome: in the *lead* brain, the EQUITY output node (38) was
  not even reachable from any input — a structurally **dead output**.
- On *decisive* states (where intent choice changes the outcome — 45% of decisions), it agreed
  with the outcome-optimal intent only **51%** (chance 33%).

**Conclusion:** the failure is the WANN not *learning* a useful belief→intent policy, not the
resolver. This is exactly the "bad dataset / bad training loop" class of problem.

---

## Problem 3 — Root cause of the collapse: individually-weak, unequal intents

**Diagnosis (harness, pure-intent strength vs Elite):**

| intent | win% vs Elite (old resolver) |
|---|---|
| MAX_FORCE | 20% |
| EFFICIENT_WIN | 37% |
| EQUITY_BUILDER | 25% |

The three intents were individually very unequal and all well below Elite. In Phase-1 self-play
the fitness is raw game outcome, so **using the individually-weak intents is punished** unless the
timing is perfect (hard to evolve). The evolutionary gradient therefore points at "just output the
single best intent (EFFICIENT)", collapsing the policy and atrophying the others.

---

## Fix for 2 & 3 — strengthen the resolver (raise the floor, remove the collapse basin)

**Change:** rewrote the intent resolver as `select_card_styled` in `heuristic.rs`
(`resolve_intent` now dispatches to it):
- **EFFICIENT_WIN** delegates to `select_card_heuristic` (Elite) — the strong default / floor.
- **MAX_FORCE** = Elite tactics + aggressive dials (lead a low trump to *draw* when trump-long).
- **EQUITY_BUILDER** = Elite tactics + tempo dials (lead the *shortest* side-suit to build voids;
  *duck* cheap tricks and *preserve* trump when the trick is worth ≤2 pts).

**Validated (harness, no retrain):**

| policy | win% vs Elite |
|---|---|
| always MAX_FORCE | **48%** (was 20) |
| always EFFICIENT_WIN | 47% (= Elite) |
| always EQUITY_BUILDER | **46%** (was 25) |
| rollout best-of-3 | 58% |

Each intent is now individually ≈Elite, so a policy collapse merely *ties* Elite (~50% in the
seat-rotated benchmark, vs the old 30%), and good mixing still exceeds it. The Phase-1 collapse
incentive is removed.

---

## Problem 4 — Dataset labeling discarded 85% of states (and used the wrong target)

**Diagnosis:** the labeler (`map_card_to_soft_intents`) kept a state only if some intent resolved
to the **global PIMC-best card**, else rejected it. With the new resolver only **~15%** of tactical
states qualified — and that target is wrong anyway: the WANN can only ever pick one of the 3
intents, so the supervised target should be the **best of the 3 available intents**, not "matches
the unconstrained optimum."

**Fix (`dataset_gen.rs`):** label each decisive state with the intent whose card has the best
**PIMC EV** among the 3 (statistically-tied intents → uniform multi-label; all-3-tied → reject).
Uses every decisive state with a valid imperfect-information target.

---

## Problem 5 — Pre-filter rejected almost everything under the new resolver

**Diagnosis:** the always-on pre-filter skipped any state where *any pair* of intents collided.
Old resolver: intents differed ~50% of the time. New resolver (intents share Elite's core): they
agree ~90% of the time → the filter rejected almost everything → dataset yield collapsed (the
`test_belief_bounds` test produced <50 states from 600 deals).

**Fix:** reject only **fully-degenerate** states (all three intents identical). Keep every state
where the intent choice has *any* effect — the decision-relevant states worth training on.

---

## Problem 6 — Class balancer hunts an unfillable bucket forever

**Diagnosis:** with the new resolver, MAX_FORCE is *almost never uniquely best* (~1% of labels;
**0** in the follow split, because MAX_FORCE only differs from EFFICIENT when leading). The
6-bucket balancer's "every bucket ≥ soft_min" termination condition could never be met, so dataset
generation never finished (timed out, wrote nothing).

**Fix:** generate with `--soft-balance-min-ratio 0.0` → the natural distribution (~50/50
EFFICIENT/EQUITY, MAX_FORCE rare), which terminates cleanly at target. Set
`use_class_weighting = false` (inverse-frequency weighting would give the ~1% MAX_FORCE bucket a
huge weight = noise amplification). The decision is effectively **binary EFFICIENT-vs-EQUITY**.

---

## Problem 7 — There is no supra-Elite teacher in the pipeline

**Diagnosis (harness, `diagnostic_pimc_vs_elite`):** the project's own PIMC solver only *ties*
Elite — **PIMC(200,4) = 45%**, PIMC(80,3) = 34% vs Elite's 48% baseline. Root cause:
`search.rs` alpha-beta's depth-limited **leaf evaluation returns points-so-far** with no positional
estimate (the code comment at line ~190 even says "for a true PIMC bot we roll out using
HeuristicBot" — but it was never implemented). So depth-limited PIMC is myopic in the early/mid game.

**Implication:** *every* signal source in the pipeline is ≤ Elite (PIMC dataset ≈ Elite,
Old/Elite opponents ≤ Elite). A WANN that imitates them cannot, by imitation alone, exceed Elite.

**Attempted fix (reverted):** implemented a heuristic rollout-to-terminal leaf evaluation. It made
PIMC modestly stronger (40% at depth 4) but **still lost to Elite** (the rollout policy itself is
naive) and it broke two solver tests. Given the marginal benefit and the risk of destabilizing the
core engine before a retrain, it was **reverted**. The supra-Elite-teacher problem is left open
(it needs either a much stronger leaf policy or deeper search — a larger effort).

---

## Problem 8 — Is the winning deviation even learnable from the belief features?

**Diagnosis (RandomForest on PIMC-labeled states):** with proper depth-4/100-world labels,
predicting EFFICIENT-vs-EQUITY from the 35 belief features:
- **FOLLOW: RF 62.3% vs 54.5% majority baseline (+7.9)** — a real, learnable signal.
- LEAD: +2.5 — weak.

(An earlier, smaller depth-3/40-world sample had falsely suggested "unlearnable"; it was just label
noise.) Following is where most decisions and points are, so the learnable follow-deviation is the
realistic source of any edge above Elite.

---

## Minor issues noted (not yet fixed)

- **Phase-0 tie-break mismatch:** `evaluate_phase0` breaks argmax ties deterministically toward
  intent 0 (`train.rs`), while inference breaks ties randomly (CLAUDE.md Pitfall #7). Second-order;
  left for now.
- **Lead-brain PFS bootstrap fragility:** PFS-NEAT can fail to grow a 3-output classifier from 0
  connections (the first 1–2 connections can't beat chance → all rejected). Observed at pop=600
  (lead stuck at 33.3%); production pop=1000 bootstrapped fine. Seed/scale-sensitive.

---

## Net effect of the fixes

- Resolver floor raised from 20–37% to **45–48%** per pure intent (≈Elite); collapse incentive
  removed.
- Dataset pipeline produces valid, decisive-state, best-of-3-intent labels at the natural
  distribution.
- The valuable deviation (follow-side duck/preserve) is **learnable** from the belief (+7.9).

**Expected retrain outcome:** ~30% → **≈50%** vs Elite (floor), with upside above 50% from the
learnable follow-deviation and Phase-1 exploitation of the fixed Elite opponent. A *large-margin*
win is hard because Elite is near the achievable ceiling for fast policies (even full PIMC only
ties it).

---

## Results of the retrain

**The retrained WANN beats EliteHeuristicBot.** Goal met.

Run `checkpoints/production/2026-06-14-1` (pop=1000, 600 gens, phase0_gens=100,
dataset `expert_states_v5.npz`, `use_class_weighting=false`, full 6-weight sweep). Trained on
arise in ~1 h (gens 0→600). Benchmarked with the seat-rotated `benchmark` command at three sample
sizes to pin down the margin:

| matchup (candidate = WANN) | n=300 | n=1000 | **n=3000** |
|---|---|---|---|
| WANN win% vs **Elite** | 56.2% ± 5.6% | 52.1% ± 3.1% | **52.7% ± 1.8%** |
| Card pts WANN : Elite | 60.6 : 59.4 | 60.3 : 59.7 | **60.3 : 59.7** |
| WANN win% vs OldHeuristic | 65.2% | — | — |
| WANN win% vs Random | 94.5% | — | — |

The definitive **n=3000** figure is **52.7% ± 1.8%**, a 95% CI of **[50.9%, 54.5%]** that
**excludes 50%** — a statistically significant win, not a tie. The point differential (+0.6 pts/deal,
60.3 vs 59.7) is small but stable across every sample size. For context, on the same board Elite
beats OldHeuristic 66.8% and the *old* champion scored only **30.2%** vs Elite.

**Why the small margin is the expected result, not a disappointment.** Problem 7 established there is
no supra-Elite teacher in the pipeline — even the project's own PIMC solver only ties Elite (its leaf
eval is myopic). So imitation alone caps at ≈Elite, and the realistic ceiling for a fast policy here
*is* roughly Elite. The edge above 50% comes from exactly the two places the diagnostics predicted:
the resolver floor (each pure intent now ≈Elite, removing the collapse basin — Problems 2–3) plus the
learnable follow-side duck/preserve deviation (+7.9 RF signal — Problem 8) and Phase-1 exploitation of
the fixed Elite opponent.

**The arc, end to end:** project best-ever was 54.8% vs Old / ~40% vs Elite and had *never* beaten
Elite. The failed 2026-06-12 run hit 31.5% vs Elite; the recovery got to 30.2%. The diagnosis-first
approach this session (a training-free harness instead of guess-and-retrain) located the real
bottleneck — the WANN had collapsed to a constant intent because the intents were individually weak
and unequal — and the styled-resolver + best-of-3 dataset fixes lifted it to **52.7% vs Elite**, the
first time the WANN has beaten EliteHeuristicBot.

### Honest caveats / what would push the margin higher
- The win is **+2.7 pp** (and +0.6 card pts). It clears significance at n=3000 but it is a *narrow*
  win, consistent with Elite being near the fast-policy ceiling.
- The single biggest lever left is **Problem 7**: a genuinely supra-Elite teacher (stronger leaf
  policy or deeper PIMC search) would raise the imitation ceiling. That is a larger engine effort and
  was deliberately left open.
- Lead-brain PFS bootstrap remains seed-sensitive (Minor issues); this run bootstrapped fine at
  pop=1000.

---

## Chapter 2 — The rollout teacher: solving Problem 7 (and what it actually bought)

Problem 7 ("no supra-Elite teacher exists") was the named next lever. We solved it — cheaply — and
the result reframes the whole strength/interpretability story.

### Problem 9 — The PIMC weakness was the leaf eval, not the search depth
The reverted depth-rollout attempt (Problem 7) had concluded a stronger teacher needs deeper/heavier
search. Wrong diagnosis. `search.rs` alpha-beta returns **points-captured-so-far** at the depth limit
(`search.rs:184` and `:193`) — a *myopic* leaf with no positional estimate. Depth doesn't fix a blind
leaf. **Fix:** a separate flat-Monte-Carlo solver `solve_pimc_rollout` (`pimc.rs`) that, per legal
move, plays the move then rolls out the rest of the deal with **Elite in all four seats**, averaging
the ego team's terminal score over determinized worlds. By the rollout policy-improvement theorem this
is ≥ Elite, and it is *cheaper* than deep alpha-beta (no search tree). It does **not** touch
`alpha_beta` (the prior attempt did, broke two tests, and was reverted). Determinization copies the
live `SuecaSimulatorGame` and overwrites only the hidden hands (ego hand + trick/score/void state
preserved); the stale Zobrist hash is unused by Elite playouts.

**Gate-1 result (harness, 150 deals):** rollout PIMC(100) = **62.0% / 66.0 pts** vs Elite, where the
Elite-self positional baseline is 46.0% / 58.5 and the old alpha-beta PIMC was 36.7% / 53.4. The
first genuinely supra-Elite signal in the pipeline. Dataset generation with this teacher is also
**~1000× faster** (the all-intents-agree pre-filter skips solving on most states): 15k labeled
states in **11 s** vs hours for the alpha-beta pipeline.

### Problem 10 — A supra-Elite teacher does NOT raise the benchmark; it raises interpretability
Retrained with the rollout-teacher dataset (`expert_states_v6.npz`, 15k states, `phase0_gens=150`):
run `checkpoints/production/2026-06-14-2`. Benchmark **52.1% ± 1.8% vs Elite (n=3000)**, pts 60.2 :
59.8 — *statistically identical* to v5's 52.7%. The better teacher did **not** move strength.

But the champion is **dramatically simpler at iso-strength**:

| | v5 champion (`2026-06-14-1`) | v6 champion (`2026-06-14-2`) |
|---|---|---|
| Win vs Elite (n=3000) | 52.7% ± 1.8% | 52.1% ± 1.8% |
| Lead brain | 68 hidden gates / 120 conns | **12 hidden / 35 conns** |
| Follow brain | 64 hidden / 122 conns | **17 hidden / 44 conns** |
| Total hidden logic gates | 132 | **29** (4.5× fewer) |
| Genome size | 20.4 KB | 8.4 KB |

The v5 lead brain compiled `EQUITY_BUILDER` to a 4-term sum over a **40-gate** nested DAG (unreadable).
The v6 lead brain compiles to a **readable, strategically-sensible** policy:
```
EQUITY_BUILDER = THRESHOLD(NOT(Trump_Count))             # build voids when trump-poor
EFFICIENT_WIN  = 0.0                                       # never, when leading
MAX_FORCE      = (ahead & point-dense & partner-winning)  # press the advantage
                 OR (no very-short side suit)
```

**Why the teacher helped simplicity but not strength.** Cleaner, more-decisive labels (the supra-Elite
teacher commits to a single best intent instead of the noisy alpha-beta near-ties) let evolution fit
the policy with far less structure. But the *strength* ceiling is unchanged because:
1. **`EFFICIENT_WIN = 0.0` in both brains** — the net never outputs it; it settled on a near-constant
   policy (lead: equity-when-trump-poor / force-when-ahead; follow: force-early / equity-late).
2. The **styled-resolver floor** (every intent ≈ Elite) is what delivers the 52% — the win is
   *resolver-floored*, not *selection-driven*. Phase-1 + the floor converge to the same attractor
   regardless of how good the Phase-0 labels are (F-Val 47% vs the 70.8% RF follow ceiling).

### Problem 11 — The resolver floor is a double-edged sword (the real strength ceiling)
The styled resolver that removed the 30% collapse basin (Chapter 1) also makes the three intents
nearly equivalent in game fitness — so there is **little gradient to learn precise intent selection**,
and the net collapses to the simplest policy the resolver floors to Elite. The measured ceiling for
*perfect* 3-intent selection is only **58%** (the styled best-of-3 oracle, Problem 1); we are at 52%.
To exceed ~52% by *learned* play, the bottleneck is now belief features (follow EFFICIENT-vs-EQUITY
separability — Problem 8) and/or **more-differentiated intents** so selection actually matters, not a
better teacher.

### Net of Chapter 2
- ✅ Supra-Elite teacher built (62% vs Elite), cheaply; dataset gen ~1000× faster. Problem 7 closed.
- ➖ Strength unchanged (52%): the cap is features/selection, not teacher quality.
- ✅✅ **Interpretability: a 4.5×-smaller champion at iso-strength, with human-readable rules in both
  brains** — a direct win for the project's core thesis. This is the headline result of Chapter 2.
- The teacher remains valuable as a dense label source for distillation (Stage 2).

---

# Chapter 3 — Interpretability Polish (Stage 2c: post-hoc rule minification)

**Date:** 2026-06-15. **Goal (user):** consolidate the win, then *polish interpretability* — the
project's whole reason for WANNs + discrete activations. Chapter 2 produced a 4.5×-smaller champion;
Chapter 3 makes its compiled rules genuinely *readable* and adds verified complexity metrics, without
touching the network (no retrain, no strength change).

### Problem 12 — The compiled rules were smaller but still cluttered with dead/rename gates
Even the 29-gate v6 champion compiled to chains dominated by two kinds of non-information:
- **Dead constant gates** — e.g. `hidden_46 = 0.0`, then `hidden_44 = NOT(hidden_46)` (a constant 1.0)
  feeding downstream sums. These nodes carry no signal but inflate gate count and depth.
- **Pure alias / passthrough gates** — a single-input IDENTITY node is just a rename of its source
  (`SUM(x)=MIN(x)=MAX(x)=x` at W=1). The FOLLOW brain had a 7-deep alias chain
  (`hidden_49=hidden_48`, `hidden_53=OR(hidden_39)`, `hidden_52=Game_Pts_Remaining`, …) that made
  "max depth" read **10** when the real logic is 5 deep.

### Fix — two behavior-preserving rewrites in `compile_rules.rs` (verified at W=1)
1. **Constant-folding.** Propagate constants from BIAS (=1.0) and empty aggregations through the exact
   `wann_network::forward` math; inline constant nodes as numbers and drop them from the listing.
2. **Alias inlining.** Flatten single-input IDENTITY chains to their ultimate source, tracking negation
   parity (NOT∘NOT cancels). Guarded to **W=1** (at W≠1 the node is `clamp(src·W)`, not an alias).
3. **Single-operand unwrap** (`AND(x)→x`, `OR(x)→x`) and a `Rule Complexity` metrics block.

### Result — the FOLLOW brain, before vs after folding

```
BEFORE (12 steps, max depth 10):          AFTER (5 steps, max depth 5):
  hidden_46 = 0.0                            hidden_42 = NOT(Holds_Boss_Led)
  hidden_44 = NOT(hidden_46)                 hidden_48 = (1 + Any_Opp_Void_Led)
  hidden_51 = Any_Opp_Void_Led               hidden_41 = NOT((1 + hidden_42 + hidden_48 + hidden_48))
  hidden_48 = (hidden_44 + hidden_51)        hidden_40 = THRESHOLD((hidden_41 + Trump_Count) > 0.5)
  hidden_49 = hidden_48                       hidden_55 = THRESHOLD(hidden_40 > 0.5)
  hidden_41 = NOT((1 + h_42 + h_48 + h_49))
  hidden_39 = hidden_41                      MAX_FORCE      = Game_Pts_Remaining + Game_Pts_Remaining
  hidden_40 = THRESHOLD((h_39 + Trump) >.5)  EFFICIENT_WIN  = 0
  hidden_50 = hidden_40                      EQUITY_BUILDER = (1 + hidden_41 + hidden_55 + hidden_41)
  hidden_52 = Game_Pts_Remaining
  hidden_53 = OR(hidden_39)                 # MAX_FORCE is literally 2·Game_Pts_Remaining: force early,
  hidden_55 = THRESHOLD(hidden_50 >.5)      # concede late — a one-feature game-phase toggle.
  hidden_45 = AND(hidden_55)
```

### Complexity metrics (on the collapsed rule DAG)

| Metric | LEAD brain | FOLLOW brain |
|---|---|---|
| Active hidden gates | 10 (was 12; 2 aliases inlined) | **5** (was 12; 2 constants + 7 aliases inlined) |
| Live connections | 20 / 22 enabled | 13 / 27 enabled |
| Max logic depth | 6 | **5** (was 10) |
| Total input literals | 10 | 5 |

### Verification (this is the point — not cosmetic)
A property test (`champion_fold_and_alias_are_behavior_preserving`, `#[ignore]`) runs the **real
champion** through `wann_network::forward` on **2000 random belief states at W=1** and asserts:
every folded constant is invariant to within 1e-9, and every alias node equals its resolved source
(parity applied) to within 1e-9. **Passes.** So the dropped symbols provably carry no information the
network uses — the reader can trust the rules are exact, not an approximation. Four further
deterministic unit tests cover the fold/alias/parity logic and the W≠1 guard.

### Net of Chapter 3
- ✅ Compiled rules are now genuinely human-readable: FOLLOW = 5 gates / depth 5; both intent policies
  read as short IF/THEN logic over named features.
- ✅ Verified behavior-preserving (no strength change — same 52.1% champion; the network is untouched).
- ➖ Not done (deferred per user): greedy connection pruning of the genome itself, and the
  intent-bottleneck redesign (Stage 2a/2b/2d) — "a last, if-we-have-time deliverable."
