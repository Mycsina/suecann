# Resolver / Action-Space Overhaul — Design Spec

**Date:** 2026-06-19
**Status:** Approved for planning (brainstorming complete; next step = implementation plan)
**Related:** `ideas.md` §3 (Tier S), `reference.md` → "Imperfect-Information Search & EV-Model Quality"

---

## 1. Context & Problem

The pipeline is `belief(35) → WANN → intent(3) → resolver → card`. The current resolver
(`select_card_styled`, `heuristic.rs`) is `select_card_heuristic` (the full Elite policy) plus a
few narrow conditional dials:

- `EFFICIENT_WIN` is `return select_card_heuristic(...)` verbatim.
- `MAX_FORCE` diverges only when leading with ≥5 trumps and no cashable master.
- `EQUITY_BUILDER` diverges only on lead-suit choice and on ≤2-point ducks/cuts.

So all three intents are ~95% identical code and the WANN chooses between near-duplicates.
Consequence (measured): every pure intent benchmarks ≈Elite individually (48/47/46%), a
collapsed policy merely *ties* Elite, and the realisable headroom above Elite is **≈8 points**.
`EQUITY_BUILDER` is nearly unused in the follow split; `MAX_FORCE` is uniquely best ~0–1%.

**The bottleneck is the action vocabulary, not the network.** The champion v6 already beats
Elite (52.1% ± 1.8%, n=3000); to go meaningfully further we must widen what the network is
allowed to *express*.

## 2. Goals, Non-Goals, Scope

Decisions fixed during brainstorming (2026-06-19):

- **Substrate: negotiable.** Keep the network-picks-intent paradigm and most infra, but allow
  targeted relaxations where they pay: more / continuous outputs, SIGMOID activations, learned
  weights, richer belief features. *Full direct card-scoring (network as the whole policy) is
  out of scope.*
- **Success metric: maximize raw strength, no fixed target ("frontier mode").** Kill/go gates
  are **relative** (does design B's ceiling beat design A's?), not absolute.
- **Teacher / trick-end explanation paradigm: out of scope.** Optimize the action vocabulary
  purely for strength; opaque/continuous actions are fair game. Teaching is a later project.
- **Interpretability: relaxed**, not required.
- **Search-backed resolver (old "Stage C"): dropped.**

Non-goals: equilibrium/Nash solving; replacing neuroevolution with gradient RL; human-data
opponent models; any frontend/teaching work.

## 3. Literature Grounding (condensed)

Full notes in `reference.md`. Load-bearing conclusions:

1. **Sueca is a favorable PIMC regime** (Long et al. 2010): public plays + fast void revelation
   ⇒ high disambiguation and leaf correlation ⇒ the rollout teacher is trustworthy, with error
   concentrated in the **early tricks** and near-exact late game.
2. **The teacher's real weakness is sampling/inference, not strategy fusion.** Strategy-fusion
   fixes (EPIMC 2024, αµ) give ~no benefit in public-observation trick-takers — **deprioritized**.
3. **The fitting teacher upgrade is policy-based inference** (Rebstock et al. 2019): posterior
   determinization weighted by an opponent/partner policy model. Use **Elite as `π`** (no human
   data). **Caveat:** better inference ≠ better play (validate empirically).
4. **Learned EV for trick-takers is validated prior art** (Buro 2009, Solinas 2019) — we are not
   freelancing.

## 4. Measurement Backbone (the spine — shared by both stages)

Everything is gated on cheap, honest measurement *before* expensive training.

### 4.1 Teacher trust & calibration
- Build a **per-phase trust map**: on a held set of states, compare rollout-PIMC's *primitive
  ranking* (not absolute values) against (a) **exact endgame minimax** (≤16 cards, already in
  `pimc.rs`) and (b) **deep alpha-beta PIMC** as an independent estimator. Report agreement
  rates by trick number.
- We only ever ask the teacher to **rank ~6 already-reasonable cards**, a far weaker (more
  robust) demand than "find the optimal move."

### 4.2 Three-level ceiling ladder (this is what "provably raise the ceiling" means)
- **Ceiling 1 — Action-space envelope:** best-primitive-per-state under a perfect selector with
  EV oracle. Measures the *vocabulary*. Upper bound only (selector sees EVs the network never
  gets).
- **Ceiling 2 — Realizable selector:** train an *unconstrained* learner (MLP/GBM) on
  `belief → best-primitive` (card-match) to see how much of Ceiling 1 is recoverable **from the
  belief features alone**. Separates "vocabulary good but unselectable from our features" from
  "good and learnable." Ceiling 2 ≪ Ceiling 1 ⇒ the **belief features** are the bottleneck.
- **Ceiling 3 — The WANN:** what the logical-gate network actually reaches. Ceiling 2 − Ceiling
  3 gap quantifies whether the **substrate** is the limiter (and thus how much the "negotiable
  substrate" budget — SIGMOID/weights — is worth).

### 4.3 Debiasing & statistics
- **Winner's-curse debiasing:** split determinized worlds into a **selection set** (pick the
  argmax primitive per state) and an independent **evaluation set** (score that choice).
  Without this, every richer vocabulary looks better merely from maxing over more noisy
  options — the exact false positive we must avoid.
- **Paired comparisons:** designs are compared on the *same* held states and *same* CRN deals,
  with confidence intervals, at each ceiling level. A rise is claimed only when intervals clear.
- **Final arbiter:** the envelope is a *leading indicator*; a statistically-pinned Phase-1
  head-to-head benchmark is the verdict. An envelope↔benchmark disagreement is itself a finding
  (teacher bias exposed), not a silent failure. Note: Phase-1 self-play may *exceed* the
  EV-envelope by exploiting real (non-Elite) opponents — attributed, not treated as a bug.

## 5. Unified Training Signal

Switch Phase-0 fitness from **intent-label classification** to **resolved-card-match accuracy
vs. the teacher** (fraction of states where the resolver's output card == the teacher's
best card; ties → multi-label). This:
- works identically for 3 intents, ~6 primitives, or continuous knobs (decouples training from
  action representation);
- optimizes the thing we actually care about (playing strong cards), not a proxy label;
- is the **same machinery** as the envelope harness (build once, use for both stages).

## 6. Stage A — Richer Discrete Primitives

- **Vocabulary:** ~5–6 primitives *per brain* (separate lead/follow vocabularies, leveraging the
  existing modular split), each a styled deviation that keeps Elite's tactical core (preserves
  the high floor / no-collapse-basin property) but with **bigger, better-separated dials** than
  today's near-duplicates.
  - Lead: `DRAW_TRUMP`, `ESTABLISH_LONG`, `SHORTEN_TO_VOID`, `FEED_PARTNER_LEAD`, `SAFE_EXIT`.
  - Follow: `CHEAP_WIN`, `OVERTAKE_PARTNER`, `SMEAR_POINTS`, `DUCK`, `TRUMP_CUT`, `PRESERVE_TRUMP`.
  - Each is a pure, always-legal function extending `select_card_styled`.
- **Relabel** the dataset by resolved-card-match; **train** with card-match fitness.
- **Gate:** debiased Ceiling-1 envelope of this vocab vs. the current 3-intent baseline on common
  states; train a WANN only if it clears. Report Ceilings 2 and 3 to locate any shortfall.

## 7. Stage B — Continuous Utility Resolver (the high-ceiling bet)

- **Card-utility features** `φ(card, state)`: ~6–10 hand-designed terms (rank, point value,
  wins-trick, beat-margin, preserves-boss, builds-void, feeds-partner, trump-spent, …).
- **Resolver:** `card* = argmax_{legal} Σ_k g(w_k)·φ_k(card,state)`, with the WANN emitting the
  continuous knobs `w` per state. SIGMOID for smooth `g`; sign via the existing ±1 mechanism or
  a [−1,1] remap; optional learned-weight calibration via the existing DE/CMA-ES optimizer
  (`optimize.rs`) — the "negotiable substrate" budget spent precisely here.
- **B-specific Ceiling-1 = reachability:** fraction of states where *some* knob vector resolves
  to the teacher's best card. Low reachability indicts `φ` (enrich features), cleanly separating
  "feature set too weak" from "selection too hard."
- Reuses the card-match fitness and `φ`-style features from Stage A — staging is also code-reuse
  ordering.

## 8. Optional Teacher-Quality Escalation (only if the trust map demands it)

If the §4.1 trust map shows the rollout teacher is too unreliable in early tricks to rank
primitives:
- **Policy-based inference** (Rebstock 2019) with **Elite as `π`**: weight determinized worlds by
  `η(s)=Π π(h,a)` and sample from the posterior. Validate empirically (better inference ≠ better
  play). ~5× cost, offline labeling only, so acceptable. Bonus: the per-world posterior is a
  source for opponent/partner belief features.
- EPIMC / αµ are **last resort** only (expected low-value for public-observation Sueca).

## 9. Implementation Cost & Touch Points

The first real chunk of work is plumbing, not primitives:
- `OUTPUT_COUNT` is a **shared** constant (`sueca_solver/constants.rs`); A needs *per-brain*
  output sizes (lead ≠ follow), B needs `K` knob outputs. This ripples through genome init
  (`OUTPUT_START`, `FIRST_HIDDEN_ID`), `wann_network`, `genome`, `compile_rules`.
- The dataset loader (`dataset.rs`) hard-rejects shapes not matching the current 3-intent /
  35-feature layout → format/version bump + regeneration (per the migration discipline in
  `CLAUDE.md`).
- New card-match fitness path in `train.rs` / `evaluator.rs` (reused by the envelope harness).
- The envelope harness itself (new): teacher-EV per primitive on held states, selection/eval
  world split, three-level reporting.

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Richer vocab re-opens the Phase-1 collapse basin | Every primitive keeps an Elite-quality core/fallback |
| Envelope looks higher just from noise (winner's curse) | Selection/evaluation world split (§4.3) |
| Belief features can't support the new selection | Ceiling-2 diagnostic; enrich features if it lags Ceiling-1 |
| Teacher mis-ranks primitives early-game | Trust map + endgame-exact calibration; rank-not-value demand |
| Better inference doesn't help (or hurts) | Treat §8 as validated-only, not assumed |
| More outputs make WANN search harder | Per-brain vocabularies keep each output layer small; Stage A probes cheaply first |

## 11. Staged Plan with Relative Kill/Go Gates

0. **Harness:** build envelope + card-match fitness + trust map. *(Gate: trust map shows
   rankings are reliable enough to proceed.)*
1. **Stage A:** design discrete vocabularies; measure debiased Ceiling-1 vs. 3-intent baseline.
   *(Gate: A's envelope > baseline by a clear margin → train WANN; report Ceilings 2/3.)*
2. **Stage B:** build the utility resolver + `φ`; measure reachability and Ceilings 1–3.
   *(Gate: B's ceiling > A's → train WANN.)*
3. **Decide** the winner by paired Phase-1 head-to-head benchmark (the arbiter).
4. **Escalate** to §8 only if the trust map demanded it.

## 12. Out of Scope / Future

- Opponent/partner **belief-feature enrichment** (Tier A in `ideas.md`) — the policy-based
  posterior from §8 is the natural data source; pick up after A/B.
- **Trick-end teacher paradigm** — separate project; will adapt to whatever vocabulary wins.
