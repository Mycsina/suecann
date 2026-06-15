# Report Skeleton — GROUND TRUTH for the LaTeX worker

> This file is the authoritative content + numbers for `report/main.tex`. Every number
> here is verified (see `problems.md`, `README.md`). **Do not invent or alter numbers.**
> Expand each section into polished LNCS prose; keep the body to **10–12 pages**.

**Working title:** *Interpretable Card-Play Strategies by Evolving Weight-Agnostic Logic Networks: A Neurosymbolic Agent for Sueca*

**Authors:** TODO (placeholders `\author{First Author\inst{1}}` etc.) — user will supply names/IDs/affiliation. Leave a clearly-marked TODO.

**Venue note:** LNCS conference-paper format, `\documentclass[runningheads]{llncs}`.

---

## Abstract (≈180 words)
Neurosymbolic agent for **Sueca** (4-player, partnership, trick-taking, imperfect-information).
We evolve **Weight-Agnostic Neural Networks** (WANNs; Gaier & Ha 2019) whose nodes are **logic gates**
(aggregations SUM/AND/OR, activations IDENTITY/NOT/THRESHOLD) and whose connections carry only a
**sign** (±1) under a shared weight sweep. The network reads a **35-feature, public-information-only
belief state** and emits one of **3 abstract play intents**, which a styled resolver maps to a legal
card. Training is two-phase: supervised bootstrap on a PIMC expert dataset (PFS-NEAT from zero
connections), then co-evolutionary self-play of separate **lead** and **follow** brains. A key
contribution is a cheap **supra-Elite rollout teacher** (flat Monte-Carlo PIMC with Elite playouts,
62% vs the strong heuristic) replacing a myopic deep search. The champion **beats the strong
EliteHeuristic baseline 52.1% ± 1.8% (n=3000)** and, crucially, **compiles to a handful of
human-readable IF/THEN rules** via verified constant-folding and alias inlining. We give an honest
analysis of the *resolver-floor* ceiling and the strength↔interpretability trade-off.

**Keywords:** weight-agnostic neural networks · neuroevolution · neurosymbolic AI · rule extraction · imperfect-information games · trick-taking card games.

---

## 1. Introduction (≈1.5 pp)
- Trick-taking card games are a classic AI-hard domain: imperfect information, partnership, large
  state space. Sueca specifics (Portuguese, 40-card, manilha rank order, 120 points/deal).
- Most strong card AIs (deep PIMC, neural nets) are **black boxes**. Our thesis: can we get
  *competitive* play that is **human-auditable** — extractable as IF/THEN rules?
- Approach in one line: **Belief(35) → WANN of logic gates → intent(3) → resolver → card.** WANNs +
  discrete activations are chosen *precisely because* they compile to Boolean-ish rules (cite WANN).
- **Contributions:**
  1. A neurosymbolic Sueca agent that **beats a strong heuristic** (52.1%±1.8%, n=3000) while
     remaining interpretable — first time this project beat Elite (prior best 30.2%).
  2. A cheap **rollout policy-improvement teacher** (62% vs Elite, ~1000× faster than deep
     alpha-beta) — diagnosis that the deep search's *myopic leaf eval*, not depth, was the weakness.
  3. **Verified post-hoc rule minification** (constant-folding + alias inlining), giving an
     iso-strength champion with **4.5× fewer gates** (29 vs 132) and readable rules.
  4. An **honest negative result**: the styled-resolver *floor* is the real strength ceiling
     (≈58% oracle cap); a stronger teacher buys *simplicity*, not strength.

## 2. Background and Related Work (≈1.5 pp)
- **WANNs** (Gaier & Ha 2019): topology search with a single shared weight; we adapt with sign-only
  edges + a 6-point weight sweep {−2,−1,−0.5,0.5,1,2} averaged at inference.
- **NEAT / PFS-NEAT** (Stanley & Miikkulainen 2002; Whiteson et al. 2005): topology evolution;
  progressive feature selection (we start from 0 connections, classify mutations structural vs not).
- **Coevolving modular behavior** (Reisinger et al. 2004 — L-NEAT): justifies separate lead/follow brains.
- **MAP-Elites** (Mouret & Clune) and **multi-objective parsimony** for diversity + bloat control.
- **PIMC / determinization** and **rollout policy improvement** (Tesauro-style; Ng et al. 1999 for
  shaping; the determinization refs in reference.md); **CRN** for low-variance evaluation.
- **Rule extraction / neurosymbolic** framing: discrete activations → Boolean rules.

## 3. Sueca and Problem Formulation (≈1 p)
- Rules that affect strategy: rank order A>7>K>J>Q>6>5>4>3>2 (7=manilha is 2nd highest!), points
  A=11,7=10,K=4,J=3,Q=2 (120/deal), partners opposite (0&2 vs 1&3), counter-clockwise, follow-suit,
  trump cuts, game-point tiers. Public void tracking.
- **POMDP framing**: each seat sees only its hand + public history. No hand leakage (belief uses
  public info only).
- **Belief state (35 floats ∈ [0,1])** — summarize the categories (hand composition, trick context,
  void knowledge, boss/cut detection, "can I win & at what cost", game progress). Reference full
  table to README (Table in appendix optional). Emphasize: *no future-trick knowledge*.
- **3 intents** (MAX_FORCE, EFFICIENT_WIN, EQUITY_BUILDER) + **styled resolver**: each intent is a
  styled deviation *around* the strong Elite policy (EFFICIENT == Elite; MAX_FORCE = Elite +
  trump-draw lead; EQUITY = Elite + short-suit lead + duck cheap tricks + preserve trump). Resolver
  guarantees 100% legality.

## 4. Method (≈3 pp) — **FIG: arch.pdf (pipeline) near here**
### 4.1 WANN representation
- Connection genes (innov, src, dst, sign∈{±1}, enabled); node genes (id, type, activation, aggregation).
- Sign-only weights + shared weight sweep; aggregations SUM/MIN(AND)/MAX(OR); activations
  IDENTITY/NOT/THRESHOLD; all outputs clamped [0,1]. Why no MEAN/SIGMOID (rule-extraction + precision).
- 35 input + 1 bias + 3 output base nodes; argmax over intents with random tie-break.
### 4.2 Evolution
- PFS-NEAT from 0 connections; structural vs non-structural mutation classes; adaptive 2-stage PFS
  sampling + tabu veto list. Speciation (first-fit, capped at 20). Rank-based selection;
  50/50 Pareto(perf,simplicity)/perf. HOF (50) + MAP-Elites (10×10) opponent pools.
- **Delta-fitness with CRN**: each genome vs HeuristicBot on identical deal/seat rotations; deals
  re-seeded per generation to prevent overfitting.
### 4.3 Two-phase curriculum  — **FIG: training_curve.pdf near here**
- Phase 0 (gens 0–149): supervised classification accuracy on lead/follow dataset splits.
- Phase 1 (gens 150–599): co-evolutionary self-play; dynamic per-decision routing to lead/follow
  brain via the AmILeading flag; HOF→Phase-1 fitness transfer.
### 4.4 The rollout teacher (a key contribution)
- Problem: even the project's deep PIMC (alpha-beta + late minimax) only **ties** Elite — root cause
  is the **myopic leaf eval** returning raw team score, not positional value.
- Fix: **`solve_pimc_rollout`** — determinize unseen cards, then finish each world with **Elite
  playouts**. By rollout policy improvement (1-ply + base policy ≥ base policy) it is **supra-Elite
  (62% vs Elite)** and ~1000× cheaper (15k labeled states ≈ 11s vs ~hours).
- Dataset labeling: **best-of-3-intent by resolved-card EV** (statistically-tied → multi-label;
  all-tied → reject). Natural ~binary EFFICIENT-vs-EQUITY split.

## 5. Interpretability: from network to rules (≈1.5 pp) — **FIG: topology + rules listing**
- `compile-rules` walks the genome → IF/THEN per intent. Two **behavior-preserving** rewrites
  (proved/empirically verified **at W=1**):
  1. **Constant-folding**: nodes fed only by BIAS / empty aggregations are constants; inline them.
  2. **Alias inlining**: a single-input IDENTITY node is a rename (SUM(x)=MIN(x)=MAX(x)=x); flatten
     chains with negation parity. Single-operand AND/OR unwrap.
- **Verification**: a property test runs the *real champion* on 2000 random states and asserts every
  folded constant is invariant and every alias equals its resolved source to 1e-9 → the dropped
  symbols carry no information the network uses.
- **Result (FOLLOW brain): 12 steps / depth 10 → 5 steps / depth 5.** Show the actual rules:
```
FOLLOW brain (compiled, folded):
  hidden_42 = NOT(Holds_Boss_Led)
  hidden_48 = (1 + Any_Opp_Void_Led)
  hidden_41 = NOT((1 + hidden_42 + hidden_48 + hidden_48))
  hidden_40 = THRESHOLD((hidden_41 + Trump_Count) > 0.5)
  hidden_55 = THRESHOLD(hidden_40 > 0.5)
  MAX_FORCE      = Game_Pts_Remaining + Game_Pts_Remaining   # = 2·Game_Pts_Remaining
  EFFICIENT_WIN  = 0
  EQUITY_BUILDER = 1 + hidden_41 + hidden_55 + hidden_41
LEAD brain (compiled): EQUITY_BUILDER = THRESHOLD(NOT(Trump_Count));  EFFICIENT_WIN = 0;
  MAX_FORCE = (ahead & point-dense & partner-winning) OR (no very-short side suit)
```
- Read it strategically: the agent never plays "pure Elite" (EFFICIENT_WIN=0 in both brains); it
  always applies one of two learned deviations — lead: build-voids-when-trump-poor vs press-when-ahead;
  follow: force-early vs equity-late. Applying these selectively is what turns a 50% tie into 52%.

## 6. Experiments and Results (≈2 pp) — **FIG: tournament.pdf, complexity.pdf; TABLE: matrix**
- **Protocol**: seat-rotated duplicate deals, CRN, n=3000; report win% ± 95% CI (half-width).
- **Headline**: WANN beats Elite **52.1% ± 1.8% (CI [50.3,53.9], excludes 50%)**, card pts 60.2 vs 59.8.
- **Tournament matrix** (row vs col, win%, n=3000):
```
              Random   Old    Elite   WANN
Random         50.0   10.8    4.7     5.1
Old            89.2   50.0   32.5    32.3
Elite          95.3   67.5   50.0    47.9
WANN           95.0   67.7   52.1    50.0
```
- **The journey**: prior champion 30.2% → styled resolver + best-of-3 dataset (v5, 52.7%) →
  rollout teacher (v6, 52.1% but **4.5× simpler**: 29 vs 132 hidden gates; 49 vs 188 conns).
- **Teacher quality**: rollout 62% vs Elite vs deep-PIMC ≈ ties; gen time ~1000× faster.
- **Diagnostics (training-free harness)**: 3-intent oracle (perfect selection) caps at **58%**;
  RandomForest on the belief features reaches **LEAD 62.6% / FOLLOW 70.8%** (5-fold) — features carry
  signal but the intent set + resolver floor cap realizable strength.

## 7. Discussion (≈1 p)
- **The resolver-floor ceiling**: every intent resolves to an Elite-flavored card, so a policy
  collapse merely *ties* Elite; the +2% edge is the learned deviation overlay. Stronger teacher →
  cleaner labels → *simpler* champion, **not stronger** (the central, slightly counter-intuitive
  finding). Realizable headroom over Elite ≈ 8 points (58% oracle cap).
- **"Did we just learn Elite's IF-THEN?"** Substantially yes (action space is Elite-floored) but the
  net never delegates to pure Elite; it learned *when to deviate*. Honest framing.
- **Limitations**: 3-intent bottleneck; lead-brain bootstrap fragility; Elite ≈ fast-policy ceiling;
  single game (Sueca).
- **Open question (fog of war)**: can a tiny logic circuit learn to *infer opponent cards* from
  assumed-optimal play, given only public info? A path to break the 58% cap via richer
  inference-features + more-differentiated intents.

## 8. Conclusion and Future Work (≈0.5 p)
- We evolved an interpretable, Elite-beating Sueca agent and gave a verified pipeline from network to
  readable rules. Future: richer intents (raise the oracle cap), opponent-inference belief features,
  knowledge distillation into an even smaller student (measured strength↔interpretability Pareto curve).

---

## Tables/figures inventory (all in `report/figures/`)
- `arch.pdf` — **worker to create in TikZ**: Belief(35) → WANN(logic gates) → intent(3) → resolver → card.
- `training_curve.pdf` — Phase-0 val-acc + Phase-1 fitness (generated).
- `tournament.pdf` — win% vs baselines, n=3000 (generated).
- `complexity.pdf` — v5 vs v6 gates/conns (generated).
- `topology_lead.pdf`, `topology_follow.pdf` — v6 champion graphs (generated). Use one (follow) as the
  "before minification" visual; pair with the folded rules listing.
- Tournament matrix → a `tabular` (data above).
- Belief-feature table → optional appendix (full 35 from README; not counted in 10–12 pp).

## Citations to include (build `references.bib` from `reference.md`)
Gaier & Ha 2019 (WANN); Stanley & Miikkulainen 2002 (NEAT); Whiteson et al. 2005 (PFS-NEAT);
Reisinger et al. 2004 (coevolving modular/L-NEAT); Mouret & Clune (MAP-Elites); Ng et al. 1999
(reward shaping); La Cava et al. 2019 (lexicase, future work); Burch et al. 2018 (AIVAT, eval);
plus PIMC/determinization + parsimony refs listed in `reference.md`. Use `splncs04.bst`.
