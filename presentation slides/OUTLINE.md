# Presentation Outline — Sueca WANN

**Talk duration:** 10 minutes + 3 minutes Q&A
**Presenters:** 3 (Presenter 1, Presenter 2, Presenter 3)
**Total slides:** 13 (12 content + 1 backup/Q&A)

---

## Presenter 1 — Problem, Domain & Architecture (slides 1–5, ≈3.5 min)

### Slide 1 — Title (0:30)
**Message:** *We evolved an interpretable, Elite-beating Sueca agent from logic gates.*

- Title: "Interpretable Card-Play Strategies from Evolved Logic Networks"
- Subtitle: A Neurosymbolic Agent for Sueca
- Authors: TODO Author One, TODO Author Two, TODO Author Three
- Affiliation: TODO University, TODO Course
- **Speaker notes:** Welcome the audience. One-sentence hook: "We built an AI that plays a Portuguese card game at a competitive level — and you can read its entire strategy in 5 lines of IF/THEN rules." Briefly introduce the team.

### Slide 2 — The Problem: Black-Box AI vs. Interpretable Play (1:00)
**Message:** *Most strong card-playing AIs are black boxes; we asked whether competitive play can be human-auditable.*

- Left side: examples of opaque AI (deep PIMC, neural nets — millions of floats)
- Right side: our goal — a handful of readable IF/THEN rules
- Key question: "Can we get *competitive* play that is simultaneously *human-auditable*?"
- **Speaker notes:** Frame the tension. Strong card AIs exist (Bridge, Skat) but they're opaque matrices. Our thesis: discrete logic gates + neuroevolution can produce something you can read, verify, and learn from. This isn't just academic — interpretability matters for trust, teaching, and debugging.

### Slide 3 — Sueca in 60 Seconds (0:45)
**Message:** *Sueca's distinctive rules create a rich strategic texture that challenges any AI.*

- 4 players, 2 partnerships (opposite seats), 40-card deck
- Counter-intuitive rank: **A > 7 > K > J > Q > 6 > 5 > 4 > 3 > 2** (7 = *manilha*)
- Points: A=11, 7=10, K=4, J=3, Q=2 → 120 per deal
- Must follow suit; trump cuts; public void tracking
- Imperfect information: you see only your hand + public history
- **Speaker notes:** Don't recite all rules — highlight the ones that shape strategy. The 7-as-second-highest is counter-intuitive and matters. Void tracking means players know who's out of which suit. 120 points with game-point tiers (1, 2, or 4) means every trick counts. Keep it visual — show a diagram of the table, not bullet points.

### Slide 4 — Architecture: Belief → Gates → Intent → Card (0:45)
**Message:** *The agent reads a 35-feature belief state through evolved logic gates and outputs one of three strategic intents.*

- Show `arch.pdf` (the pipeline diagram)
- 35 public-information-only features (no future-trick leakage)
- 3 intents: MAX_FORCE, EFFICIENT_WIN, EQUITY_BUILDER — each a styled deviation around Elite policy
- Resolver guarantees 100% legal card output
- **Speaker notes:** Walk through the pipeline left to right. Emphasize: the belief state uses ONLY information a human can observe — hand composition, trick state, void knowledge, tactical affordances like "can I beat the current winner?" The three intents aren't weak standalone tactics; each is Elite-level play plus a stylistic dial. The resolver maps intent → legal card deterministically. Point to the figure as you explain.

### Slide 5 — WANN = Evolved Logic Gates (0:30)
**Message:** *Weight-Agnostic Neural Networks replace learned weights with evolved topology — perfect for rule extraction.*

- Node types: aggregations (SUM, AND, OR) + activations (IDENTITY, NOT, THRESHOLD)
- Connections carry only a sign (±1), no learned weight
- Networks evaluated across a weight sweep W ∈ {−2, −1, −0.5, 0.5, 1, 2}
- No MEAN (precision issues), no SIGMOID (breaks rule extraction)
- **Speaker notes:** This is the "why WANNs" slide. Traditional neural nets have continuous weights — you can't read them. WANNs use discrete logic gates: a SUM node adds its inputs, an AND node takes the minimum, a THRESHOLD node fires at >0.5. Because every component is a logic primitive, the whole network compiles to Boolean-ish rules. The shared weight sweep means the topology must work at multiple scales — it can't cheat by fine-tuning weights. This is the architectural constraint that enables interpretability.

---

## Presenter 2 — Training & Results (slides 6–9, ≈3.5 min)

### Slide 6 — Two-Phase Training Pipeline (0:45)
**Message:** *Training bootstraps on expert demonstrations, then switches to co-evolutionary self-play of separate lead and follow brains.*

- Phase 0 (gens 0–149): Supervised pretraining on PIMC expert dataset
  - PFS-NEAT from zero connections — networks grow their own inputs
  - Dataset split by leading vs. following
- Phase 1 (gens 150–599): Co-evolutionary self-play
  - Lead and Follow brains co-evolve; dynamic per-decision routing
  - Delta-fitness with Common Random Numbers for low-variance evaluation
- Show `training_curve.pdf`
- **Speaker notes:** The two phases address different problems. Phase 0 gives the networks a warm start — they learn to imitate expert play from labeled states. PFS-NEAT means they start with zero connections and only add connections that provably help. Phase 1 is where they learn to beat the heuristic: lead and follow brains co-evolve, playing games where every decision routes to the right brain. CRN means we compare apples to apples — same deals, same seats, same opponents.

### Slide 7 — The Rollout Teacher: A Key Contribution (1:00)
**Message:** *A cheap supra-Elite teacher (62% vs. Elite, ~1000× faster than deep search) made the champion 4.5× simpler at equal strength.*

- Problem diagnosed: the project's deep PIMC solver only *tied* Elite — its leaf evaluation was myopic (raw score, no positional estimate)
- Solution: flat Monte-Carlo PIMC with Elite playouts — by rollout policy improvement, ≥ Elite
- Result: 15k labeled states in ~11 seconds (vs. hours for alpha-beta)
- Surprising finding: stronger teacher → *simpler* champion, not stronger (strength is resolver-floored)
- **Speaker notes:** This is the most novel technical contribution. We discovered the deep search wasn't weak because of shallow depth — it was weak because the leaf evaluator was blind. The rollout teacher fixes this cheaply: for each legal move, determinize the hidden cards and play out the rest of the deal with Elite in all seats. Average the outcomes. This is provably ≥ Elite and massively cheaper. But here's the surprising part: a better teacher didn't make a stronger champion. It made a *simpler* one — 29 gates instead of 132 — because cleaner labels need less structure to fit. This reframes the whole story: the teacher buys interpretability, not strength.

### Slide 8 — Result: The WANN Beats EliteHeuristicBot (0:45)
**Message:** *The champion beats EliteHeuristicBot 52.1% ± 1.8% (n=3000) — a statistically significant win.*

- Show `tournament.pdf` (the bar chart) and the tournament matrix table
- 52.1% ± 1.8%, 95% CI [50.3%, 53.9%] — excludes 50%
- Card points: 60.2 vs. 59.8
- Dominates RandomBot (95.0%), beats OldHeuristic (67.7%)
- Context: the project had *never* beaten Elite before (prior best: 30.2%)
- **Speaker notes:** Present the headline number clearly. The 95% confidence interval excludes 50%, so this is a real win, not noise. The margin is small (+2.1 pp) but that's expected — Elite is near the fast-policy ceiling. For context, this project's best-ever result before June 14 was 30.2% vs. Elite. The jump from 30% to 52% came from diagnosing *why* the network was failing (it had collapsed to a constant output) and fixing the resolver. Point to the tournament matrix: WANN beats OldHeuristic 67.7% vs. Elite's 67.5% — they're neck-and-neck against the weaker baseline.

### Slide 9 — The Journey: From 30% to 52% (0:30)
**Message:** *Three root-cause fixes — not blind retraining — turned a collapsed network into an Elite-beating champion.*

- Old resolver: pure intents at 20%/37%/25% vs. Elite → network collapsed to "always EFFICIENT"
- Fix 1: Styled resolver — every intent now ≈Elite (48%/47%/46%) → collapse merely ties Elite
- Fix 2: Best-of-3-intent labeling — train on the best available choice, not the global optimum
- Fix 3: Rollout teacher — cleaner labels → 4.5× simpler champion
- **Speaker notes:** This is the "honest engineering" slide. We built a diagnostic harness that ran in seconds instead of retraining for 90 minutes. It revealed the network had functionally dropped an entire intent and was playing a constant policy. The fix wasn't more training — it was redesigning the resolver so each intent was individually strong. Once the floor was raised, the network could actually learn situational selection. This slide transitions to Presenter 3.

---

## Presenter 3 — Interpretability, Honest Analysis & Future (slides 10–13, ≈3.0 min)

### Slide 10 — Interpretability Payoff: 5 Gates, Fully Readable (1:00)
**Message:** *The follow brain compiles to 5 logic gates at depth 5 — a strategy you can read, verify, and learn from.*

- Show `topology_follow.pdf` (before) and the folded rules listing (after)
- Before: 12 steps, max depth 10, cluttered with dead constants and alias chains
- After: 5 steps, max depth 5 — verified behavior-preserving on 2000 random states
- Show the actual rules:
  ```
  hidden_42 = NOT(Holds_Boss_Led)
  hidden_48 = (1 + Any_Opp_Void_Led)
  hidden_41 = NOT((1 + hidden_42 + 2·hidden_48))
  hidden_40 = THRESHOLD((hidden_41 + Trump_Count) > 0.5)
  hidden_55 = THRESHOLD(hidden_40 > 0.5)
  MAX_FORCE      = 2·Game_Pts_Remaining   ← force early, concede late
  EFFICIENT_WIN  = 0                        ← never, when following
  EQUITY_BUILDER = 1 + 2·hidden_41 + hidden_55
  ```
- Show `complexity.pdf` (v5 vs. v6 gate/connection counts)
- **Speaker notes:** This is the payoff slide — the whole reason we used WANNs. Walk through one rule concretely: `hidden_42 = NOT(Holds_Boss_Led)` means "I don't have the boss in the led suit." The network chains these together into a decision. MAX_FORCE tracks game progress — force early, concede late. EFFICIENT_WIN is always 0 in both brains: the network never delegates to pure Elite; it always applies a learned deviation. The constant-folding and alias inlining are *verified* — we ran the real champion on 2000 random states and proved the removed symbols carry no information. The complexity chart shows v6's 29 gates vs. v5's 132 at the same strength.

### Slide 11 — Honest Analysis: The Resolver Floor (0:45)
**Message:** *The styled resolver that enabled the win also sets the strength ceiling — and we can measure exactly where it is.*

- Resolver floor: every pure intent ≈Elite → collapse merely ties; good mixing wins
- Oracle cap: perfect 3-intent selection = 58% vs. Elite (measured, not guessed)
- We're at 52% — ~6 points of headroom remain
- Bottleneck: the 3 intents are nearly equivalent in game fitness → weak gradient for learned selection
- Also: lead-brain decision is harder (RandomForest: +2.5 vs. majority); follow is learnable (+7.9)
- **Speaker notes:** This is the "honest negative result" the paper owns. The styled resolver is a double-edged sword: it removed the collapse basin (good) but made the intents nearly equivalent (limiting). Perfect intent selection — an oracle that always picks the best of the 3 — caps at 58%. We're at 52%, so the learned overlay is real but small. The path to >58% requires richer intents or opponent-card inference features. This is honest science: we measure our ceiling and report it.

### Slide 12 — Live Demo: The Website Network Inspector (0:30)
**Message:** *You can explore the v6 champion's decision-making interactively in the browser.*

- Screenshot or description of the web interface at localhost:5173
- Shows the network graph, belief features, intent outputs, and resolved card
- Built with React + WASM (Rust compiled to WebAssembly)
- **Speaker notes:** "Let me show you this live." Open the browser, play a hand of Sueca against the WANN. Point out: the network inspector shows every feature value, every gate's output, and which intent fired. You can see the logic flowing in real time. This isn't a black box — you can literally watch it think. Transition: if time is tight, skip the live demo and describe it from the slide.

### Slide 13 — Future Work & Conclusion (0:30)
**Message:** *Richer intents and opponent-card inference are the path to breaking the 58% ceiling — and the interpretability pipeline is ready.*

- Richer intents: raise the oracle cap above 58% by adding more-differentiated play styles
- Opponent-card inference: can a tiny logic circuit learn to infer hidden cards from optimal play?
- Knowledge distillation: train an even smaller student from the champion
- Conclusion: We evolved an interpretable, Elite-beating Sueca agent with a verified pipeline from logic gates to human-readable rules. The code, dataset, and rules are available.
- **Speaker notes:** Three clear directions. (1) More intents with sharper distinctions would raise the oracle cap. (2) The fog-of-war problem: can a logic network infer opponent holdings from their play? This is the Bridge AI problem in miniature. (3) Distillation: the champion could teach an even smaller network. End with the takeaway: we showed that competitive, interpretable card-play AI is possible — and the rules fit on one slide.

---

## Backup Slide — Q&A / References

**Message:** *Thank you. Questions?*

- Key references: Gaier & Ha 2019 (WANN), Stanley & Miikkulainen 2002 (NEAT), Mouret & Clune (MAP-Elites)
- Reproducibility: all commands in README.md; canonical champion at `checkpoints/production/2026-06-14-2/`
- Contact: TODO
- **Speaker notes:** Have this slide visible during Q&A. Be prepared for likely questions: "Why only 52%?" (resolver floor — see slide 11), "Does this generalize to other games?" (architecture is game-agnostic; the belief features and resolver are Sueca-specific), "How long does training take?" (~1 hour on a laptop), "What happens if you add more intents?" (open question — slide 13).

---

## Presenter Division Summary

| Presenter | Slides | Time | Content |
|-----------|--------|------|---------|
| Presenter 1 | 1–5 | ~3.5 min | Title, Problem, Sueca rules, Architecture, WANN concept |
| Presenter 2 | 6–9 | ~3.5 min | Training pipeline, Rollout teacher, Results, The journey |
| Presenter 3 | 10–13 | ~3.0 min | Interpretability, Honest analysis, Live demo, Future work |

**Handoff cues:**
- P1→P2: "Now [Presenter 2] will explain how we train these networks."
- P2→P3: "And [Presenter 3] will show you what the network actually learned — you can read it yourself."
- P3 closes and opens Q&A.
