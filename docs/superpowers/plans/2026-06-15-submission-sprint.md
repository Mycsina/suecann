# Submission Sprint Plan — Sueca WANN (deadline 2026-06-15 23:59)

**Goal:** Ship the four mandatory repo artifacts — `README.md`, `code`, `report`, `presentation slides` — at high quality, in ~22 hours, mostly autonomously overnight.

**Grading:** Code 50% · Report 30% · Presentation 20%.

**Current state (02:06, 2026-06-15):**
- ✅ **Code (50%)** — clean, tested (52 unit + 4 ignored + 1 doctest), documented in README/CLAUDE. v6 champion shipped to website. *Gap:* a one-shot reproducibility path + an explicit "Setup / Reproduce" README section.
- ❌ **Report (30%)** — does not exist. LNCS conference format, 10–12 pp. **Biggest gap & risk.**
- ❌ **Slides (20%)** — do not exist. ≤13 min talk (10 present + 3 Q&A), all members present.
- ✅ Python deps: `pyproject.toml` present (viz scripts only); core is Rust/Cargo. *No further deps work needed* (per user).

## Execution model

- **Orchestrator = this Claude session (Opus).** Authors all accuracy-critical content (report skeleton with REAL numbers, results/analysis prose, slide outline), generates figures, reviews worker output. Usage-limited → delegates bulk.
- **Workers = DeepSeek-v4 via `/tmp/spawn_worker.sh "<prompt>"`** (headless, `--dangerously-skip-permissions`, runs in repo root). Capable but NOT to be trusted on numbers — every worker prompt must cite ground-truth files and forbid inventing results. Run in background; review on completion.

**Ground-truth files workers must cite (never contradict):** `problems.md` (narrative spine: Chapters 1–3), `README.md`, `CLAUDE.md`, `reference.md` (literature), `checkpoints/production/2026-06-14-2/compiled_rules_*.txt`, `docs/superpowers/plans/2026-06-14-wann-strength-then-distill.md`.

## Canonical numbers (single source of truth — paste into skeleton, workers must not alter)

| Fact | Value |
|---|---|
| Champion | v6 = `checkpoints/production/2026-06-14-2`, pop=1000, 600 gens, phase0_gens=150, dataset `expert_states_v6.npz` (rollout teacher) |
| Win vs Elite | **52.1% ± 1.8%**, n=3000, 95% CI [50.3, 53.9] (excludes 50%); card pts 60.2 vs 59.8 |
| v5 champion | 52.7% ± 1.8%; **132 hidden gates / 188 conns** |
| v6 complexity | **29 hidden gates / 49 conns** (4.5× / 3.8× fewer) |
| Prior project champion | 30.2% vs Elite |
| Rollout teacher | 62% vs Elite; ~1000× faster gen (15k states ≈ 11s) |
| 3-intent oracle ceiling | 58% |
| RandomForest feature ceilings | LEAD 62.6% / FOLLOW 70.8% (5-fold) |
| Belief / intents | 35 features (public info only) → 3 intents (MAX_FORCE, EFFICIENT_WIN, EQUITY_BUILDER) → styled resolver → legal card |
| WANN | sign-only ±1; agg SUM/MIN/MAX; act IDENTITY/NOT/THRESHOLD; weight sweep {−2,−1,−0.5,0.5,1,2}, mean at inference |
| Stage 2c | FOLLOW brain 12 steps/depth10 → **5 steps/depth5**; `MAX_FORCE = 2·Game_Pts_Remaining`; verified behavior-preserving (2000-state property test) |

Tournament matrix (row vs col, n=3000), for the results table:
```
                 Random      Old        Elite       WANN
Random           50.0        10.8        4.7         5.1
Old              89.2        50.0       32.5        32.3
Elite            95.3        67.5       50.0        47.9
WANN             95.0        67.7       52.1        50.0
```

---

## Workstream 1 — REPORT (highest priority, 30%)

**Dir:** `report/`. **Format:** LNCS (`llncs.cls`), 10–12 pp body. **Build:** `latexmk -pdf main.tex` (pdflatex available).

**Files:**
- Create: `report/main.tex`, `report/references.bib`, `report/llncs.cls` (+ `splncs04.bst`), `report/figures/*`
- Source of truth: skeleton authored by orchestrator at `report/SKELETON.md`.

### Tasks
- [ ] **1.1 (orchestrator)** Obtain `llncs.cls` + `splncs04.bst` into `report/`. Try `curl` from CTAN; fallback WebFetch. Verify a trivial `\documentclass{llncs}` doc compiles with `latexmk -pdf`.
- [ ] **1.2 (orchestrator)** Write `report/SKELETON.md`: every section with real numbers, key claims, and the exact figures/tables to include. (Outline below.) This is the airtight ground truth for the LaTeX worker.
- [ ] **1.3 (orchestrator)** Generate figures into `report/figures/`:
  - `arch.pdf` — Belief(35)→WANN→intent(3)→resolver→card pipeline (TikZ or draw).
  - `training_curve.pdf` — from `checkpoints/production/2026-06-14-2/training_stats.csv` (fitness/delta vs gen; mark phase0→1 at gen 150).
  - `tournament.pdf` — bar chart of win% vs each bot (from matrix above).
  - `topology_v6.pdf` — v6 champion topology (`compile-rules` already emits `topology_graph_*.png`).
  - `rules_follow.pdf`/listing — the before/after folded FOLLOW rules (interpretability money figure).
  - `complexity.pdf` — v5 vs v6 gate/conn/depth comparison.
- [ ] **1.4 (WORKER A, background)** Expand `report/SKELETON.md` → full `report/main.tex` in LNCS, 10–12 pp, include figures/tables, `references.bib` from `reference.md`. **Must compile** via `latexmk -pdf` (iterate until clean) and **must not alter any canonical number**. Acceptance: PDF exists, 10–12 pp body, 0 LaTeX errors.
- [ ] **1.5 (orchestrator)** Review compiled PDF: numbers correct, figures render, claims match `problems.md`, page count in range, no DeepSeek hallucinations. Fix abstract/intro/conclusion by hand (highest-visibility prose). Commit.

### Report section outline (≈ pages)
1. **Introduction** (1.5) — Sueca; the interpretability motivation (why WANNs + discrete gates); contributions (interpretable rule-extraction that *beats* a strong heuristic; the rollout teacher; the verified rule-minification; the honest resolver-floor analysis).
2. **Background & Related Work** (1.5) — WANNs (Gaier & Ha 2019), NEAT/PFS-NEAT, MAP-Elites, PIMC & rollout policy improvement, trick-taking AI, neurosymbolic rule extraction. (from `reference.md`)
3. **Sueca & Problem Formulation** (1) — rules that matter (rank order, manilha, follow-suit, scoring), POMDP framing, the 35-feature public-info belief state, the 3 intents + styled resolver.
4. **Method** (3) — architecture; WANN constraints (sign-only, gates, weight sweep); co-evolution (lead/follow brains, dynamic routing); PFS-NEAT bootstrap; HOF/MAP-Elites; **the rollout teacher** (policy-improvement theorem, why deep PIMC's myopic leaf was the real weakness); dataset labeling (best-of-3-by-EV).
5. **Interpretability** (1.5) — rule compilation; constant-folding + alias inlining (verified at W=1); complexity metrics; show the actual extracted FOLLOW rules.
6. **Experiments & Results** (2) — benchmark protocol (CRN, seat-rotated, n=3000); the 30%→52% journey; v5→v6 iso-strength @ 4.5× simpler; rollout teacher 62%; ablations; training curves.
7. **Discussion** (1) — the resolver-floor as the true ceiling ("we learned Elite's IF-THEN + a small overlay"); 58% oracle cap; limitations; the fog-of-war / opponent-inference open question.
8. **Conclusion & Future Work** (0.5).

---

## Workstream 2 — SLIDES (20%)

**Dir:** `presentation slides/` (exact mandated name). **Format:** Beamer (`slides.tex` → `slides.pdf`) — reuses LNCS figures.

### Tasks
- [ ] **2.1 (orchestrator)** Write `presentation slides/OUTLINE.md`: ~12–14 slides for a 10-min talk, mapped to the report, with speaker notes and a suggested 3-member split.
- [ ] **2.2 (WORKER B, background)** Build `slides.tex` (Beamer, metropolis or default theme), compile to `slides.pdf`, reuse `report/figures/`. Acceptance: compiles, ≤14 content slides, every claim traceable to the outline.
- [ ] **2.3 (orchestrator)** Review; fix the title/results/demo slides; ensure a live-website demo slide (the inspector shows the v6 net). Commit.

### Slide arc (~10 min)
Title · Problem & motivation (interpretability) · Sueca in 60s · Architecture (the pipeline figure) · WANN = logic gates · How we train (co-evolution + rollout teacher) · Result: beats Elite 52.1% · The interpretability payoff (folded FOLLOW rules) · Honest analysis (resolver floor) · Live demo (website) · Future work · Q&A backup slides.

---

## Workstream 3 — CODE / REPRODUCIBILITY (defends the 50%)

### Tasks
- [ ] **3.1 (orchestrator)** Verify clean state: `cargo build -p sueca_wann --release`, `cargo test --all`, `cargo clippy --all` — record results.
- [ ] **3.2 (WORKER C, background)** Add a top-level **"Reproducing the Results"** section to `README.md` (or `REPRODUCE.md`): exact commands to (a) build, (b) benchmark the v6 champion → 52.1%, (c) compile rules, (d) regenerate the v6 dataset, (e) generate the figures. Add a `Makefile` with `build/test/bench/rules/figures` targets. Must only use commands that already exist in the CLI (verify against `main.rs`). Do NOT touch Rust source.
- [ ] **3.3 (orchestrator)** Run the README commands end-to-end to confirm reproducibility. Fix discrepancies. Commit.

---

## Timeline & checkpoints

| Window | Orchestrator | Workers (background, DeepSeek) |
|---|---|---|
| 02:00–04:00 | LNCS template; report SKELETON; figures; spawn Workers A/B/C | A: report LaTeX · B: slides · C: README/Makefile |
| 04:00–08:00 | Review worker output as it lands; fix prose/numbers; commit | re-spawn for fixes if needed |
| morning (user away) | iterate report→PDF, slides→PDF, repro pass; keep everything compiling & committed | targeted fixes |
| user returns | hand back: PDFs + author names/affiliation/team-split needed; final polish | — |

## Needs the user (flagged, non-blocking)
- Author names, student IDs, affiliation, course/instructor for the title page (placeholders used until then).
- Team split for the 3 presenters (suggested split provided in OUTLINE.md).
- Any framing preference on emphasis (interpretability headline vs strength).

## Acceptance (definition of done)
- `report/main.tex` → `report/main.pdf`, 10–12 pp, LNCS, compiles clean, all numbers match canonical table.
- `presentation slides/slides.pdf` compiles, ≤14 slides, 10-min pace.
- README has a verified reproduce section; `cargo test --all` green.
- Repo root has: `README.md`, `code` (the Rust/Python tree — confirm naming vs criterion), `report/`, `presentation slides/`. All committed & pushed.

## Risks
- **DeepSeek hallucinating numbers** → mitigated: canonical table baked into skeleton; orchestrator reviews every number.
- **LaTeX not compiling** → workers must iterate to a clean build; orchestrator has pdflatex/latexmk to verify.
- **"code" naming** — criterion lists a `code` entry; our code is at repo root (`src/`, `scripts/`). Decide: keep root layout + document in README, or add a thin pointer. (Low risk; confirm with user.)
- **Network for `llncs.cls`** — if Bash has no network, use WebFetch; fallback is a vendored copy.

---

## OVERNIGHT RESULTS (completed ~03:00, 2026-06-15)

All four mandatory deliverables exist, compile, are committed and pushed to `origin/master`.

| Deliverable | Status | Notes |
|---|---|---|
| README.md | ✅ | + "Reproducing the Results" section; stale recovery banner replaced with v6 status |
| code | ✅ | Rust workspace (`src/`) + Python viz (`scripts/`); `cargo test --all` green (53 tests); top-level `Makefile` |
| report | ✅ | `report/main.pdf`, LNCS, **11 body pages** + refs, compiles 0 errors |
| presentation slides | ✅ | `presentation slides/slides.pdf`, Beamer/metropolis, 14 slides, 10-min split |

**Workers (DeepSeek) ran:** D (figures, /semiology-of-graphics), C (README+Makefile), A (report, /scientific-writing), B (slides, both skills). All output **reviewed by orchestrator**.

**Issues I caught & fixed during review (DeepSeek is not trusted on facts):**
- Report: invented positional figure (46.0%) → softened to verified statement.
- Report: missing `\titlerunning`/`\authorrunning` ("Title Suppressed…" header) → added.
- Report: **technical error** — sign-edge described as `s·W·out` (negation); corrected to NOT-inversion `1−out` then ×W (matches `wann_network::forward`).
- Verified independently: every canonical number, no undefined refs/citations, all figures present, page count in range.

**Bonus fix:** real DOT-export bug (`fontname=Helvetica-Bold` unquoted) — topology graphs now render.

## ⚠️ NEEDS YOU (blocking final submission)
1. **Author names + student IDs + affiliation + course** — placeholders `TODO Author One/Two/Three`, `TODO University, TODO Course` in:
   - `report/main.tex` lines ~39–40 (`\author`, `\institute`, and `\authorrunning`)
   - `presentation slides/slides.tex` (title frame)
   Then rebuild: `make report` and `make slides` (or `cd report && latexmk -pdf main.tex`).
2. **Presenter split** — proposed in `presentation slides/OUTLINE.md` (P1 slides 1–5, P2 6–9, P3 10–13). Confirm/adjust.
3. **"code" folder naming** — the criterion lists a `code` entry; our code lives at repo root (`src/`, `scripts/`) with README at root. Most graders read this as "the codebase," but confirm whether they want a literal `code/` folder (a restructure I did NOT do — it would churn the Cargo workspace paths; low risk to leave as-is + document).

## Optional polish if time allows
- Read `report/main.pdf` end-to-end for tone/framing once your name is on it.
- `make bench` reproduces the 52.1% headline (a few minutes) if you want a fresh `tournament_report.csv` as evidence.
