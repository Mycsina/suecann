# Report v2 Revision Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Apply the **scientific-writing** skill throughout. This is a *prose* revision, not code: each task gives exact line locations and before/after text instead of failing tests.

**Goal:** Make `report/main.tex` readable by a cold LNCS reviewer with zero project context, by (1) defining all in-project jargon at first use, (2) reordering so prior blockers are introduced before the runs that overcame them, (3) justifying the public-information-only belief features (especially Boss), and (4) purging all em-dashes.

**Approach:** Single-file edit pass on `report/main.tex`, ordered major-to-minor per the scientific-writing skill (structure/content first, mechanical purge last). Rebuild and verify after each task. Keep the body within 10–12 pages.

**Tech Stack:** LaTeX (llncs.cls, splncs04.bst), latexmk, `make report`, pdfinfo, grep.

---

## Why / Who (the audience, per the First Law)

The reader is a **secondary, mixed audience**: a conference reviewer or program-committee member who is a specialist in AI/games but a *nonspecialist in this project*. They read the PDF in isolation, cannot ask us questions, and will not have read the README, the code, or any prior draft. Every term must resolve from the page. Every claim must defend itself on the page. The four defects below all stem from writing as if the reader shares our project memory.

## File Structure

- **Modify:** `report/main.tex` (only file touched for content).
- **Unchanged:** `report/references.bib`, figures, `llncs.cls`, `splncs04.bst`.
- **Rebuild artifact:** `report/main.pdf` (regenerated, committed).

## Definitions to standardize (use these exact names everywhere)

Decide once and apply globally:

- **Elite** — the strong hand-crafted heuristic opponent; the bar to beat. First mention must gloss what it does. Internal code names `EliteHeuristicBot` / `HeuristicBot` collapse to this single name in prose.
- **OldHeuristic** — the earlier, weaker rule-based bot (code `OldHeuristicBot`).
- **RandomBot** — uniform-random legal play.
- **earlier (alpha-beta-teacher) champion** — replaces "v5 champion".
- **final (rollout-teacher) champion** — replaces "v6 champion".
- **rollout-teacher dataset** — replaces "v6 dataset".

> **Accuracy check before writing (ask the specialist = the codebase):** confirm whether the training fitness baseline (`HeuristicBot`, main.tex:305, 348) is the *same* policy as the tournament `EliteHeuristicBot`. If yes, call both "Elite" and say so explicitly. If no, they are two different bots and must keep distinct names with a one-line gloss for each. Do not unify names on assumption. Verify in `src/sueca_wann/src/evaluator.rs` / `benchmark.rs`.

---

## Task 0: Figure alignment / overlap fixes (COMPLETED 2026-06-15)

All six figures were diagnosed by rendering each PDF and inspecting it visually, then
fixed at the source and re-verified by re-rendering. Visual-overlap bugs need a
vision-in-the-loop, so these were done directly rather than via a sightless worker.

- [x] **arch.pdf** (`report/figures/arch.tex`): the floating "each $\approx$ Elite floor"
  label sat on the Intents box top and collided with the phase brace. **Fix:** moved it
  *inside* the Intents box as a tiny italic subtitle; removed the floating node; widened
  horizontal node spacing (`8mm`$\to$`11mm`). Top now shows only the two clean braces.
- [x] **training_curve.pdf** (`scripts/make_report_figures.py`): (a) the "Phase 0/1"
  annotations (top-right) collided with the right-edge direct labels $\to$ moved to
  **top-left**; (b) right-panel "lead"/"follow" labels overlapped each other (both series
  end near the same fitness) $\to$ **staggered vertically** ($\pm$8 pt by which ends higher);
  (c) widened right x-margin so labels fit.
- [x] **tournament.pdf**: title de-jargoned $\to$ "WANN Champion vs Baselines (n = 3000,
  seat-rotated)" (dropped "v6", dropped em-dash).
- [x] **complexity.pdf**: title $\to$ "Equal strength, 4.5$\times$ fewer gates"; legend
  relabelled "Earlier champion (52.7%)" / "Final champion (52.1%)" (no v5/v6) and **moved
  to upper-left** so it no longer covers the "188" bar-top label.
- [x] **topology_lead.pdf / topology_follow.pdf** (`src/sueca_wann/src/compile_rules.rs`,
  `export_topology`): stray unlabeled numbered circles ("47", "54", ...) and dotted
  edge-spaghetti. **Root cause:** edges were emitted for *all* connection genes, including
  those touching unreachable (undeclared) nodes, so graphviz auto-created bare default
  nodes. **Fix:** skip any edge whose endpoints are not both in `reachable`. Rebuilt,
  re-ran `compile-rules`, re-rendered both PDFs via `dot -Tpdf`. Every node is now labeled;
  only faithful disabled-edge stubs between real nodes remain.

Verification: report rebuilt clean (13 pp total; body within 10--12), no undefined
refs/citations; `cargo test --all` re-run after the Rust change.

---

## Task 1: Name the baselines, fix abstract jargon

**Files:** Modify `report/main.tex:46-70` (abstract), `report/main.tex:88-90` (intro black-box paragraph).

The abstract names "supra-Elite" (line 58) and "EliteHeuristic baseline" (line 61) before "Elite" is ever defined. Define it at its first mention, then later uses are legal.

- [ ] **Step 1: Gloss Elite at first abstract mention.**

Locate (lines 57-60):
```
A key contribution is a
cheap \textbf{supra-Elite rollout teacher} (flat Monte-Carlo PIMC with Elite
playouts, achieving 62\% vs.\ the strong heuristic) that labels the pretraining
dataset $\sim$1000$\times$ faster than deep alpha-beta search.
```
Replace with:
```
A key contribution is a
cheap rollout teacher that is \textbf{stronger than our hand-crafted heuristic
baseline} (which we call \emph{Elite}): it determinizes hidden cards and finishes
each world with Elite playouts, reaching 62\% against Elite while labeling the
pretraining dataset $\sim$1000$\times$ faster than a deep alpha-beta search.
```

- [ ] **Step 2: Fix the headline result line.**

Locate (lines 60-61):
```
The champion
\textbf{beats the EliteHeuristic baseline 52.1\% $\pm$ 1.8\% ($n=3000$)} and,
```
Replace with:
```
The champion
\textbf{beats Elite on 52.1\% $\pm$ 1.8\% of deals ($n=3000$)} and,
```

- [ ] **Step 3: Add a one-sentence baseline ladder to the intro.**

After the black-box paragraph (after line 90, before the "Our thesis is different" sentence on line 90-92), insert a new paragraph:
```
We benchmark against a ladder of three non-learned opponents of increasing
strength: a uniform-random player (\emph{RandomBot}); an earlier rule-based bot
(\emph{OldHeuristic}); and a strong hand-crafted heuristic we call \emph{Elite},
which plays the cheapest card that wins the trick, cuts cheaply with trump when
void, and otherwise discards its least valuable legal card. Elite is the bar to
beat: it already defeats OldHeuristic roughly two times in three, and the same
Elite policy supplies the self-play fitness signal during training.
```
> If the Step-0 accuracy check found fitness uses a *different* bot, change the last clause to name it instead of claiming "the same Elite policy."

- [ ] **Step 4: Rebuild and eyeball the abstract.**

Run: `cd report && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: 0 errors. Read the abstract aloud; every proper noun (`Elite`, `WANN`, `PIMC`) is now defined inline.

---

## Task 2: Introduce the two blockers before the contributions reference them

**Files:** Modify `report/main.tex:109-128` (contributions), add a motivating paragraph at `report/main.tex:~108`.

Contribution 1 cites "previous best: 30.2\%" and contribution 2 the "myopic leaf evaluator," but the *collapse* and *myopic teacher* blockers are only explained in §5.3 (Results). State them where they motivate the work.

- [ ] **Step 1: Insert a "two blockers" paragraph before the contributions list.**

Immediately before line 109 (`We make four contributions:`), insert:
```
Two obstacles had kept every earlier version of this agent below the Elite bar,
and naming them up front motivates our contributions. First, the abstract intent
vocabulary \emph{collapsed} during self-play: because the three intents were each
individually weak, training drove the network toward one dominant intent while the
others atrophied, leaving the best prior agent at only 30.2\% against Elite.
Second, the natural supervised teacher, a deep alpha-beta PIMC search, itself only
\emph{tied} Elite, so imitating it could never exceed Elite. We diagnose both in
detail in Sections~\ref{sec:method} and~\ref{sec:results}; the contributions below
are the two fixes and what they revealed.
```

- [ ] **Step 2: Add the referenced labels.**

Add `\label{sec:method}` on the Method section heading line (line 252 area) and `\label{sec:results}` on the Experiments section heading (line 463 area). (`\label{sec:discussion}` already exists at line 548.)

- [ ] **Step 3: Trim contribution 1 so it no longer first-introduces the collapse.**

Locate (lines 112-115):
```
\item A neurosymbolic Sueca agent that \textbf{beats a strong heuristic
  baseline} (52.1\% $\pm$ 1.8\%, $n=3000$) while remaining interpretable ---
  the first time this project has exceeded EliteHeuristicBot (previous best:
  30.2\%).
```
Replace with:
```
\item A neurosymbolic Sueca agent that \textbf{beats Elite} (52.1\% $\pm$ 1.8\%,
  $n=3000$) while remaining interpretable: the first agent in this project to
  exceed Elite, up from the 30.2\% of the collapsed predecessor described above.
```

- [ ] **Step 4: Reframe §5.3 from cold-open to quantification.**

Rename the §5.3 heading (line 507): `The Journey: From Collapse to Victory` -> `From Prior Failure to the Final Champion`. The body (lines 509-524) already quantifies 30.2 -> 52.7 -> 52.1; ensure its first sentence back-references the intro ("As introduced in Section~1, ...") rather than presenting the collapse as news. Apply the version-name swaps from Task 4 here.

- [ ] **Step 5: Rebuild and check the reading order.**

Run: `cd report && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: 0 errors, no `undefined reference` in `main.log` (`grep -i 'undefined' report/main.log` returns nothing). Confirm the intro now reads: problem -> baselines -> blockers -> contributions.

---

## Task 3: Justify the public-information-only belief features (the Boss objection)

**Files:** Modify `report/main.tex:200-222` (§3.3 Belief State).

§3.2 claims "never leaks opponent hand data" (line 197-198); §3.3 then lists "highest *unplayed* card" features (line 211-215). Resolve the apparent contradiction explicitly.

- [ ] **Step 1: Insert a justification paragraph after the feature bullet list.**

After the `\end{itemize}` at line 219, before "The full 35-feature layout..." (line 221), insert:
```
\textbf{Why these are public.} Every feature is a deterministic function of the
player's own hand and the public play history, so none requires knowledge of an
opponent's cards. \emph{Boss} detection illustrates the point. A card is the
``boss'' of a suit when it is the highest-ranked card of that suit not yet seen in
any completed or in-progress trick. The player computes this purely from cards
everyone at the table can see, the pile of already-played cards, together with its
own hand; an opponent could hold a higher card of the suit only if such a card
were still unseen, which is exactly what holding the boss rules out. The same
reasoning covers ``can I beat the current winner?'', the cheapest winning and
sacrifice costs, and the trump- and point-remaining counts: each compares the
player's own legal cards against the publicly known set of played cards. Voids
(features 10--13) are likewise public, since failing to follow suit is observed by
all players.
```

- [ ] **Step 2: Tighten the §3.2 claim so it forward-points.**

Locate (lines 197-198):
```
hand data} --- it uses only information a human player at the table can observe.
```
Replace the em-dash and add a pointer (this also serves Task 4):
```
hand data: it uses only information a human player at the table can observe, as
Section~3.3 makes concrete for the tactical features.
```

- [ ] **Step 3: Rebuild and verify the appendix is consistent.**

Run: `cd report && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: 0 errors. Spot-check that no appendix feature description (lines 646-680) implies hidden information; `Trumps_Remaining`, `Game_Pts_Remaining`, `Any_Opp_Void_*` are all public counts, consistent with the new paragraph.

---

## Task 4: Replace internal version labels with descriptive names

**Files:** Modify `report/main.tex` wherever `v5`/`v6` appear (captions and §5.3).

- [ ] **Step 1: Enumerate occurrences.**

Run: `grep -n 'v5\|v6' report/main.tex`
Expected hits include lines ~312-313, 351, 519-533 (figure captions and the journey paragraph).

- [ ] **Step 2: Apply the standard swaps.**

- `v6 champion` -> `final (rollout-teacher) champion` (first use) then `final champion`.
- `v5 champion` -> `earlier (alpha-beta-teacher) champion` (first use) then `earlier champion`.
- `v6 dataset` -> `rollout-teacher dataset`.
- In figure captions (training_curve line 312, complexity line 529-531), use the descriptive names so a reader who jumps to a figure still understands it.

Example, complexity caption (lines 529-531):
```
\caption{Complexity comparison: v5 champion (alpha-beta teacher, 132 hidden
gates) vs.\ v6 champion (rollout teacher, 29 gates). Iso-strength, 4.5$\times$
simpler.}
```
becomes:
```
\caption{Complexity comparison: the earlier alpha-beta-teacher champion (132
hidden gates) against the final rollout-teacher champion (29 gates). Equal
playing strength, 4.5$\times$ simpler.}
```

- [ ] **Step 3: Rebuild and re-grep.**

Run: `cd report && latexmk -pdf -interaction=nonstopmode main.tex && grep -n 'v5\|v6' report/main.tex`
Expected: 0 build errors; grep returns nothing (or only an intentional, defined use).

---

## Task 5: Purge all em-dashes

**Files:** Modify `report/main.tex` (every `---` occurrence).

LaTeX `---` renders an em-dash. **Keep `--` (en-dash) in numeric ranges** (`0--4`, `61--90`, `0--149`, `[50.3,~53.9]`) — those are correct and must not be touched. Only the triple-hyphen `---` is removed.

- [ ] **Step 1: Enumerate every em-dash.**

Run: `grep -n -- '---' report/main.tex`
Expected: ~18–22 occurrences across abstract, intro, background, method, results, discussion, conclusion. Some Task 1–4 edits already removed a few.

- [ ] **Step 2: Apply the substitution policy to each, preserving meaning.**

| Pattern | Rule | Worked example |
|---|---|---|
| Parenthetical pair `X --- Y --- Z` (Y explains) | Both `---` -> commas (or parentheses) | `commitment --- logic gates, not learned weights --- is` becomes `commitment, logic gates rather than learned weights, is` |
| Trailing elaboration `clause --- detail` | `---` -> colon if detail completes the clause | `alias inlining --- the follow brain reduces` becomes `alias inlining: the follow brain reduces` |
| Joinable with a conjunction `A --- and B` / `A --- a finding that` | `---` -> comma | `not strength --- a finding that reframes` becomes `not strength, a finding that reframes` |
| Strong break / new independent statement | `---` -> period + capital, or semicolon | `human-readable rules --- the first time` becomes `human-readable rules. This is the first time` |

- [ ] **Step 3: Handle each remaining occurrence individually** using the table. Representative locations (re-confirm with the Step-1 grep, since earlier tasks shift line numbers): abstract (line 64), intro thesis (line 92), neurosymbolic-commitment pair (lines 105–107), contribution 2 (line 119), contribution 4 (line 127), background rule-extraction (line 175), styled-resolver (lines 247, 249), rollout/positional (line 350), verification (line 393), discussion (lines 559, 568, 574, 577, 588, 597), conclusion (lines 606, 609, 617, 619).

- [ ] **Step 4: Verify zero em-dashes remain, ranges intact.**

Run: `grep -n -- '---' report/main.tex`
Expected: **no output**.
Run: `grep -n -- '--' report/main.tex | grep -v -- '---'`
Expected: only numeric ranges (`0--4`, `61--90`, `0--149`, `91--119`, `[50.3, 53.9]`, generation spans). Confirm none were collapsed.

- [ ] **Step 5: Rebuild.**

Run: `cd report && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: 0 errors. Skim affected sentences for punctuation that reads naturally.

---

## Task 6 (optional polish, non-blocking): so-what figure captions

**Files:** `report/main.tex` figure captions.

Per the scientific-writing skill, a caption states the *so-what*, not the *what*. Only do this if time remains after Tasks 1–5.

- [ ] Training-curve caption (line 312): lead with the message, e.g. "Both brains learn from zero connections; Phase 1 self-play lifts game-point delta above the Elite baseline." Keep the phase/generation detail as the second sentence.
- [ ] Topology caption (line 398) and tournament caption (line 502): ensure the first clause is the message, not a label.

---

## Final Verification (run after all tasks)

- [ ] **Build clean:** `make report` (or `cd report && latexmk -pdf main.tex`) — 0 errors, 0 warnings beyond benign font/hyperref.
- [ ] **No undefined refs/citations:** `grep -i 'undefined\|??' report/main.log` returns nothing; `grep -c 'Citation' report/main.log` shows no "undefined citation".
- [ ] **No em-dashes:** `grep -n -- '---' report/main.tex` returns nothing; ranges preserved.
- [ ] **No stray internal labels:** `grep -n 'v5\|v6\|EliteHeuristicBot\|HeuristicBot\|OldHeuristicBot' report/main.tex` — prose uses only the standardized names; remaining hits are intentional (e.g., one code-identifier gloss in `\texttt{}`).
- [ ] **Page budget:** `pdfinfo report/main.pdf | grep Pages` — body (text + figures + refs, excluding the clearly-marked optional appendix) stays within 10–12 pages. If it overflowed, trim the now-partly-redundant §5.3 prose (the blocker is stated in the intro, so §5.3 only needs the numbers).
- [ ] **Cold-reader pass:** read abstract + introduction once as if you'd never seen the project. Confirm: Elite, OldHeuristic, RandomBot, collapse, and the two blockers are all defined before they are relied upon; no "v5/v6"; Boss/public-info objection answered in §3.3.
- [ ] **Commit:** `git add report/main.tex report/main.pdf && git commit` with a message describing the v2 readability revision.

## Out of scope (do not do here)

- No new experiments, numbers, or figures (numbers are frozen and verified).
- No restructuring of section order beyond the §5.3 reframing and the intro blocker paragraph.
- No author/affiliation changes (already set, line 39-41).
