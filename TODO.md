# Sueca WANN — TODO

## Phase 1: Environment & Feature Engineering

- [x] `src/engine/cards.py` — Card/Suit/Rank enums, deck building, point values, dealing
- [x] `tests/test_cards.py` — 31 tests: deck composition, points sum to 120, rank ordering, deal correctness
- [x] `src/engine/sueca_engine.py` — 4-player game state machine, trick resolution, void tracking
- [x] `tests/test_engine.py` — 28 tests: follow-suit, trick winner, full game invariant (120 pts), game-point tiers
- [x] `src/engine/belief_state.py` — 18-node belief vector encoder
- [x] `tests/test_belief_state.py` — 17 tests: all 18 fields, [0,1] bounds, edge cases
- [ ] `src/engine/duplicate_loop.py` — Duplicate deal generator + symmetric seat-rotation evaluator

## Phase 2: Custom WANN Framework

- [x] `src/wann/logical_nodes.py` — Aggregation (SUM/MIN/MAX) + Activation (IDENTITY/NOT/THRESHOLD) lookup tables
- [x] `tests/test_logical_nodes.py` — 58 tests: each function, boundary values, compound gates, sign inversion
- [ ] `src/wann/genome.py` — Gene arrays (conn[5,N], node[4,M]), express() via topological sort, zero-link init
- [ ] `src/wann/network.py` — Forward pass: sign inversion → shared weight scaling → aggregation → activation → clamp [0,1]
- [ ] `tests/test_network.py` — Minimal hand-built topologies, zero-link output, sign inversion, W sweep
- [ ] `src/wann/mutations.py` — Add node, add connection, toggle, flip sign, change activation/aggregation
- [ ] `src/wann/species.py` — Compatibility distance, speciation, stagnation removal
- [ ] `src/wann/population.py` — NEAT ask/tell loop, tournament selection, elitism, multi-objective ranking

## Phase 3: Oracle & Evolution

- [ ] `src/oracle/legal_system.py` — 5-intent resolver (DUCK_OR_DUMP, TAKE_CHEAPLY, FORCE_HIGH, FEED_PARTNER, CUT_LOW)
- [ ] `tests/test_legal_system.py` — Each intent, illegal fallback scenarios, Oracle Tax triggers
- [ ] `src/oracle/fitness.py` — Evaluate genome: duplicate deals × seat rotations × W sweep, Oracle Tax warm-up
- [ ] `src/oracle/hall_of_fame.py` — Frozen champion archive, .npz serialization, opponent sampling
- [ ] `src/baselines/random_bot.py` — Uniform random legal card (floor baseline)
- [ ] `src/baselines/heuristic_bot.py` — Hard-coded rules: follow high, trump when void, lead aces (curriculum partner)
- [ ] `src/train.py` — Main evolution loop: curriculum training → mixed → self-play, checkpointing, CSV stats

## Phase 4: Compilation & Benchmarking

- [ ] `src/export/export_flowchart.py` — Backward-chain rule compiler + Graphviz topology export
- [ ] `src/baselines/pimc_bot.py` — Perfect Information Monte Carlo solver (ceiling baseline)
- [ ] `src/benchmark.py` — Round-robin tournament runner, win rate tracking, binomial CIs
- [ ] `configs/default.toml` — All hyperparameters in one place
