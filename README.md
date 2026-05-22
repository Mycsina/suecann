# 🃏 Neurosymbolic WANNs for Sueca

A neurosymbolic framework that evolves **Weight-Agnostic Neural Networks (WANNs)** to master **Sueca** (a Portuguese four-player partnership trick-taking card game). Rather than fitting continuous weights to specific tasks, this system focuses on evolving discrete network topologies (using logical gates and custom aggregations) that produce abstract strategic intents. The resulting networks can be compiled directly into human-readable decision rules (IF/THEN trees) for complete transparency.

---

## 📐 Architecture Overview

The system operates via a decoupled pipeline to ensure strict adherence to game rules while allowing maximum symbolic flexibility:

```
Belief State (21 features in [0,1])
             │
             ▼
Weight-Agnostic Neural Network (Logical Gates)
             │
             ▼
Oracle Intents (5 abstract play intents)
             │
             ▼
Legal Subsystem (Filters invalid plays & chooses best card)
             │
             ▼
Selected Card Play
```

### 🧠 1. Belief State (21 Inputs)
The input vector represents the agent's current belief state of the game, normalized to `[0,1]`:
- **`Has_Led_Suit`** / **`Has_Trump`** (Binary): Hand status indicators.
- **`Led_Suit_Power`** / **`Trump_Power`** (Float): Relative strength of top cards.
- **`Hand_Point_Density`** (Float): High-value points remaining in the hand.
- **`Am_I_Leading`** / **`Am_I_Last_To_Play`** (Binary): Trick position context.
- **`Is_Partner_Winning`** (Binary): Partnership status.
- **`Trick_Point_Value`** (Float): Point density in the current trick.
- **`Has_Trick_Been_Cut`** (Binary): True if trump was played on an off-suit trick.
- **`Partner_Void_Led`** / **`Partner_Void_Trump`** (Binary): Partnership void tracking.
- **`Any_Opp_Void_Led`** / **`Any_Opp_Void_Trump`** (Binary): Opponent void tracking.
- **`Led_Suit_Ace_Played`** / **`Led_Suit_7_Played`** (Binary): Card counting (led suit).
- **`Trump_Ace_Played`** (Binary): Card counting (trumps).
- **`Game_Pts_Remaining`** (Float): Unplayed card points in the deal.
- **`Trick_Number`** (Float): Index of the current trick.
- **`Trumps_Remaining`** (Float): Count of unplayed trumps.
- **`Score_Delta`** (Float): Normalized score differential.

### 🎯 2. Oracle Intents (5 Outputs)
The WANN outputs a real value for each of the 5 play intents. The highest-valued output (tied maximums are resolved randomly via `rng.choice` to prevent bias) determines the selected intent:
1. **`DUCK_OR_DUMP`**: Play the lowest legal card.
2. **`TAKE_CHEAPLY`**: Play the lowest card that beats the current trick winner.
3. **`FORCE_HIGH`**: Play the highest-power card.
4. **`FEED_PARTNER`**: Play the highest point-value card.
5. **`CUT_LOW`**: Play the lowest trump (when void in led suit).

*Note: If an intent is illegal (e.g. attempting to cut when holding led suit), the legal subsystem overrides it to `DUCK_OR_DUMP` and levies a fitness penalty (Oracle Tax).*

### 🛠️ 3. Weight-Agnostic Topologies
The networks utilize discrete configuration features:
- **Sign-Only Connections**: Connections carry a sign ($+1$ or $-1$) instead of continuous weights. A shared weight $W$ is swept over $W \in \{-2.0, -1.0, -0.5, 0.5, 1.0, 2.0\}$ to score true weight-agnostic fitness.
- **Logical Nodes**:
  - **Aggregations**: `SUM` ($0$), `MIN/AND` ($1$), and `MAX/OR` ($2$).
  - **Activations**: `IDENTITY` ($0$), `NOT` ($1$), and `THRESHOLD` ($2$).

---

## ⚡ Rust Acceleration FFI

Simulation and population evaluations are offloaded to Rust (`src/sueca_solver`) via PyO3, which releases the GIL and distributes game rollouts in parallel using Rayon. This provides a **~500x speedup** over pure Python.

### Compilation Command:
Compile the Rust solver and place the resulting shared object `.so` file in the python virtual environment site-packages:
```bash
cargo build --manifest-path src/sueca_solver/Cargo.toml --release && \
cp src/sueca_solver/target/release/libsueca_solver.so \
.venv/lib/python3.13/site-packages/sueca_solver/sueca_solver.cpython-313-x86_64-linux-gnu.so
```

---

## 📈 Evolutionary Curriculum

The framework employs a curriculum-driven approach divided into two primary phases:

```
Phase 0: Classification Accuracy (Gens 0-100)
 ├── 🏋️ Bulking Phase (Gens 0-50): Disable Pareto. Aggressive bloat.
 └── ✂️ Cutting Phase (Gens 50-100): Enable Pareto. Prune topologies.
               │
               ▼
Phase 1+: Rollout-based RL (Self-Play against HOF & Heuristic Anchor)
```

### Phase 0: Supervised Warmup (Generations 0 to 100)
Aligns network intents with a pre-recorded dataset of expert decisions.
1. **The Bulking Phase (Gen 0 to 50)**:
   The **Pareto simplicity penalty is completely disabled**, and selection is done strictly by classification accuracy. This encourages networks to bloat aggressively, throwing out complex, redundant connections that discover advanced logic gates and high-accuracy intersections.
2. **The Cutting Phase (Gen 50 to 100)**:
   The **Pareto simplicity penalty is reactivated** (using non-dominated Pareto ranking). The algorithm acts like a sculptor, pruning away useless connections and redundant hidden layers, keeping only the elegant, multi-input logic pathways that successfully boost accuracy.

### Phase 1+: Self-Play Reinforcement Learning (Generation 100+)
Replaces supervised training with game rollouts. Networks play duplicate matches against a mixture of Hall of Fame champions and calibrated heuristic anchors (`HeuristicBot`).

---

## 🚀 Getting Started

### 📦 Prerequisites & Installation
Ensure you have Python 3.13, a Rust toolchain (`cargo`), and `uv` installed:

```bash
# Clone the repository
cd project

# Install Python packages using uv
uv sync
```

### 🧪 Running Tests
Execute the test suite to ensure both Python logic and the Rust FFI module behave correctly:
```bash
uv run pytest
```

### 🏋️ Running Training
Start the evolutionary curriculum run with:
```bash
uv run python -m src.train --config configs/default.toml
```

You can customize hyperparameters in [configs/default.toml](file:///home/mycsina/Projects/Uni/CAA/project/configs/default.toml), such as population size, phase lengths, and mutation rates.

---

## 📊 Benchmarking & Analysis

### 🏁 Round-Robin Tournament
Run a round-robin tournament between `RandomBot`, `HeuristicBot`, `PIMCBot`, and your WANN Champion:
```bash
uv run python -m src.benchmark \
  --deals 200 \
  --genome checkpoints/best_genome_final.npz \
  --output-dir checkpoints/results
```

This generates:
1. **ASCII results table** printed directly to the terminal.
2. **`tournament_report.csv`** containing exact win rates, average card points, and confidence intervals.
3. **`tournament_matrix.png`**: A beautiful, green-to-red performance heatmap showing 95% binomial confidence intervals.

---

## 🔍 Checkpoint & Dump Analysis

Checkpoints are saved automatically under the `checkpoints/` directory as compressed NumPy files (`.npz`).

### 📦 Checkpoint File Structure
Inside a `.npz` genome checkpoint, the following variables are stored:
- `next_innovation`: The next innovation ID.
- `node_ids` / `node_types` / `node_acts` / `node_aggs`: Node descriptions.
- `conn_innovs` / `conn_srcs` / `conn_dsts` / `conn_signs` / `conn_enabled`: Connection mapping.

### 🐍 Programmatic Inspection
You can easily load and inspect a saved genome using Python:

```python
from src.train import load_genome

# Load the genome
genome = load_genome("checkpoints/best_genome_final.npz")

# Print enabled connections
print("--- Enabled Connections ---")
for conn in genome.conn_genes.values():
    if conn.enabled:
        print(f"Inno {conn.innovation}: Node {conn.src} ──[{'+' if conn.sign > 0 else '-'}]──> Node {conn.dst}")
```

### 📊 Training Statistics
The training statistics are written iteratively to `training_stats.csv`. This file logs:
- `generation`: The generation index.
- `phase`: The curriculum phase (0 = Supervised, 1 = RL).
- `best_fitness` / `avg_fitness` / `median_fitness`: Population fitness trends.
- `n_species`: Species counts in the population.
- `n_connections_best` / `n_hidden_best`: Node and link complexities of the top genome.
- `oracle_tax`: The active illegal intent penalty factor.
