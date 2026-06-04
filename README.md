# Restaurant Layout Optimizer

An agent-based restaurant simulator combined with a machine learning pipeline that finds optimal table layouts by minimizing waiter travel distance.

Built entirely from scratch — no frameworks, no library dependencies for the core simulation.

---

## What it does

The simulator runs a full restaurant floor in the browser: guests arrive, get seated, wait for drinks, and leave. Waiters navigate the floor using A* pathfinding to serve orders.

The ML pipeline collects thousands of these simulation runs, trains a surrogate model on the layout data, and uses that model to search for better table arrangements — without running the expensive full simulation for every candidate.

**Result: 31% reduction in waiter travel distance** (367k → 252k pixels), found through surrogate-guided optimization and confirmed via headless browser validation.

---

## Demo

Open `simulatie.html` directly in a browser — no build step, no server needed.

!<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/38bdbaa5-1150-4d52-80d3-f4cd0067026c" />


Use the sidebar controls to adjust guest count, waiter count, table mix, and simulation speed. The efficiency score updates live.

---

## Architecture

### Browser Simulator (`simulatie.html`)

A single-file Canvas 2D simulation with five core classes:

| Class | Role |
|---|---|
| `PathNavigator` | A\* pathfinding on an 18px cell grid, rasterizes furniture as obstacles |
| `Table` | Furniture unit with chair anchors, seating state, and order tracking |
| `Guest` | State-machine agent: `WALKING → SEATED → WAITING → DRINKING → LEAVING` |
| `Waiter` | Task-driven agent, carries up to 8 drinks, plans multi-table serving trips |
| `SimulationEngine` | World owner: agents, metrics, layout optimizer, render loop |

The built-in layout optimizer (`generateLayout`) scores 80 random candidate positions per table and picks the best non-colliding one. This greedy approach seeds the ML pipeline with diverse starting layouts.

**Efficiency score formula:**
```
score = (servedDrinks × 12) − (avgWait × 3) − (impatientGuests × 25) − (waiterDistance × 0.02)
```

### ML Pipeline

```
Browser sim (batch export)
        │
        ▼
merge_datasets.py ──► restaurant-sim-merged.json (2,045 layouts)
        │
        ▼
train_surrogate.py  ──► surrogate_model.pkl  (RandomForest, R²=0.623)
train_gnn.py        ──► gnn_model.pt         (GATv2, R²≈0.35–0.40)
        │
        ▼
optimize_layout.py ──► optimizer-results.json  (top candidate layouts)
        │
        ▼
validate_headless.js ──► validation-results.json  (real scores via Playwright)
        │
        ▼
active_learning.py ──► merge + retrain + re-optimize (loop)
```

### Surrogate Model

- **Algorithm**: RandomForest (best of RF / XGBoost / GradientBoosting comparison)
- **Features**: 71 hand-crafted features — raw table positions, per-table distances to bar, corridor width estimates, cluster compactness
- **Target**: `waiterDist` (total waiter travel distance per run, averaged over multiple seeds)
- **Performance**: R² = 0.623, MAE = 85,608 px on held-out data

### GNN Model

Each layout is encoded as a graph where tables are nodes and edges connect nearby tables. Node features include position, size, distance to bar, and distance to entrance.

- **Architecture**: 3× GATv2Conv layers, hidden=64, 4 attention heads
- **Training**: AdamW + CosineAnnealingLR + Huber loss + early stopping + ±15px jitter augmentation
- **Current status**: R² ≈ 0.35–0.40 (gradient search activates automatically at R² > 0.55)

### Headless Validator

Candidate layouts from the optimizer are validated by actually running the simulator in a headless Chromium browser via Playwright. This closes the simulation–surrogate gap and generates labeled data for the next active learning iteration.

---

## Results

| Metric | Baseline (greedy) | Optimized |
|---|---|---|
| Waiter travel distance | ~367,532 px | ~252,102 px |
| Improvement | — | **31.4%** |
| Efficiency score | ~−3,800 | ~−3,007 |

**Best layout pattern:** All 8 variable tables rotated 90°, concentrated toward the right side of the room (near the bar dock). 6 of 8 tables placed at x > 300, minimizing waiter round-trip distance to the bar.

---

## Getting started

### Run the simulator

```bash
# Just open the file — no server required
open simulatie.html         # macOS
xdg-open simulatie.html     # Linux
```

### Run the ML pipeline

**Prerequisites:**
```bash
pip install numpy scikit-learn xgboost joblib torch torch-geometric
npm install   # installs Playwright for headless validation
```

**Full active learning cycle:**

```bash
# 1. Export training data from the browser (use the "Export batch" button)

# 2. Merge batch files
python3 merge_datasets.py

# 3. Train surrogate model
python3 train_surrogate.py       # RandomForest baseline
python3 train_gnn.py             # GATv2 GNN (requires PyTorch + PyG)

# 4. Generate optimized layouts
python3 optimize_layout.py --candidates 500000

# 5. Validate in headless browser (writes validation-results.json)
node validate_headless.js --top 10 --seeds 9

# 6. Feed results back into training data + retrain
python3 active_learning.py --validated validation-results.json
```

**Or run a full overnight optimization cycle:**
```bash
bash start_overnight.sh
```

---

## Project structure

```
simulatie.html          # Full browser simulator (single file, no build step)
optimize_layout.py      # Surrogate-guided layout search (500k candidates)
train_surrogate.py      # RF/XGBoost/GBT comparison, saves best model
train_gnn.py            # GATv2 GNN trainer (GPU-accelerated)
gnn_layout.py           # GNN architecture + graph builder + gradient optimizer
validate_headless.js    # Playwright headless validator
active_learning.py      # Merge → retrain → optimize loop
merge_datasets.py       # Combines batch export files
collect_overnight.py    # Unattended overnight data collection
start_overnight.sh      # Shell wrapper for overnight runs
best-layout.json        # Best layout found (252k px, confirmed)
```

---

## Tech stack

- **Simulation**: Vanilla JS, Canvas 2D, A\* pathfinding — zero runtime dependencies
- **ML**: PyTorch, PyTorch Geometric (GATv2), scikit-learn, XGBoost, NumPy
- **Validation**: Node.js, Playwright (headless Chromium)

---

## Background

This project started as an experiment in agent-based simulation and grew into a full ML optimization loop. The core question: *can a machine learning model learn what makes a restaurant layout efficient, without running thousands of full simulations for every candidate?*

The answer is yes — but the surrogate model needs enough diversity in the training data to generalize. The active learning loop addresses this by iteratively collecting validated layouts from the regions of the search space the model is most uncertain about.

The GNN approach models the layout as a relational structure (tables interact with each other and with the bar), which should capture spatial dependencies that flat feature vectors miss. It needs more training data to outperform the Random Forest baseline.
