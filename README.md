# Restaurant Layout Optimizer

An agent-based restaurant simulator combined with a machine learning pipeline that finds optimal table layouts by minimizing waiter travel distance.

Built entirely from scratch — no frameworks, no library dependencies for the core simulation.

---

## What it does

The simulator runs a full restaurant floor in the browser: guests arrive, get seated, wait for drinks, and leave. Waiters navigate the floor using A* pathfinding to serve orders.

The ML pipeline collects thousands of these simulation runs, trains a surrogate model on the layout data, and uses that model to search for better table arrangements — without running the expensive full simulation for every candidate.

**Result: 31% reduction in waiter travel distance** (367k → 253k pixels), found through surrogate-guided optimization and confirmed via headless browser validation.

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
merge_datasets.py ──► restaurant-sim-merged.json (10,715 runs)
        │
        ▼
train_surrogate.py  ──► surrogate_model.pkl  (XGBoost, R²=0.678)
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

evaluate.py        ──► reports the four headline metrics at any point
```

### Surrogate Model

- **Algorithm**: XGBoost (best of RF / XGBoost / GradientBoosting comparison)
- **Features**: 71 hand-crafted features — raw table positions, per-table distances to bar, corridor width estimates, cluster compactness
- **Target**: `log(waiterDist)`, back-transformed to pixels on predict. Waiter distance spans 253k–1.3M px, so log space keeps the errors evenly weighted across that range
- **Deduplication**: the 10,715 raw runs collapse to 4,272 unique layouts. `validate_headless.js` writes one record *per seed*, each carrying the batch's full seed list, so the same layout legitimately appears many times. Merging them into one row with a per-simulation weight is what keeps identical layouts out of train and test at the same time
- **Validation**: `GroupKFold` on the layout key. After deduplication every group holds exactly one row, so this is a safety net rather than a fix — the deduplication is what removes the leak
- **Performance**: R² = 0.678, MAE = 60,787 px out-of-fold on 4,272 unique layouts

### Why R² is not the headline metric

Global R² is dominated by the gap between disastrous and mediocre layouts, which is not what the optimizer needs. What matters is ranking *good* layouts against each other. `evaluate.py` reports four numbers instead:

| Metric | Meaning | Current |
|---|---|---|
| Validated top-1 | Best layout after real headless validation | 254,012 px (n=1) |
| Calibration error | Mean (predicted − actual) on validated candidates | −0.1% (n=1) |
| Spearman ρ, best decile | Ranking quality among the top 10% of layouts | 0.328 |
| R² out-of-fold | Deduplicated, weighted per simulation, GroupKFold | 0.678 |

The gap between ρ = 0.85 overall and ρ = 0.33 within the best decile is the single most useful diagnostic in the pipeline: the model separates bad from good easily, but barely ranks the good ones. Simulator noise is not the limit — measured against the per-seed spread in the dataset, the ceiling sits near R² ≈ 0.99.

A second systematic effect matters more than the log transform: prediction bias runs from **+10% in the best decile to −21% in the worst**, plain regression to the mean. That is what makes the surrogate unreliable exactly where the optimizer searches.

The first two metrics currently rest on a single validated layout, so treat them as placeholders until a proper validation round lands.

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
| Waiter travel distance | ~367,532 px | ~253,102 px |
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
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd ml && npm install   # installs Playwright for headless validation
```

PyTorch is only needed for the GNN (`train_gnn.py`, `gnn_layout.py`) and is deliberately left out of `requirements.txt` — see the comment there for the install command.

Data files (`restaurant-sim-*.json`, `surrogate_model.pkl`, `gnn_model.pt`, results JSON) live in the **repository root**; the scripts live in `ml/` and resolve paths relative to it.

**Full active learning cycle:**

```bash
cd ml

# 1. Export training data from the browser (use the "Export batch" button),
#    place the downloaded JSON files in the repository root

# 2. Merge batch files
python3 merge_datasets.py

# 3. Train surrogate model
python3 train_surrogate.py       # XGBoost/RF/GBT comparison, saves the best
python3 evaluate.py              # report the four headline metrics
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
bash ml/start_overnight.sh
```

---

## Project structure

```
simulatie.html              # Full browser simulator (single file, no build step)
best-layout.json            # Best layout found (253k px, confirmed)
requirements.txt            # Python dependencies for the core pipeline
ml/
├── optimize_layout.py      # Surrogate-guided layout search (500k candidates)
├── train_surrogate.py      # RF/XGBoost/GBT comparison, saves best model
├── log_target.py           # Log-space wrapper around the regressor
├── evaluate.py             # Reports the four headline metrics
├── train_gnn.py            # GATv2 GNN trainer (GPU-accelerated)
├── gnn_layout.py           # GNN architecture + graph builder + gradient optimizer
├── validate_headless.js    # Playwright headless validator
├── active_learning.py      # Merge → retrain → optimize loop
├── merge_datasets.py       # Combines batch export files
├── collect_overnight.py    # Unattended overnight data collection
├── start_overnight.sh      # Shell wrapper for overnight runs
└── package.json            # Playwright dependency
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
