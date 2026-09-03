# Restaurant Layout Optimizer

An agent-based restaurant simulator combined with a machine learning pipeline that finds optimal table layouts by minimizing waiter travel distance.

Built entirely from scratch — no frameworks, no library dependencies for the core simulation.

---

## What it does

The simulator runs a full restaurant floor in the browser: guests arrive, get seated, wait for drinks, and leave. Waiters navigate the floor using A* pathfinding to serve orders.

The ML pipeline collects thousands of these simulation runs, trains a surrogate model on the layout data, and uses that model to search for better table arrangements — without running the expensive full simulation for every candidate.

**Result: 12.7% reduction in waiter travel distance against the best of 10,240 randomly sampled
layouts** — 320,577 px → 279,828 px, each measured over 30 seeds in a headless browser
(95% CI on the difference: 11.0%–14.4%).

Both sides are measured identically, and the winning layout was re-validated on 15 fresh seeds
after selection: it comes out 2,004 px higher there, and that selection effect is already absorbed
into the pooled figure above.

> **This replaces an earlier claim of "31% reduction, 367k → 253k px".** That number came from a
> layout that walled in the bar: every table was unreachable, and the optimizer was exploiting a
> pathfinding fallback rather than finding a floor plan — see
> [Reward hacking](#reward-hacking-the-optimizer-found-a-bug-not-a-layout). The simulator is fixed,
> the dataset has been re-collected, and the figure above comes from the new data.
>
> Note the change of baseline. The old comparison ran against a mediocre greedy layout, which
> flatters the result. The honest question is whether the optimizer beats *blind sampling*, so the
> baseline above is the best layout out of 10,240 randomly generated valid ones.

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
merge_shards.py    ──► restaurant-sim-clean.json (30,720 runs)
        │
        ▼
train_surrogate.py  ──► surrogate_model.pkl  (two models: 95-feature base, R²=0.990
                                                 109-feature frontier, R²=0.991)
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

- **Algorithm**: GradientBoosting (best of RF / XGBoost / GradientBoosting comparison)
- **Features**: 95 features — 71 hand-crafted (raw table positions, per-table Euclidean distances
  to bar, corridor width estimates, cluster compactness) plus 24 A\* path features from
  `pathgrid.py`. The Euclidean features cannot see that a table blocks a corridor; the path
  features are a Dijkstra sweep from the bar dock over the simulator's own walking grid. They are
  what the model actually runs on. The strongest individual features are the path distances to the
  furthest-but-one tables (`sorted_paths[5]` and `[6]`, 0.54 combined) — a floor plan is priced by
  its worst-placed tables. Next is the `seats × path distance` work proxy (0.30 combined across
  `work.mean` and `work.sum`), a direct physical estimate of total waiter travel: waiters make
  round trips per order, and orders scale with seats
- **Target**: `log(waiterDist)`, back-transformed to pixels on predict. Waiter distance spans 316k–1.25M px, so log space keeps the errors evenly weighted across that range
- **Deduplication**: the 30,720 raw runs collapse to 10,240 unique layouts, each run with 3 seeds. `validate_headless.js` writes one record *per seed*, each carrying the batch's full seed list, so the same layout legitimately appears many times. Merging them into one row with a per-simulation weight is what keeps identical layouts out of train and test at the same time
- **Validation**: `GroupKFold` on the layout key. After deduplication every group holds exactly one row, so this is a safety net rather than a fix — the deduplication is what removes the leak
- **Performance**: R² = 0.990, MAE = 12,896 px out-of-fold on 10,240 unique layouts. A second
  model adds the 14 tour features (109 total) and reaches R² = 0.991, MAE = 12,051 px; the
  optimizer uses the cheap one to search and the frontier one to refine

### Why R² is not the headline metric

Global R² is dominated by the gap between disastrous and mediocre layouts, which is not what the optimizer needs. What matters is ranking *good* layouts against each other. `evaluate.py` reports four numbers instead:

| Metric | Meaning | Current |
|---|---|---|
| Validated top-1 | Best layout after real headless validation | 278,826 px (n=8, 15 seeds) |
| Calibration error | Mean (predicted − actual) on validated candidates | −2.7% (n=8) |
| Spearman ρ, best decile | Ranking quality among the top 10% of layouts | 0.513 |
| R² out-of-fold | Deduplicated, weighted per simulation, GroupKFold | 0.990 |

The gap between ρ = 0.96 overall and ρ = 0.51 within the best decile is the single most useful
diagnostic in the pipeline: the model separates bad from good easily, but ranks the good ones
poorly. Ranking inside the frontier is the largest piece of value still on the table.

**Simulator noise is not constant across the range**, and missing that is easy to do. Per-seed
noise runs from sd ≈ 13,200 px among the best layouts to sd ≈ 31,500 px among the worst, against a
global figure of 18,300 px. Comparing a spread measured *inside the best decile* against the
*global* noise figure is not a like-for-like comparison, and it manufactures a null result — an
earlier revision of this section did exactly that and concluded the decile was flat. It is not:

| | sd |
|---|---|
| observed spread inside the best decile | 10,144 px |
| noise on a 3-seed mean, measured locally | 7,621 px |
| → true spread between layouts (naive) | 6,694 px |
| → true spread, split-half (decile picked on seed *j*, measured on the other two) | **14,390 px** |

The split-half figure is the trustworthy one: selecting on one seed and measuring on the others
makes the selection independent of the measurement, which the naive decomposition is not. The
nearest-neighbour check flips with the corrected noise too — near-identical layouts differ by
slightly *more* than noise predicts (ratio 1.08, not 0.78).

**What better ranking would be worth: 6.8%.** Inside the decile the model itself selects, pick the
8 best by model prediction versus the 8 best by an oracle that ranks on two seeds — then score both
on the held-out third seed, so the oracle cannot cherry-pick favourable noise:

```
evaluation seed 0:  model 370,217 px   oracle 341,288 px
evaluation seed 1:  model 361,077 px   oracle 342,378 px
evaluation seed 2:  model 363,506 px   oracle 336,808 px
                    mean gap 24,776 px  =  6.8%
```

That is roughly the size of everything the surrogate has bought over random search so far (~8%).
Frontier ranking is an open problem worth real money here, not a solved or dead one.

The target of ρ(top decile) > 0.70 is still a poor instrument, but for a narrower reason than
"nothing to rank": it selects the decile on the noisy 3-seed mean and then scores against that same
noisy quantity, so it understates the model. Judge frontier ranking with a held-out seed, as above.

**One idea worked, four did not.**

What worked was asking what the waiter actually does. It carries up to 8 drinks and chains several
tables into one trip (`simulatie.html:2061`), picking which tables to chain by **Euclidean**
distance between them — and then walking **A\*** paths. Two tables that sit close together with an
obstacle between them therefore land in the same trip and cost a detour, and nothing in the feature
set could see that: the features measured distance to the bar and Euclidean distances between
tables, never the cost of the chain itself.

Fourteen tour features fix that — table-to-table *path* distances, a greedy tour cost from the bar,
and mismatch ratios (path ÷ Euclidean per pair) that measure exactly where the simulator's chaining
heuristic trips over its own geometry. Judged against a held-out seed, ranking inside the
model-selected decile rises from **ρ = 0.392 to 0.427**, positive on all three seeds
(+0.053, +0.023, +0.030), and out-of-fold R² from 0.9897 to 0.9909.

They cost 35 ms per layout against 4 ms, far too slow for a 200,000-candidate sweep, so the
optimizer runs two stages: the cheap 95-feature model for the broad search, the 109-feature model
for refinement and final ranking — which is where these features help anyway.

**The four that failed** — the opportunity is real, these particular routes to it are not:

| Attempt | ρ(top decile) |
|---|---|
| baseline (μ only) | 0.51 |
| LCB, bootstrap-ensemble σ | 0.57 |
| LCB, random-forest σ | 0.58 |
| LCB, k-NN-distance σ | 0.54 |
| specialist trained on best 25% | 0.50 |
| deeper trees / 3× more trees | 0.50 / 0.52 |

(The LCB rows are measured on a 3-fold split whose μ-only baseline is 0.59, so all three made the
ranking *worse*, and all three made the selection worse too: picking 8 layouts by μ + κσ gave a
mean actual of 349k–360k against 345k for μ alone.) No σ estimate correlated with the actual error
above ρ ≈ 0.12, which is why no value of κ can help: the quantity being weighted carries almost no
information about where the model is actually wrong. Uncertainty-guided search is not refuted in
general — these three estimators are.

The specialist row carries a similar caveat: restricting training to the best 25% also cuts the
training set from 10,240 rows to 2,048, so the experiment shows that *specialising on the data
already collected* does not pay, not that a specialist model cannot work. Collecting densely inside
the frontier and retraining there is untested.

**Label noise is not the constraint either.** A model trained on **1-seed** targets ranks as well
as one trained on 3-seed targets (0.502 vs 0.508) — the learner averages label noise away across
10,240 layouts. More seeds therefore improves how well a result can be *measured*, not how well
the model ranks. Collecting more *layouts* does nothing either: from 1,000 to 10,240 layouts,
R² climbs 0.85 → 0.95 but ρ in the best decile shows no trend at all.

For reference, the noise ceiling on global R² is 0.9966 for a 3-seed mean, so R² = 0.990 still
leaves headroom on the global metric.

Prediction bias used to be the second systematic effect, running from **+10% in the best decile to
−21% in the worst** — plain regression to the mean, which made the surrogate unreliable exactly
where the optimizer searches. On the clean dataset with path features it is largely gone: the bias
now runs from +2.5% in the best decile to −1.1% in the worst.

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
| Waiter travel distance | 320,577 px | **279,828 px** |
| Improvement | — | **12.7%** (95% CI 11.0–14.4%) |
| Standard error | ±1,733 px | ±2,101 px |

Both figures pool 30 seeds per layout from headless validation. The baseline is the best of the
10,240 randomly sampled valid layouts in `restaurant-sim-clean.json`, re-validated at the same
seed count so the two sides are measured identically.

**Winner's curse is measured, not assumed.** The layout was picked as the best of 8 on one set of
15 seeds, then re-validated on 15 fresh ones: 278,826 px on the selection seeds, 280,830 px on the
fresh ones. A 2,004 px selection effect, small, and the headline pools all 30 seeds anyway.

**What moved the needle was the candidate pool, not luck.** Earlier runs had a real problem: the
*average* of the 8 candidates was no better than the baseline group (326,020 px, p = 0.41), so the
win rested on one lucky layout while the model ranked it fourth. Adding the tour features and
refining against them changed exactly that — the group average dropped to **307,262 px**. The pool
itself got better, which is what a working surrogate is supposed to do.

The model still cannot fine-rank its own output (ρ = 0.02 across the 8), so the last step is still
brute force: validate a handful and keep the best. The 6.8% oracle gap below is the standing price
of that.

One caveat worth stating plainly: the *average* of the 8 optimizer candidates (326,020 px) is not
significantly better than the baseline group (335,863 px, p = 0.41). The win comes from one
genuinely excellent layout in the pool, not from the model reliably steering toward good ones.
Validation is doing the selecting, and that is exactly the weakness described above.

---

## Reward hacking: the optimizer found a bug, not a layout

`PathNavigator.findPath` used to return a straight line to the destination when no route
existed. A straight line is shorter than any real path, so walling in the bar dock scored
*better* than an efficient floor plan — and the optimizer found that out. Every one of the
top 50 layouts by waiter distance encloses the dock.

Three things had to change, because closing the fallback naively makes the exploit worse
rather than better — a waiter with no route walks 0 px, which is cheaper still:

1. **A failed search returns a partial route**, not a straight line, and sets `path.failed`.
   The agent stays put instead of clipping through furniture.
2. **A validity check marks unusable layouts invalid** instead of scoring them cheap. It is a
   connected-component analysis over the walkable grid: all waiters must share one floor, the
   bar approach point must lie in it, and every table must have a service point in it. A
   blocked cell is never a bridge — an agent can step *off* furniture but not *through* it,
   and treating the dock cell as a through-route was what made an enclosed bar look reachable.
3. **The pipeline filters on both signals.** `layoutValid` is the static check; the run also
   reports `waiterPathFailures`, which counts routes that actually failed during the
   simulation. `merge_shards.py` drops a run that fails either test.

On the old dataset, 65% of layouts are invalid under this check — including all of the top 50.
The cheapest *valid* layout runs 333,694 px against the 253,102 px of the best exploited one,
which is why the whole dataset was collected again. The re-collected set
(`restaurant-sim-clean.json`) has 0 invalid layouts out of 10,240, because `collect_parallel.py`
now sieves candidates through the same check before simulating them.

The exploit had a second home. `optimize_layout.py` scored candidates without any reachability
check, and the surrogate is trained only on valid layouts — so outside that region it extrapolates
freely, and that is precisely where an optimizer goes looking. Four of its first five proposals
walled in the bar again (one reachable cell out of 814). `_score_layouts()` now gates every
candidate through `pathgrid.layout_valid()` before scoring, in both the random-search and the
refinement phase.

Two regression tests guard this:

```bash
node ml/test_reachability.js --sample 250   # geldig => geen enkele mislukte oberroute
python3 ml/test_pathgrid.py                 # Python-spiegel == simulator, layout voor layout
```

`test_reachability.js` asserts the property that matters: a layout the simulator calls valid
must not produce a single failed waiter route. Guest routes may still fail (a chair behind a
wall) — those are reported separately and do not affect waiter distance.

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

# 2. Merge the collected shards into restaurant-sim-clean.json
python3 merge_shards.py
#    (merge_datasets.py is the older path; it writes restaurant-sim-merged.json,
#     which is the pre-fix dataset and is no longer what training reads)

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
best-layout.json            # Best layout found (279,828 px, 30-seed validated)
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
