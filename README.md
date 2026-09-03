# Restaurant Layout Optimizer

An agent-based restaurant simulator combined with a machine learning pipeline that finds optimal table layouts by minimizing waiter travel distance.

Built entirely from scratch — no frameworks, no library dependencies for the core simulation.

---

## What it does

The simulator runs a full restaurant floor in the browser: guests arrive, get seated, wait for drinks, and leave. Waiters navigate the floor using A* pathfinding to serve orders.

The ML pipeline collects thousands of these simulation runs, trains a surrogate model on the layout data, and uses that model to search for better table arrangements — without running the expensive full simulation for every candidate.

**Result: ~9% reduction in waiter travel distance against the best of 10,240 randomly sampled
layouts** — 320,577 px → 288,884 px, each measured over 30 seeds in a headless browser
(95% CI on the difference: 8.4%–11.4%).

Read that as a range rather than a point. An independent re-validation on a third, separate seed
set put the same comparison at 7%, so the honest span is **7–11%**. The effect survives every
split of the data; its exact size does not pin down to one decimal.

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
train_surrogate.py  ──► surrogate_model.pkl  (GradientBoosting, R²=0.990)
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
- **Performance**: R² = 0.990, MAE = 12,896 px out-of-fold on 10,240 unique layouts

### Why R² is not the headline metric

Global R² is dominated by the gap between disastrous and mediocre layouts, which is not what the optimizer needs. What matters is ranking *good* layouts against each other. `evaluate.py` reports four numbers instead:

| Metric | Meaning | Current |
|---|---|---|
| Validated top-1 | Best layout after real headless validation | 287,929 px (n=8, 15 seeds) |
| Calibration error | Mean (predicted − actual) on validated candidates | −2.7% (n=8) |
| Spearman ρ, best decile | Ranking quality among the top 10% of layouts | 0.513 |
| R² out-of-fold | Deduplicated, weighted per simulation, GroupKFold | 0.990 |

The gap between ρ = 0.96 overall and ρ = 0.51 within the best decile is the single most useful
diagnostic in the pipeline: the model separates bad from good easily, but ranks the good ones
poorly. Two separate limits produce that gap, and only one of them is the model's fault.

**Measurement noise caps it.** Within the best decile the *observed* spread between layouts is
sd ≈ 10,100 px, but that already contains the measurement noise: the noise on a 3-seed mean is
sd ≈ 7,600 px, leaving a true spread of only ≈ 6,700 px. Signal is smaller than noise, so even a
perfect model would score only ρ ≈ 0.62–0.64 there. Reaching the target of 0.70 needs 5 seeds per
layout, not 3 (ceiling ≈ 0.73); 9 seeds would allow ≈ 0.82. Collecting *more layouts* does nothing for this — measured across
1,000 → 10,240 layouts, ρ in the best decile bounces between 0.14 and 0.28 with no trend.

**The remaining gap is the model's.** At 0.51 against a ceiling of 0.62 there is real room left,
and the sharpest symptom is that the model cannot rank its own output: across the 8 validated
candidates, predicted versus actual gives ρ = 0.10. The best layout it produced was the one it
ranked fourth. Candidate *generation* is working; candidate *selection* is not, which is why the
next step is optimizing a lower confidence bound (μ + κσ) rather than μ.

For reference, the noise ceiling on global R² is 0.9966 for a 3-seed mean, so R² = 0.990 still
leaves headroom.

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
| Waiter travel distance | 320,577 px | **288,884 px** |
| Improvement | — | **9.9%** (95% CI 8.4–11.4%, Welch p = 1.7e-18) |
| Standard error | ±1,733 px | ±1,774 px |

Both figures pool 30 seeds per layout from headless validation. The baseline is the best of the
10,240 randomly sampled valid layouts in `restaurant-sim-clean.json`, re-validated at the same
seed count so the two sides are measured identically.

Two honest caveats on this number:

**Winner's curse.** The optimized layout was picked as the best of 8 candidates using the same
seeds it is scored on. Re-measured on fresh seeds it comes out 1,910 px higher — a real but small
selection effect, already absorbed into the 30-seed pooled figure above. An independent reviewer
re-ran the whole comparison on yet another seed set and landed at 7%; that is why the headline is
stated as a range.

**The model is not what is choosing well.** The *average* of the 8 optimizer candidates
(326,020 px) is not significantly better than the baseline group (p = 0.41 against the top 3,
p = 0.06 against the top 12). The win comes from one genuinely excellent layout in the pool, and
the model ranked it fourth. Validation is doing the selecting, not the surrogate — which is the
weakness described under [Why R² is not the headline metric](#why-r-is-not-the-headline-metric).

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
best-layout.json            # Best layout found (287,929 px, 15-seed validated)
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
