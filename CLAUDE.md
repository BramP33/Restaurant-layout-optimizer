# Restaurant Simulator — CLAUDE.md

## What this project is

A single-file, browser-based **agent-based restaurant simulation** built with vanilla JS and Canvas 2D. No build step, no npm, no external libraries. Everything lives in `index.html`.

The core idea: simulate a restaurant floor where guests arrive, sit, wait for drinks, and leave. A layout optimizer regenerates the table placement each run and scores efficiency. The goal is to use repeated simulation runs to find which table layouts minimize waiter travel distance and guest impatience.

> **Note:** The second `<script data-pplx-inline-edit>` block at the bottom of the file is Perplexity's screenshot/iframe bridge — it's not part of the simulation logic and should be removed when deploying standalone.

---

## Architecture

All code is in a single `<script>` block inside `index.html`. Layout: left sidebar controls, center canvas, right metrics panel.

### Classes

| Class | Role |
|---|---|
| `PathNavigator` | A* pathfinding on an 18px-cell grid. Rasterizes furniture into blocked cells. Used by both guests and waiters. |
| `Table` | Furniture unit. Owns bounding rect, chair anchor points, seated guest list, and order state. |
| `Guest` | Agent with state machine. Handles movement, waiting logic, impatience, excursions. |
| `Waiter` | Task-driven agent. Carries up to 8 drinks, plans multi-table serving trips. |
| `SimulationEngine` | Owns the world: tables, agents, bar, entrance, metrics, rAF loop. |

### Guest state machine

```
WALKING → SEATED → WAITING → DRINKING → LEAVING
                       ↓                  ↑
                      BAR ──────────────→ (exit)
                                DRINKING → SOCIALIZING → (back to DRINKING)
                                DRINKING → TOILET → (back to DRINKING)
```

- Wait threshold: 18 simulated seconds. After that, guest storms to BAR → counted as impatient.
- Impatient guests release their seat so new arrivals can use it.

### Waiter state machine

```
IDLE → WALKING_TO_TABLE → TAKING_ORDER → WALKING_TO_BAR → COLLECTING_DRINKS → SERVING → IDLE
```

- Builds a greedy nearest-first multi-table serve plan (up to 8 drinks capacity).
- Waiters won't double-claim a table already targeted by a colleague.

### Layout optimizer (`generateLayout`)

Greedy sampler: for each table, generate 80 random candidate positions, score each, pick the best non-colliding one.

Score terms:
- Distance from bar dock (lower = better)
- Penalty for being near the entrance corridor
- Bonus for grouping with same-size tables

Fallback: coarse grid scan if 80 random samples fail (very dense configs).

### Efficiency score formula

```
score = (servedDrinks × 12) - (avgWait × 3) - (impatientGuests × 25) - (waiterDistance × 0.02)
```

A well-tuned run scores ~100–300. Poor configs go negative.

---

## UI controls

| Control | What it does |
|---|---|
| Room Width / Height | Canvas dimensions, regenerated on next run |
| Grid | Visual grid overlay size (cosmetic only) |
| Guests (5–50) | Total guests spawned, staggered over ~30 sim-seconds |
| Waiters (1–6) | Number of waiter agents |
| Table Mix | Count of small(2), medium(4), large(6) seat tables |
| Speed | 1×/2×/5×/10× simulation time multiplier |

---

## Key constants / magic numbers

- `minCorridor = 36px` — minimum gap between tables in layout optimizer
- `waitThreshold = 18` simulated seconds before guest goes impatient
- `waiter.capacity = 8` drinks max per trip
- `guest.speed = 55–75 px/sec`, `waiter.speed = 85–105 px/sec`
- Guests spawn staggered: one every `30 / guestCount` seconds + small jitter
- Bar dock: fixed to right wall at `roomW - 110`
- Entrance: fixed bottom-left at `x=48, y=roomH-54`

---

## What likely needs to change

Based on the project being a training/research tool for table layouts:

1. **Remove Perplexity inline-edit script** — the `<script data-pplx-inline-edit>` block at lines ~1508–1835 is Perplexity boilerplate, not part of the sim.
2. **Persistence / run history** — currently `bestScore` resets on page reload. A run log (localStorage or export) would let you compare layouts across sessions.
3. **Layout seed / reproducibility** — runs are fully random; no way to replay a specific layout.
4. **ML/optimizer hook** — the architecture is designed so `generateLayout()` can be swapped for a smarter algorithm without touching state-machine code.
5. **Language** — UI labels are English; project context (tafelindeling) is Dutch.
