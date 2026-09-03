"""
Layout optimizer voor Restaurant Simulator.

Genereert kandidaat-layouts sequentieel-vectorized met numpy (tafels één voor
één plaatsen, maar N layouts tegelijk), scoort ze in batch met het surrogate
model, en verfijnt de top-50 met local search.

Gebruik:
    python3 optimize_layout.py
    python3 optimize_layout.py --candidates 200000 --top 20
"""

import json
import sys
import time
import numpy as np
import joblib

import pathgrid as pg
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent          # data en modelbestanden staan in de repo-root

# ── Constanten (zelfde als simulator) ────────────────────────────────────────
ROOM_W = 640
ROOM_H = 640
BAR_DOCK_X    = ROOM_W - 110          # 530 (voor feature engineering)
BAR_DOCK_Y    = 80                    # simulatie.html:1153 — bovenaan de bar, niet in het midden
BAR_RECT      = (ROOM_W - 90, 50, 70, ROOM_H - 100)
ENTRANCE_RECT = (0, ROOM_H - 90, 110, 90)
WALL_MARGIN   = 20
MIN_CORR      = 50
HALF_CORR     = 25

FIXED_TABLES = [
    {"size": "custom", "x": 130.59, "y":  57.32, "rotation": 0, "w": 46.85,  "h":  97.69, "seats": 6},
    {"size": "custom", "x": 347.91, "y":  60.81, "rotation": 0, "w": 44.86,  "h":  99.69, "seats": 6},
    {"size": "custom", "x":  59.81, "y": 204.36, "rotation": 0, "w": 49.84,  "h": 145.55, "seats": 8},
]

TABLE_TYPES = {
    "small":  {"w":  60, "h": 60, "seats": 2},
    "medium": {"w":  80, "h": 60, "seats": 4},
    "large":  {"w": 110, "h": 70, "seats": 6},
    "comb1":  {"w":  60, "h": 60, "seats": 2},
    "comb2":  {"w": 120, "h": 60, "seats": 4},
    "comb3":  {"w": 180, "h": 60, "seats": 6},
    "comb4":  {"w": 240, "h": 60, "seats": 8},
}

VARIABLE_TYPES = ["large", "large", "medium", "medium", "medium", "medium", "medium", "medium"]


# ── AABB helper ───────────────────────────────────────────────────────────────

def _aabb_scalar(x, y, w, h, rot):
    if int(rot) % 180 == 90:
        cx, cy = x + w / 2, y + h / 2
        return cx - h / 2, cy - w / 2, h, w
    return x, y, w, h


# ── Sequential vectorized layout generator ───────────────────────────────────
#
# Probleem met all-at-once sampling: 8 random tafels in één keer → 99.9%
# botsing door paargewijze overlaps. Oplossing: tafel voor tafel plaatsen,
# maar N layouts tegelijk per stap → vectorized én correct.
#
# Hit rate ~35–50%. Met N=100k per batch → ~40k geldige layouts per batch.

def _collision_ok(ax, ay, aw, ah, placed):
    """(N,) bool — True als positie (ax,ay,aw,ah) vrij is van alle geplaatste items."""
    bx, by, bw, bh = BAR_RECT
    ex, ey, ew, eh = ENTRANCE_RECT
    g, g2 = HALF_CORR, MIN_CORR

    ok = (~((ax < bx+bw+g) & (ax+aw+g > bx) & (ay < by+bh+g) & (ay+ah+g > by)) &
          ~((ax < ex+ew+g2) & (ax+aw+g2 > ex) & (ay < ey+eh+g) & (ay+ah+g > ey)))
    for pax, pay, paw, pah in placed:
        ok &= ~((ax < pax+paw+g2) & (ax+aw+g2 > pax) & (ay < pay+pah+g2) & (ay+ah+g2 > pay))
    return ok


def generate_batch(N, rng, max_tries=200):
    """Genereert tot N geldige layouts (sequentieel-vectorized)."""
    T     = len(VARIABLE_TYPES)
    rot90 = rng.integers(0, 2, N).astype(bool)   # (N,) één rotatie per layout

    # Vaste tafels als placed-lijst (zelfde AABB voor elk van de N layouts)
    placed = []
    for ft in FIXED_TABLES:
        fx, fy, fw, fh = _aabb_scalar(ft["x"], ft["y"], ft["w"], ft["h"], ft["rotation"])
        placed.append((np.full(N, fx, np.float32), np.full(N, fy, np.float32),
                       np.full(N, fw, np.float32), np.full(N, fh, np.float32)))

    result_xs = np.zeros((N, T), np.float32)
    result_ys = np.zeros((N, T), np.float32)
    alive     = np.ones(N, dtype=bool)

    for t_idx, size in enumerate(VARIABLE_TYPES):
        w, h   = float(TABLE_TYPES[size]["w"]), float(TABLE_TYPES[size]["h"])
        hdw    = (w - h) / 2    # x-shift van origin bij rot=90
        hdh    = (h - w) / 2    # y-shift
        ew_arr = np.where(rot90, h, w).astype(np.float32)   # (N,)
        eh_arr = np.where(rot90, w, h).astype(np.float32)

        x_hi   = (ROOM_W - WALL_MARGIN) - ew_arr   # bovengrens x
        y_hi   = (ROOM_H - WALL_MARGIN) - eh_arr
        # Bias: 60% van layouts op rechterkant (x ≥ 250) — dichter bij bar
        x_lo_default = float(WALL_MARGIN)
        x_lo_biased  = 250.0
        bias_mask = (rng.random(N) < 0.60).astype(np.float32)
        x_lo = np.where(bias_mask, x_lo_biased, x_lo_default).astype(np.float32)
        y_lo   = np.full(N, float(WALL_MARGIN), np.float32)

        accepted = np.zeros(N, dtype=bool)

        for _ in range(max_tries):
            need = alive & ~accepted
            if not need.any():
                break
            xs   = x_lo + rng.random(N).astype(np.float32) * (x_hi - x_lo)
            ys   = y_lo + rng.random(N).astype(np.float32) * (y_hi - y_lo)
            ax   = np.where(rot90, xs + hdw, xs)
            ay   = np.where(rot90, ys + hdh, ys)
            ok   = _collision_ok(ax, ay, ew_arr, eh_arr, placed)
            newly = need & ok
            result_xs[newly, t_idx] = xs[newly]
            result_ys[newly, t_idx] = ys[newly]
            accepted |= newly

        alive &= accepted
        if not alive.any():
            break

        # Voeg geplaatste tafel toe aan placed (voor volgende iteraties)
        ax_f = np.where(rot90, result_xs[:, t_idx] + hdw, result_xs[:, t_idx])
        ay_f = np.where(rot90, result_ys[:, t_idx] + hdh, result_ys[:, t_idx])
        placed.append((ax_f, ay_f, ew_arr, eh_arr))

    alive_idx = np.where(alive)[0]
    layouts   = []
    for idx in alive_idx:
        r      = 90 if rot90[idx] else 0
        layout = list(FIXED_TABLES)
        for t_idx, size in enumerate(VARIABLE_TYPES):
            td = TABLE_TYPES[size]
            layout.append({"size": size,
                           "x": float(result_xs[idx, t_idx]),
                           "y": float(result_ys[idx, t_idx]),
                           "rotation": r, "w": td["w"], "h": td["h"],
                           "seats": td["seats"]})
        layouts.append(layout)

    return layouts, len(alive_idx) / N


# ── Feature extractie (zelfde als train_surrogate.py) ────────────────────────

def extract_features(tables, frontier=False):
    from train_surrogate import extract_features_from_list, extract_frontier_features
    var = [t for t in tables if t["size"] != "custom"]
    var.sort(key=lambda t: (-t["w"], t["x"], t["y"]))
    return (extract_frontier_features(var) if frontier
            else extract_features_from_list(var))


def pad(rows, feat_len):
    padded = [r + [0.0] * max(0, feat_len - len(r)) for r in rows]
    return np.array(padded, dtype=np.float32)


# ── Scoring helper (duck-typed voor RF én GNN) ────────────────────────────────

def _score_layouts(model, feat_len, layouts, frontier=False):
    """
    Scoort layouts met RF of GNN — transparant voor de aanroeper.

    Onbereikbare indelingen krijgen inf en worden dus nooit gekozen.

    Die poort is geen luxe. Het model is uitsluitend op geldige layouts
    getraind — de zeef in collect_parallel hield de rest buiten de dataset —
    dus buiten dat gebied voorspelt het niets zinnigs; het extrapoleert. En
    laat dat nu precies zijn waar een optimizer op afgaat. Zonder poort levert
    hij indelingen op die de bar inmetselen: een ober die nergens heen kan
    legt nul afstand af, en dat is de goedkoopste layout die bestaat. Dezelfde
    exploit als in de oude dataset, alleen een niveau hoger — niet in de data
    maar in de zoekstap.

    De toets staat bewust vóór de feature-extractie: hij wijst ongeveer de
    helft van de kandidaten af (gemeten 49%), en die hoeven dan geen
    Dijkstra-sweep meer. Beide stappen kosten een sweep over hetzelfde grid,
    dus dat scheelt ruwweg een derde van de scoretijd.
    """
    if feat_len is None:
        # GNNSurrogate: accepteert rauwe layout-lijsten
        return model.predict(layouts)

    scores    = np.full(len(layouts), np.inf, dtype=float)
    keep, rows = [], []
    for i, layout in enumerate(layouts):
        # Toets de wereld die de simulator daadwerkelijk opbouwt: alleen de
        # variabele tafels. De custom tafels in de kandidaatlijst zijn inerte
        # metadata en staan niet op de vloer (zie pathgrid.py). Ze wel
        # meerekenen zou indelingen afkeuren die in de simulatie prima lopen.
        var = [t for t in layout if t.get("size") != "custom"]
        ok, _unreachable, _trapped = pg.layout_valid(pg.build_blocked(var), var)
        if ok:
            keep.append(i)
            rows.append(extract_features(layout, frontier=frontier))
    if keep:
        scores[keep] = model.predict(pad(rows, feat_len))
    return scores


# ── Random search ─────────────────────────────────────────────────────────────

def random_search(model, feat_len, n_candidates, rng, batch=50_000):
    print(f"Fase 1 — random search ({n_candidates:,} kandidaten)…")
    t0 = time.time()

    all_layouts, all_scores = [], []
    generated = rejected = 0

    while generated < n_candidates:
        size             = min(batch, n_candidates - generated)
        layouts, hit     = generate_batch(size, rng)
        generated       += size

        if not layouts:
            continue

        scores    = _score_layouts(model, feat_len, layouts)
        keep      = [i for i, sc in enumerate(scores) if np.isfinite(sc)]
        rejected += len(layouts) - len(keep)
        all_layouts.extend(layouts[i] for i in keep)
        all_scores.extend(float(scores[i]) for i in keep)
        best = min(all_scores) if all_scores else float("inf")
        print(f"  {generated:>7,} geprobeerd — {len(all_layouts):,} bereikbaar "
              f"({hit*100:.0f}% plaatsbaar, {rejected:,} onbereikbaar afgewezen) "
              f"— beste dist: {best:,.0f}")

    order          = np.argsort(all_scores)
    sorted_layouts = [all_layouts[i] for i in order]
    sorted_scores  = [all_scores[i]  for i in order]
    print(f"  Klaar in {time.time()-t0:.1f}s")
    return sorted_layouts, sorted_scores


# ── Local refinement ──────────────────────────────────────────────────────────

def local_refine(model, feat_len, candidates, scores, top_k, n_rounds, rng,
                 frontier=False):
    if not candidates:
        print("Geen kandidaten voor refinement.")
        return candidates, scores

    print(f"\nFase 2 — local refinement (top-{top_k}, {n_rounds} rondes)…")
    t0 = time.time()

    pool   = [list(c) for c in candidates[:top_k]]
    pscore = list(scores[:top_k])

    # De binnenkomende scores komen van het basismodel. Verfijnen we met het
    # frontier-model, dan moet de startpool op dezelfde schaal staan -- anders
    # vergelijkt de acceptatietoets hieronder appels met peren en wordt een
    # perturbatie aangenomen of verworpen op een schaalverschil.
    if frontier and pool:
        pscore = list(_score_layouts(model, feat_len, pool, frontier=True))

    for _ in range(n_rounds):
        perturbed, origins = [], []
        for ci, layout in enumerate(pool):
            vi_l = [i for i, tb in enumerate(layout) if tb.get("size") != "custom"]
            if not vi_l:
                continue
            idx    = int(rng.choice(vi_l))
            t      = layout[idx]
            nx, ny = t["x"] + rng.uniform(-35, 35), t["y"] + rng.uniform(-35, 35)
            td     = TABLE_TYPES[t["size"]]
            r      = t["rotation"]
            ew     = td["h"] if r % 180 == 90 else td["w"]
            eh     = td["w"] if r % 180 == 90 else td["h"]
            if (nx < WALL_MARGIN or ny < WALL_MARGIN or
                    nx + ew > ROOM_W - WALL_MARGIN or ny + eh > ROOM_H - WALL_MARGIN):
                continue
            new_layout      = [dict(t2) for t2 in layout]
            new_layout[idx] = {**t, "x": nx, "y": ny}
            perturbed.append(new_layout)
            origins.append(ci)

        if not perturbed:
            continue
        new_sc = _score_layouts(model, feat_len, perturbed, frontier=frontier)
        for k, ci in enumerate(origins):
            if new_sc[k] < pscore[ci]:
                pool[ci]   = perturbed[k]
                pscore[ci] = new_sc[k]

    order = np.argsort(pscore)
    print(f"  Klaar in {time.time()-t0:.1f}s — beste na refinement: {pscore[order[0]]:,.0f} px")
    return [pool[i] for i in order], [pscore[i] for i in order]


# ── Gradient search (alleen voor GNN) ────────────────────────────────────────

def gradient_search(gnn_model, candidates, scores, top_k=10, steps=500):
    """
    Gradiëntgebaseerde verfijning van de top-k kandidaten via de GNN.
    Retourneert de verfijnde layouts gesorteerd op voorspelde waiterDist.
    """
    try:
        from gnn_layout import gradient_optimize
    except ImportError:
        print("  gradient_search overgeslagen (gnn_layout niet beschikbaar)")
        return candidates[:top_k], list(scores[:top_k])

    print(f"\nFase 3 — gradient search (top-{top_k}, {steps} stappen per layout)…")
    t0 = time.time()

    config = {"roomW": ROOM_W, "roomH": ROOM_H, "guests": 49, "waiters": 3}
    refined_layouts, refined_scores = [], []

    for i in range(min(top_k, len(candidates))):
        layout = candidates[i]
        try:
            opt_layout = gradient_optimize(
                gnn_model=gnn_model.model,
                initial_layout=layout,
                config=config,
                n_steps=steps,
                lr=1.0,
            )
            sc = gnn_model.predict([opt_layout], config=config)[0]
            refined_layouts.append(opt_layout)
            refined_scores.append(sc)
            print(f"  #{i+1}: {scores[i]:,.0f} → {sc:,.0f} px "
                  f"({'↓' if sc < scores[i] else '↑'}{abs(sc-scores[i]):,.0f})")
        except Exception as e:
            print(f"  #{i+1}: gradient_optimize mislukt ({e}), origineel behouden")
            refined_layouts.append(layout)
            refined_scores.append(scores[i])

    order = np.argsort(refined_scores)
    print(f"  Klaar in {time.time()-t0:.1f}s — beste: {refined_scores[order[0]]:,.0f} px")
    return [refined_layouts[i] for i in order], [refined_scores[i] for i in order]


# ── Export ────────────────────────────────────────────────────────────────────

def export(layouts, pred_scores, out_path, top_n):
    results = []
    for rank, (layout, sc) in enumerate(zip(layouts[:top_n], pred_scores[:top_n])):
        results.append({
            "rank": rank + 1,
            "predicted_waiterDist": round(float(sc)),
            "predicted_score":      round(-float(sc) * 0.02 + 421 * 12, 1),
            "config": {"roomW": ROOM_W, "roomH": ROOM_H, "guests": 49, "waiters": 3,
                       "tSmall": 0, "tMedium": 6, "tLarge": 2, "partyType": "buffet"},
            "tables": layout,
        })
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTop-{top_n} opgeslagen: {out_path}")
    print("\n── Top-5 ────────────────────────────────────────")
    for r in results[:5]:
        print(f"  #{r['rank']}  waiterDist≈{r['predicted_waiterDist']:,}  score≈{r['predicted_score']:.0f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    a, cfg = sys.argv[1:], {"top": 10, "candidates": 200_000, "refine_top": 50, "refine_rounds": 500}
    i = 0
    while i < len(a):
        if   a[i] == "--top"        and i+1 < len(a): cfg["top"]        = int(a[i+1]); i += 2
        elif a[i] == "--candidates" and i+1 < len(a): cfg["candidates"] = int(a[i+1]); i += 2
        else: i += 1
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    rng = np.random.default_rng(42)

    # ── RF laden voor hoofdoptimalisatie ──
    saved    = joblib.load(ROOT / "surrogate_model.pkl")
    model    = saved["model"]
    feat_len = len(saved["feature_names"])
    print(f"Model: {saved['model_name']} — {feat_len} features — target={saved['target']}")

    # Optioneel; een pickle van voor de tweetrapsopzet heeft het niet.
    model_frontier    = saved.get("model_frontier")
    feat_len_frontier = saved.get("feat_len_frontier")
    if model_frontier is None:
        print("Frontier-model niet in de pickle — verfijning draait op het basismodel")

    # ── GNN laden voor gradiëntoptimalisatie (optioneel) ──
    gnn_model = None
    gnn_path  = ROOT / "gnn_model.pt"
    if gnn_path.exists():
        try:
            from gnn_layout import GNNSurrogate
            ckpt = __import__("torch").load(gnn_path, map_location="cpu")
            gnn_r2 = ckpt.get("test_r2", -999)
            if gnn_r2 > 0.55:   # gradiënt-search vereist accurate GNN (R²>0.55)
                gnn_model = GNNSurrogate(gnn_path)
                mae_info  = f"R²={gnn_r2:.3f}, MAE={gnn_model.val_mae:,.0f} px" if gnn_model.val_mae else f"R²={gnn_r2:.3f}"
                print(f"GNN: geladen voor gradient-search ({mae_info})")
            else:
                print(f"GNN: R²={gnn_r2:.3f} — te laag voor gradient-search (drempel R²>0.55)")
        except Exception as e:
            print(f"GNN laden mislukt ({e})")
    print()

    # Twee trappen. De brede zoektocht draait op het basismodel, want de
    # tour-features kosten 35 ms per layout en dat is bij 200.000 kandidaten
    # uren. De verfijning draait op het frontier-model: daar zitten we in de
    # kopgroep, en juist daar tillen die features de rangschikking omhoog
    # (rho 0,392 -> 0,427 tegen een achtergehouden seed).
    layouts, scores = random_search(model, feat_len, cfg["candidates"], rng)

    if model_frontier is not None:
        print(f"\nFrontier-model: {feat_len_frontier} features "
              f"(basis + tour) voor de verfijning")
        layouts, scores = local_refine(model_frontier, feat_len_frontier,
                                       layouts, scores,
                                       top_k=cfg["refine_top"],
                                       n_rounds=cfg["refine_rounds"], rng=rng,
                                       frontier=True)
    else:
        layouts, scores = local_refine(model, feat_len, layouts, scores,
                                       top_k=cfg["refine_top"],
                                       n_rounds=cfg["refine_rounds"], rng=rng)

    # ── Gradiëntoptimalisatie met GNN (als beschikbaar en goed genoeg) ──
    if gnn_model is not None:
        layouts, scores = gradient_search(gnn_model, layouts, scores,
                                          top_k=min(5, len(layouts)),
                                          steps=150)

    export(layouts, scores, ROOT / "optimizer-results.json", top_n=cfg["top"])
