"""
Surrogate model voor Restaurant Simulator layout optimizer.

Verbeterde versie met:
- Rijkere feature set (per-tafel barstand, gesorteerde afstanden, compactheid)
- XGBoost + GradientBoosting + RandomForest vergelijking
- Dedupe op layout: identieke indelingen worden samengevoegd tot een rij met
  een target dat per simulatie gewogen is, zodat ze niet in train en test
  tegelijk belanden
- GroupKFold op layout-sleutel als vangnet; de dedupe is wat het lek wegneemt
- Log-ruimte target (waiterDist loopt van ~253k tot ~1,3M px)
- Rapporteert Spearman binnen het beste deciel naast R², en schrijft de
  out-of-fold voorspellingen weg voor evaluate.py
- Slaat het beste model op

Gebruik:
    python3 train_surrogate.py
"""

import json
import numpy as np
from pathlib import Path
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

from log_target import LogTargetModel
import xgboost as xgb  # type: ignore
import joblib

HERE       = Path(__file__).parent
ROOT       = HERE.parent          # data en modelbestanden staan in de repo-root
MODEL_FILE = ROOT / "surrogate_model.pkl"
OOF_FILE   = ROOT / "surrogate-oof.json"   # out-of-fold voorspellingen voor evaluate.py

BAR_DOCK_X = 640 - 110   # 530
BAR_DOCK_Y = 640 / 2     # 320 — consistent met optimizer


# ── Data laden ────────────────────────────────────────────────────────────────

def _find_data():
    merged = ROOT / "restaurant-sim-merged.json"
    if merged.exists():
        return merged
    files = sorted(ROOT.glob("restaurant-sim-batch-*.json"))
    if files:
        return files[-1]
    raise FileNotFoundError("Geen dataset gevonden.")


def layout_key(variable):
    """Stabiele sleutel per indeling — identieke layouts krijgen dezelfde sleutel."""
    return tuple(sorted((round(t["x"], 1), round(t["y"], 1), t["size"], t["rotation"])
                        for t in variable))


def load_data(path):
    """
    Laadt runs, dedupliceert op layout en middelt alle seeds per layout.

    Dezelfde indeling komt meermaals in de dataset voor (later opnieuw gedraaid
    met extra seeds). Zonder dedupe belandt eenzelfde layout in train en test,
    wat de cross-validatie optimistisch maakt. Hier worden ze samengevoegd tot
    een rij met een seed-gewogen target.
    """
    with open(path) as f:
        runs = json.load(f)

    agg = {}
    for run in runs:
        variable = [t for t in run["tables"] if t["size"] != "custom"]
        if not variable:
            continue

        n_seeds = len(run.get("seeds", []))
        if n_seeds < 2:               # ruis-arm: alleen multi-seed layouts
            continue

        variable.sort(key=lambda t: (-t["w"], t["x"], t["y"]))
        key = layout_key(variable)
        m   = run["metrics"]

        # Er zitten twee formaten door elkaar in de dataset:
        #   - met "runs": metrics is het gemiddelde over die sub-runs, dus dit
        #     record staat voor len(seeds) simulaties.
        #   - zonder "runs": metrics komt van EEN seed. validate_headless.js
        #     schrijft per seed een record weg met telkens de volledige
        #     seedlijst van de batch erin, dus len(seeds) zegt hier niets over
        #     dit record. Wegen op len(seeds) telt die groep n keer te zwaar
        #     (n^2 in plaats van n) en scheeft zodra dezelfde layout later met
        #     een ander aantal seeds opnieuw gevalideerd wordt.
        weight = n_seeds if "runs" in run else 1

        a = agg.setdefault(key, {"variable": variable, "dist": 0.0,
                                 "score": 0.0, "wait": 0.0, "n_seeds": 0})
        a["dist"]    += m["waiterDist"] * weight
        a["score"]   += m["score"]      * weight
        a["wait"]    += m["avgWait"]    * weight
        a["n_seeds"] += weight

    print(f"  {len(runs)} runs -> {len(agg)} unieke multi-seed layouts")

    X_rows, y_rows, w_rows, meta_rows = [], [], [], []
    for a in agg.values():
        n = a["n_seeds"]
        X_rows.append(extract_features_from_list(a["variable"]))
        y_rows.append(a["dist"] / n)
        w_rows.append(n)
        meta_rows.append({
            "score":      a["score"] / n,
            "waiterDist": a["dist"]  / n,
            "avgWait":    a["wait"]  / n,
            "n_seeds":    n,
        })

    max_len = max(len(r) for r in X_rows)
    X_rows  = [r + [0.0] * (max_len - len(r)) for r in X_rows]
    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float64)
    w = np.array(w_rows, dtype=np.float64)
    return X, y, w, max_len, meta_rows


def extract_features_from_list(variable):
    """Rijke feature-extractie uit een gesorteerde lijst van variabele tafels."""
    n   = len(variable)
    cx  = np.array([t["x"] for t in variable])
    cy  = np.array([t["y"] for t in variable])
    cw  = np.array([t["w"] for t in variable])
    ch  = np.array([t["h"] for t in variable])

    # Positie + afmetingen per tafel (raw)
    raw = []
    for t in variable:
        raw += [t["x"], t["y"], t["rotation"], t["w"], t["h"]]

    # Per-tafel afstand tot bar dock
    bar_dists = np.sqrt((cx - BAR_DOCK_X)**2 + (cy - BAR_DOCK_Y)**2)

    # Gesorteerde bar-afstanden (extra informatief voor model)
    sorted_dists = np.sort(bar_dists)

    # Pairwise afstanden (compactheid / clustering)
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2))
    diffs = np.array(diffs) if diffs else np.array([0.0])

    # Zwaartepunt en spreiding
    cx_mean, cy_mean = cx.mean(), cy.mean()
    cx_std,  cy_std  = cx.std(),  cy.std()

    # Rechts-bias: hoe dicht bij de bar (rechterkant, x > 400)
    right_bias = (cx > 400).sum() / n

    # Afstand van zwaartepunt tot bar
    centroid_to_bar = np.sqrt((cx_mean - BAR_DOCK_X)**2 + (cy_mean - BAR_DOCK_Y)**2)

    # Compactheid: gemiddeld inter-tafel afstand normaliseerd
    compactness = diffs.mean() / 640.0

    # Edge-to-edge corridor widths between table pairs (pad-blokkering detectie)
    edge_gaps = []
    for i in range(n):
        for j in range(i + 1, n):
            # Horizontale en verticale edge-to-edge gaps
            gap_x = max(0.0, max(cx[i], cx[j]) - min(cx[i]+cw[i], cx[j]+cw[j]))
            gap_y = max(0.0, max(cy[i], cy[j]) - min(cy[i]+ch[i], cy[j]+ch[j]))
            # Effectieve corridor: minimaal van x/y als ze overlappen in de andere as
            overlap_x = min(cx[i]+cw[i], cx[j]+cw[j]) - max(cx[i], cx[j])
            overlap_y = min(cy[i]+ch[i], cy[j]+ch[j]) - max(cy[i], cy[j])
            if overlap_x > 0:   # naast elkaar in x → y-gap is de corridor
                edge_gaps.append(gap_y)
            elif overlap_y > 0:  # boven/onder elkaar → x-gap is de corridor
                edge_gaps.append(gap_x)
            else:                # diagonaal → min van beide
                edge_gaps.append(min(gap_x, gap_y))
    edge_gaps = np.array(edge_gaps) if edge_gaps else np.array([640.0])

    # Minimum gap tot rechter wand (waar de bar is)
    bar_wall_x = 640 - 90   # linkerrand van bar-rect
    gap_to_bar_wall = bar_wall_x - (cx + cw)  # afstand rechterrand tafel → bar-wand
    min_gap_bar   = gap_to_bar_wall.min()

    # Engeneered features
    eng = [
        bar_dists.mean(), bar_dists.min(), bar_dists.max(),
        bar_dists.std(),  bar_dists.sum(),
        *sorted_dists,                    # gesorteerd: van dichtst naar verst
        cx_mean, cy_mean, cx_std, cy_std,
        diffs.mean(), diffs.std(), diffs.min(), diffs.max(),
        right_bias, centroid_to_bar, compactness,
        cw.mean(), ch.mean(),             # gem. tafelgrootte
        # Corridor features (pad-blokkering)
        edge_gaps.min(), edge_gaps.mean(),
        float((edge_gaps < 36).sum()),    # aantal te-nauwe corridors (< A*-cel × 2)
        float((edge_gaps < 50).sum()),    # aantal krappe corridors
        min_gap_bar,                      # ruimte naar bar-wand
    ]

    return raw + eng


# ── Modellen ─────────────────────────────────────────────────

def build_models():
    return {
        "XGBoost": xgb.XGBRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_features=0.5,
            random_state=42, n_jobs=-1,
        ),
    }


# ── Metrieken ───────────────────────────────────────────────

def top_decile_spearman(y_true, y_pred):
    """
    Rangordecorrelatie binnen de 10% beste layouts.

    Globale R² wordt gedomineerd door het verschil tussen rampzalige en oke
    indelingen. De optimizer heeft juist nodig dat het model góéde layouts
    onderling kan rangschikken — dat meet deze.
    """
    k   = max(10, len(y_true) // 10)
    idx = np.argsort(y_true)[:k]          # laagste waiterDist = beste
    return float(spearmanr(y_true[idx], y_pred[idx]).statistic)


# ── Trainen + evalueren ─────────────────────────────────────────

def train_and_evaluate(X, y, w, groups):
    print(f"\nDataset: {len(X)} unieke layouts, {X.shape[1]} features")
    print(f"waiterDist bereik: {y.min():,.0f} -> {y.max():,.0f}  (med {np.median(y):,.0f})")
    print(f"Seeds per layout: {w.min():.0f}-{w.max():.0f} (gem {w.mean():.1f})\n")

    gkf     = GroupKFold(n_splits=5)
    results = {}

    for name, base in build_models().items():
        oof = np.zeros(len(y), dtype=np.float64)
        for tr, te in gkf.split(X, y, groups=groups):
            m = LogTargetModel(clone(base))
            m.fit(X[tr], y[tr], sample_weight=w[tr])
            oof[te] = m.predict(X[te])

        results[name] = {
            "r2":      float(r2_score(y, oof)),
            "mae":     float(mean_absolute_error(y, oof)),
            "rho_all": float(spearmanr(y, oof).statistic),
            "rho_top": top_decile_spearman(y, oof),
            "oof":     oof,
        }
        r = results[name]
        print(f"  {name:>20}  R²={r['r2']:.3f}  MAE={r['mae']:,.0f} px  "
              f"rho={r['rho_all']:.3f}  rho(top10%)={r['rho_top']:.3f}")

    best_name = max(results, key=lambda n: results[n]["r2"])
    best      = results[best_name]
    print(f"\n  -> Beste: {best_name}  (R²={best['r2']:.3f}, MAE={best['mae']:,.0f} px)")

    model = LogTargetModel(clone(build_models()[best_name]))
    model.fit(X, y, sample_weight=w)

    imp = model.feature_importances_
    if imp is not None:
        print("\n  Top-10 feature importances:")
        for i in np.argsort(imp)[::-1][:10]:
            print(f"    [f{i:>2}] {imp[i]:.4f}")

    return model, best_name, best


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    data_path = _find_data()
    print(f"Dataset: {data_path.name}")
    X, y, w, feat_len, meta_rows = load_data(data_path)

    # Na dedupe is elke rij een unieke layout; de groepen houden de garantie
    # expliciet dat eenzelfde indeling nooit over folds heen kan lekken.
    groups = np.arange(len(y))
    model, model_name, best = train_and_evaluate(X, y, w, groups)

    with open(OOF_FILE, "w") as f:
        json.dump({"model":   model_name,
                   "y_true":  y.tolist(),
                   "y_pred":  best["oof"].tolist(),
                   "n_seeds": w.tolist()}, f)

    joblib.dump({
        "model":         model,
        "model_name":    model_name,
        "feature_names": [f"f{i}" for i in range(feat_len)],
        "feat_len":      feat_len,
        "target":        "waiterDist",
        "log_target":    True,
        "n_layouts":     int(len(y)),
        "r2":            best["r2"],
        "mae":           best["mae"],
        "rho_top10":     best["rho_top"],
        "bar_dock_x":    BAR_DOCK_X,
        "bar_dock_y":    BAR_DOCK_Y,
    }, MODEL_FILE)
    print(f"\nModel opgeslagen: {MODEL_FILE}  (R²={best['r2']:.3f})")
    print(f"OOF-voorspellingen: {OOF_FILE}")
