"""
Surrogate model voor Restaurant Simulator layout optimizer.

Verbeterde versie met:
- Rijkere feature set (per-tafel barstand, gesorteerde afstanden, compactheid)
- XGBoost + GradientBoosting + RandomForest vergelijking
- Cross-validated hyperparameter selectie
- Slaat het beste model op

Gebruik:
    python3 train_surrogate.py
"""

import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb  # type: ignore
import joblib

HERE      = Path(__file__).parent
MODEL_FILE = HERE / "surrogate_model.pkl"

BAR_DOCK_X = 640 - 110   # 530
BAR_DOCK_Y = 640 / 2     # 320 — consistent met optimizer


# ── Data laden ────────────────────────────────────────────────────────────────

def _find_data():
    merged = HERE / "restaurant-sim-merged.json"
    if merged.exists():
        return merged
    files = sorted(HERE.glob("restaurant-sim-batch-*.json"))
    if files:
        return files[-1]
    raise FileNotFoundError("Geen dataset gevonden.")


def load_data(path):
    with open(path) as f:
        runs = json.load(f)

    # Ruis-arm: alleen multi-seed layouts
    multi = [r for r in runs if len(r.get("seeds", [])) > 1]
    if len(multi) >= 100:
        runs = multi
        print(f"  Gefilterd: {len(runs)} multi-seed layouts")

    X_rows, y_rows, meta_rows = [], [], []

    for run in runs:
        tables   = run["tables"]
        variable = [t for t in tables if t["size"] != "custom"]
        if not variable:
            continue

        variable.sort(key=lambda t: (-t["w"], t["x"], t["y"]))

        row = extract_features_from_list(variable)
        X_rows.append(row)
        y_rows.append(run["metrics"]["waiterDist"])
        meta_rows.append({
            "seed":       run["seed"],
            "score":      run["metrics"]["score"],
            "waiterDist": run["metrics"]["waiterDist"],
            "avgWait":    run["metrics"]["avgWait"],
            "n_seeds":    len(run.get("seeds", [run["seed"]])),
        })

    max_len      = max(len(r) for r in X_rows)
    X_rows       = [r + [0.0] * (max_len - len(r)) for r in X_rows]
    X            = np.array(X_rows, dtype=np.float32)
    y            = np.array(y_rows, dtype=np.float32)
    return X, y, max_len, meta_rows


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


# ── Modellen ──────────────────────────────────────────────────────────────────

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


# ── Trainen + evalueren ───────────────────────────────────────────────────────

def train_and_evaluate(X, y):
    print(f"\nDataset: {len(X)} layouts, {X.shape[1]} features")
    print(f"waiterDist bereik: {y.min():,.0f} → {y.max():,.0f}  (med {np.median(y):,.0f})\n")

    models  = build_models()
    kf      = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        cv_r2  = cross_val_score(model, X, y, cv=kf, scoring="r2")
        cv_mae = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
        results[name] = {"r2": cv_r2.mean(), "r2_std": cv_r2.std(), "mae": -cv_mae.mean()}
        print(f"  {name:>20}  R²={cv_r2.mean():.3f} ±{cv_r2.std():.3f}  "
              f"MAE={-cv_mae.mean():,.0f} px")

    best_name  = max(results, key=lambda n: results[n]["r2"])
    best_model = models[best_name]
    best_r2    = results[best_name]["r2"]
    best_mae   = results[best_name]["mae"]
    print(f"\n  → Beste: {best_name}  (R²={best_r2:.3f}, MAE={best_mae:,.0f} px)")

    best_model.fit(X, y)

    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        imp        = best_model.feature_importances_
        idx_sorted = np.argsort(imp)[::-1][:10]
        print(f"\n  Top-10 feature importances:")
        for i in idx_sorted:
            print(f"    [f{i:>2}] {imp[i]:.4f}")

    return best_model, best_name, best_r2, best_mae


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_path = _find_data()
    print(f"Dataset: {data_path.name}")
    X, y, feat_len, meta_rows = load_data(data_path)

    n_seeds = [m["n_seeds"] for m in meta_rows]
    print(f"  Seeds/layout: {min(n_seeds)}–{max(n_seeds)} (gem {np.mean(n_seeds):.1f})")
    print(f"  Score bereik: {min(m['score'] for m in meta_rows):.0f} → "
          f"{max(m['score'] for m in meta_rows):.0f}")

    model, model_name, r2, mae = train_and_evaluate(X, y)

    joblib.dump({
        "model":         model,
        "model_name":    model_name,
        "feature_names": [f"f{i}" for i in range(feat_len)],
        "feat_len":      feat_len,
        "target":        "waiterDist",
        "r2":            r2,
        "mae":           mae,
        "bar_dock_x":    BAR_DOCK_X,
        "bar_dock_y":    BAR_DOCK_Y,
    }, MODEL_FILE)
    print(f"\nModel opgeslagen: {MODEL_FILE}  (R²={r2:.3f})")
