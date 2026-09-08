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

import argparse
import json
import math

import numpy as np
from pathlib import Path
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

import pathgrid as pg
from rooms import CLASSIC
from log_target import LogTargetModel
import xgboost as xgb  # type: ignore
import joblib

HERE       = Path(__file__).parent
ROOT       = HERE.parent          # data en modelbestanden staan in de repo-root
MODEL_FILE = ROOT / "surrogate_model.pkl"
OOF_FILE   = ROOT / "surrogate-oof.json"   # out-of-fold voorspellingen voor evaluate.py

BAR_DOCK_X = 640 - 110   # 530
# 80, niet 320: simulatie.html:1153 legt de dock bovenaan de bar, niet in
# het midden. Met 320 wees elke bar-feature naar een punt waar de obers
# nooit staan.
BAR_DOCK_Y = 80


# ── Data laden ────────────────────────────────────────────────────────────────

def _find_data(explicit=None):
    """
    Kiest de dataset. Volgorde is niet willekeurig.

    restaurant-sim-clean.json komt uit de her-collectie na de pathfinding-fix
    en is de enige set waarin geen enkele layout de bar kan inmetselen.
    restaurant-sim-merged.json is de oude set: die bevat de geexploiteerde
    runs. Hij staat nog wel in de zoekvolgorde -- als de schone set ontbreekt
    is een oud model beter dan geen model -- maar komt pas na clean, zodat hij
    niet meer per ongeluk gepakt wordt. Geef --data mee om expliciet te kiezen.
    """
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Dataset niet gevonden: {path}")
        return path
    for name in ("restaurant-sim-clean.json", "restaurant-sim-merged.json"):
        candidate = ROOT / name
        if candidate.exists():
            return candidate
    files = sorted(ROOT.glob("restaurant-sim-batch-*.json"))
    if files:
        return files[-1]
    raise FileNotFoundError("Geen dataset gevonden.")


def run_room(run):
    """De zaal waarin deze run gedraaid is; ontbreekt hij, dan de klassieke."""
    return (run.get("config") or {}).get("room") or CLASSIC


def layout_key(variable, room=CLASSIC):
    """
    Stabiele sleutel per indeling.

    De ZAAL hoort erin. Zonder dat zijn identieke tafelposities in twee
    verschillende zalen dezelfde sleutel, worden ze samengevoegd tot een rij,
    en middelt de dedupe twee onvergelijkbare metingen door elkaar.
    """
    b, f = room["bar"], room.get("buffet")
    rk = (room.get("kind", "?"), room["w"], room["h"],
          b["x"], b["y"], b["w"], b["h"],
          (f["x"], f["y"], f["w"], f["h"]) if f else None,
          room["entrance"]["x"], room["entrance"]["y"],
          tuple(sorted((x["x"], x["y"], x["w"], x["h"]) for x in room.get("blocks", []))))
    return (rk, tuple(sorted((round(t["x"], 1), round(t["y"], 1), t["size"], t["rotation"])
                             for t in variable)))


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
        room = run_room(run)
        key  = layout_key(variable, room)
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

        a = agg.setdefault(key, {"variable": variable, "room": room, "dist": 0.0,
                                 "score": 0.0, "wait": 0.0, "n_seeds": 0})
        a["dist"]    += m["waiterDist"] * weight
        a["score"]   += m["score"]      * weight
        a["wait"]    += m["avgWait"]    * weight
        a["n_seeds"] += weight

    print(f"  {len(runs)} runs -> {len(agg)} unieke multi-seed layouts")

    X_rows, y_rows, w_rows, meta_rows, var_rows = [], [], [], [], []
    for a in agg.values():
        n = a["n_seeds"]
        X_rows.append(extract_features_from_list(a["variable"], a["room"]))
        var_rows.append((a["variable"], a["room"]))
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
    return X, y, w, max_len, meta_rows, var_rows


# Vaste lengte voor per-tafel blokken. Twaalf, niet acht: sinds de tafelmix
# met de zaal meeschaalt loopt het aantal van 3 tot 11. Met acht sloten viel
# bijna een kwart van de zalen buiten de vector -- en de gesorteerde
# afstandsvector houdt de DICHTSTBIJZIJNDE acht, dus juist de dure tafels
# vielen weg.
N_TABLE_SLOTS = 12


def room_features(room):
    """
    Beschrijft de ZAAL zelf, dimensieloos waar het kan.

    Zonder deze features kan het model onmogelijk generaliseren: alle andere
    features beschrijven waar de tafels staan, en dat betekent iets anders in
    een smalle zaal met kolommen dan in een vierkante lege zaal.
    """
    w, h = float(room["w"]), float(room["h"])
    diag = math.hypot(w, h)
    dock = room["bar"]["dock"]
    ent  = room["entrance"]
    blocks = room.get("blocks", [])
    block_area = sum(b["w"] * b["h"] for b in blocks)
    bar = room["bar"]
    buf = room.get("buffet")
    return [
        w / 1000.0, h / 1000.0,                 # schaal
        w / h,                                  # zijverhouding
        diag / 1000.0,
        (w * h) / 1e6,                          # oppervlak
        block_area / (w * h),                   # aandeel vloer weggenomen
        float(len(blocks)),
        dock["x"] / w, dock["y"] / h,           # waar de bediening vandaan komt
        ent["x"] / w, ent["y"] / h,             # waar de gasten binnenkomen
        (bar["w"] * bar["h"]) / (w * h),
        1.0 if buf else 0.0,
        (buf["x"] + buf["w"] / 2) / w if buf else 0.5,
        (buf["y"] + buf["h"] / 2) / h if buf else 0.5,
    ]


def extract_features_from_list(variable, room=CLASSIC):
    """
    Feature-extractie uit een gesorteerde lijst van variabele tafels.

    Alles wat een lengte in pixels is wordt gedeeld door de diagonaal van de
    zaal, en alle posities door de zaalmaten. Zonder die normalisatie betekent
    x = 400 iets totaal anders in een zaal van 520 breed dan in een van 900, en
    dan leert het model de zaal in plaats van de indeling.

    Per-tafel blokken hebben een VASTE lengte (N_TABLE_SLOTS). Eerder had de
    gesorteerde afstandsvector lengte n; bij een ander tafelaantal schoof
    daardoor de hele rest van de featurevector op, stilzwijgend.
    """
    n    = len(variable)
    W, H = float(room["w"]), float(room["h"])
    diag = math.hypot(W, H)
    dock = room["bar"]["dock"]

    cx  = np.array([t["x"] for t in variable], dtype=float)
    cy  = np.array([t["y"] for t in variable], dtype=float)
    cw  = np.array([t["w"] for t in variable], dtype=float)
    ch  = np.array([t["h"] for t in variable], dtype=float)

    def fixed(arr, fill=0.0):
        """Kap of vul aan tot N_TABLE_SLOTS, zodat de vector nooit verschuift."""
        a = list(arr)[:N_TABLE_SLOTS]
        return a + [fill] * (N_TABLE_SLOTS - len(a))

    def fixed_far(arr):
        """
        Zelfde, maar vult aan met de VERSTE waarde in plaats van nul.

        Een ontbrekende tafel aanvullen met 0 leest als een tafel pal naast de
        bar -- de gunstigst mogelijke waarde. Aanvullen met de verste maakt van
        "die tafel is er niet" iets neutraals in plaats van iets goeds.
        """
        a = sorted(arr)[:N_TABLE_SLOTS]
        return a + [a[-1] if a else 0.0] * (N_TABLE_SLOTS - len(a))

    # Positie en afmeting per tafel, genormaliseerd op de zaal.
    #
    # Lege sloten krijgen het GEMIDDELDE van de aanwezige tafels, niet nul.
    # Sinds de tafelmix met de zaal meeschaalt heeft 99% van de zalen
    # opvulling, gemiddeld vijf sloten; die met nul vullen zet vijf
    # fantoomtafels in de linkerbovenhoek en dat is geen neutrale waarde maar
    # een extreme. Het aantal echte tafels staat als aparte feature in `eng`,
    # dus het model kan opvulling herkennen.
    fill = [float(np.mean(cx / W)), float(np.mean(cy / H)),
            float(np.mean([t.get("rotation", 0) / 90.0 for t in variable])),
            float(np.mean(cw / diag)), float(np.mean(ch / diag))] if n else [0.0] * 5
    raw = []
    for t in fixed(variable, None):
        if t is None:
            raw += fill
        else:
            raw += [t["x"] / W, t["y"] / H, t.get("rotation", 0) / 90.0,
                    t["w"] / diag, t["h"] / diag]

    # Hemelsbrede afstand tot de bardock, in eenheden van de diagonaal.
    bar_dists = np.sqrt((cx - dock["x"])**2 + (cy - dock["y"])**2) / diag

    # Paarsgewijze afstanden: compactheid en clustering.
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(math.hypot(cx[i]-cx[j], cy[i]-cy[j]) / diag)
    diffs = np.array(diffs) if diffs else np.array([0.0])

    # Hoeveel tafels staan in de helft van de zaal waar de bar is? Dat is de
    # zaalonafhankelijke versie van de oude "x > 400"-drempel, die aannam dat
    # de bar altijd rechts stond.
    if abs(dock["x"] - W / 2) >= abs(dock["y"] - H / 2):
        near_bar = (cx > W / 2) if dock["x"] > W / 2 else (cx < W / 2)
    else:
        near_bar = (cy > H / 2) if dock["y"] > H / 2 else (cy < H / 2)
    bar_side = float(near_bar.sum()) / max(n, 1)

    cx_n, cy_n = cx / W, cy / H
    centroid_to_bar = math.hypot(cx.mean() - dock["x"], cy.mean() - dock["y"]) / diag

    # Corridorbreedtes tussen tafelparen: detecteert dichtgezette doorgangen.
    edge_gaps = []
    for i in range(n):
        for j in range(i + 1, n):
            gap_x = max(0.0, max(cx[i], cx[j]) - min(cx[i]+cw[i], cx[j]+cw[j]))
            gap_y = max(0.0, max(cy[i], cy[j]) - min(cy[i]+ch[i], cy[j]+ch[j]))
            ov_x  = min(cx[i]+cw[i], cx[j]+cw[j]) - max(cx[i], cx[j])
            ov_y  = min(cy[i]+ch[i], cy[j]+ch[j]) - max(cy[i], cy[j])
            edge_gaps.append(gap_y if ov_x > 0 else (gap_x if ov_y > 0 else min(gap_x, gap_y)))
    edge_gaps = np.array(edge_gaps) if edge_gaps else np.array([diag])

    # Kleinste vrije ruimte tussen een tafel en de bar-rechthoek, aan welke
    # wand die ook staat. De oude versie rekende met `640 - 90 - (x + w)` en
    # codeerde daarmee "de bar staat rechts" in een getal.
    b = room["bar"]
    gaps_bar = []
    for i in range(n):
        dx = max(b["x"] - (cx[i] + cw[i]), cx[i] - (b["x"] + b["w"]), 0.0)
        dy = max(b["y"] - (cy[i] + ch[i]), cy[i] - (b["y"] + b["h"]), 0.0)
        gaps_bar.append(math.hypot(dx, dy) / diag)
    min_gap_bar = min(gaps_bar) if gaps_bar else 0.0

    # Zitplaatsen en belasting. Een zaal met kolommen krijgt stelselmatig
    # minder tafels per m2 vrije vloer -- die vloer ligt daar in losse stukken
    # waar geen tafel in past. Dat is echte meetkunde, maar zonder deze
    # features ziet het model alleen "kolommen" en niet "licht belast", en dan
    # verdunt precies het signaal waarvoor de zaalvariatie bestaat.
    seats = sum(int(pg.normalise_table(t).get("seats", 4)) for t in variable)
    free_cells = float((~pg.build_blocked(variable, room)).sum())
    free_area  = free_cells * pg.CELL ** 2
    load = seats / (free_area / 1e5) if free_area > 0 else 0.0

    eng = [
        # Hoeveel tafels er staan is nu een variabele: de mix schaalt met de
        # zaal. Zonder deze twee moet het model het terugrekenen uit sommen, en
        # leest een opgevulde plek als een echte tafel.
        float(n), float(n) / N_TABLE_SLOTS,
        float(seats), load,
        bar_dists.mean(), bar_dists.min(), bar_dists.max(),
        bar_dists.std(),  bar_dists.sum(),
        *fixed_far(bar_dists),
        cx_n.mean(), cy_n.mean(), cx_n.std(), cy_n.std(),
        diffs.mean(), diffs.std(), diffs.min(), diffs.max(),
        bar_side, centroid_to_bar, diffs.mean(),
        cw.mean() / diag, ch.mean() / diag,
        edge_gaps.min() / diag, edge_gaps.mean() / diag,
        float((edge_gaps < 36).sum()),
        float((edge_gaps < 50).sum()),
        min_gap_bar,
    ]

    # Padfeatures: de eng-lijst hierboven is volledig Euclidisch, terwijl het
    # target een A*-padlengte is. pathgrid bouwt hetzelfde loopgrid als de
    # simulator, nu voor deze zaal.
    path = pg.path_features(variable, room)

    return [float(v) for v in (room_features(room) + raw + eng + path)]


def extract_frontier_features(variable, room=CLASSIC):
    """
    Basisfeatures plus de tour-features.

    Apart gehouden omdat ze acht keer duurder zijn (~35 ms tegen ~4 ms per
    layout): te traag om 200.000 kandidaten mee af te zoeken, de moeite waard
    om de kopgroep mee te herordenen. Gemeten tegen een ACHTERGEHOUDEN seed
    tilt dit de rangschikking binnen het door het model gekozen deciel van
    rho 0,392 naar 0,427, consistent over alle drie de seeds. Zie README.
    """
    return extract_features_from_list(variable, room) + pg.tour_features(variable, room)


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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None,
                    help="Pad naar de dataset (standaard: restaurant-sim-clean.json)")
    cli = ap.parse_args()

    data_path = _find_data(cli.data)
    print(f"Dataset: {data_path.name}")
    X, y, w, feat_len, meta_rows, variables = load_data(data_path)

    # Na dedupe is elke rij een unieke layout; de groepen houden de garantie
    # expliciet dat eenzelfde indeling nooit over folds heen kan lekken.
    groups = np.arange(len(y))
    model, model_name, best = train_and_evaluate(X, y, w, groups)

    # Frontier-model: dezelfde data, plus de tour-features. De optimizer
    # gebruikt het basismodel voor de brede zoektocht en dit model om de
    # kopgroep te herordenen -- precies waar de tour-features helpen.
    print("\n  Frontier-model (basis + tour-features)…")
    T  = np.array([pg.tour_features(v, rm) for v, rm in variables], dtype=np.float32)
    XF = np.hstack([X, T])
    oof_f = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(XF, y, groups=groups):
        m = LogTargetModel(clone(build_models()[model_name]))
        m.fit(XF[tr], y[tr], sample_weight=w[tr])
        oof_f[te] = m.predict(XF[te])
    print(f"    {XF.shape[1]} features  R²={r2_score(y, oof_f):.4f}  "
          f"MAE={mean_absolute_error(y, oof_f):,.0f} px")
    model_frontier = LogTargetModel(clone(build_models()[model_name]))
    model_frontier.fit(XF, y, sample_weight=w)

    # De zaal per rij gaat mee. Zonder dat kan de rangschikking BINNEN een zaal
    # alleen gemeten worden door de volgorde van load_data te reconstrueren uit
    # de dataset -- een aanname die stilzwijgend fout kan gaan zodra de
    # aggregatie verandert. En die meting is juist de belangrijke: de globale
    # rho gaat vooral over welke zaal makkelijk is, niet over welke indeling.
    with open(OOF_FILE, "w") as f:
        json.dump({"model":    model_name,
                   "y_true":   y.tolist(),
                   "y_pred":   best["oof"].tolist(),
                   "y_pred_frontier": oof_f.tolist(),
                   "room_key": [json.dumps(layout_key([], rm)[0]) for _v, rm in variables],
                   "n_seeds":  w.tolist()}, f)

    joblib.dump({
        "model":            model,
        "model_frontier":   model_frontier,
        "feat_len_frontier": int(XF.shape[1]),
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
