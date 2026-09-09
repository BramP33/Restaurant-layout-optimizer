"""
Vier trainingsdoelen, afgerekend op de kopgroep van een ongeziene zaal.

De diagnose
-----------
`decompose_ranking.py` liet zien dat het volledige model op de kopgroep niets
toevoegt boven `work.sum` alleen (0,464 tegen 0,475; 4 van 8 zalen, p = 0,84),
terwijl er 0,238 tot het plafond ligt. Alle winst van de 137 extra features
gaat naar het scheiden van slechte van goede layouts, niet naar het ordenen
van goede onderling -- en dat laatste is het enige wat de optimizer doet.

De verklaring die overblijft: de verliesfunctie meet iets anders dan wij
gebruiken. Kwadratische fout in log-ruimte over een bereik van 77.000 tot
1.021.348 px wordt gedomineerd door "welke zaal is duur" en "goed versus
rampzalig". Fijne ordening aan de kop weegt daarin vrijwel niet mee.

De vier doelen
--------------
    basis        log(y)                        -- wat we nu doen
    offset       log(y) - log(work.sum)        -- leer alleen de correctie op
                                                  de natuurkundeformule
    per-zaal     log(y) - gemiddelde log(y)    -- haal het zaalverschil eruit,
                 van die zaal                     zodat elke zaal even zwaar
                                                  weegt in de loss
    rangorde     XGBRanker, groepen = zalen    -- optimaliseer de ordening zelf

`offset` en `per-zaal` geven een grootheid terug die alleen BINNEN een zaal
betekenis heeft. Dat is geen bezwaar: de optimizer gebruikt het frontier-model
uitsluitend om de kopgroep van één zaal te herordenen, en het basismodel blijft
de absolute voorspelling leveren voor de brede zoektocht en de kalibratie.

De opzet
--------
Evalueren op een ACHTERGEHOUDEN ZAAL, want zo wordt het model gebruikt: de
optimizer krijgt een zaal aangereikt die niet in de trainingsdata zat. Acht
folds, telkens één diepe zaal eruit, trainen op al het andere (inclusief de
418 brede zalen). Vergelijking is gepaard over die acht zalen.

Gebruik:
    python3 exp_frontier_target.py            # gebruikt de cache als die er is
    python3 exp_frontier_target.py --rebuild  # features opnieuw uitrekenen
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr, wilcoxon

import pathgrid as pg
import train_surrogate as ts

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "frontier-target-cache.npz"
WORK_SUM_COL = 131          # index van work.sum in de 138 basisfeatures
TOP_FRAC = 0.10


def build_cache(path):
    """Featurevector (138 basis + 14 tour) per unieke layout, plus zaalsleutel."""
    runs = json.loads((ROOT / "restaurant-sim-clean.json").read_text())
    agg = {}
    for r in runs:
        var = [t for t in r["tables"] if t["size"] != "custom"]
        if not var or len(r.get("seeds", [])) < 2:
            continue
        var.sort(key=lambda t: (-t["w"], t["x"], t["y"]))
        room = ts.run_room(r)
        key = ts.layout_key(var, room)
        w = len(r["seeds"]) if "runs" in r else 1
        a = agg.setdefault(key, {"var": var, "room": room, "d": 0.0, "n": 0})
        a["d"] += r["metrics"]["waiterDist"] * w
        a["n"] += w

    X, y, rk = [], [], []
    for i, a in enumerate(agg.values()):
        base = ts.extract_features_from_list(a["var"], a["room"])
        tour = pg.tour_features(a["var"], a["room"])
        X.append(base + list(tour))
        y.append(a["d"] / a["n"])
        rk.append(json.dumps(ts.layout_key([], a["room"])[0]))
        if (i + 1) % 2000 == 0:
            print(f"  {i+1} layouts…", flush=True)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float64)
    np.savez_compressed(path, X=X, y=y, rk=np.array(rk))
    return X, y, rk


def params():
    """Dezelfde hyperparameters als het gekozen model in train_surrogate."""
    return dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, n_jobs=-1, random_state=0)


def fit_predict(kind, Xtr, ytr, gtr, Xte):
    """Traint één doelvariant en geeft een score terug waarvan LAAG = beter."""
    if kind == "basis":
        m = xgb.XGBRegressor(**params())
        m.fit(Xtr, np.log(ytr))
        return m.predict(Xte)

    if kind == "offset":
        # De formule is al een schatting van y; leer alleen de verhouding.
        off = np.log(np.maximum(Xtr[:, WORK_SUM_COL], 1.0))
        m = xgb.XGBRegressor(**params())
        m.fit(Xtr, np.log(ytr) - off)
        # Bij voorspellen weer optellen: binnen een zaal verschilt work.sum
        # per layout, dus die term draagt echt bij aan de ordening.
        return m.predict(Xte) + np.log(np.maximum(Xte[:, WORK_SUM_COL], 1.0))

    if kind == "per-zaal":
        ly = np.log(ytr)
        mu = defaultdict(list)
        for g, v in zip(gtr, ly):
            mu[g].append(v)
        mu = {g: float(np.mean(v)) for g, v in mu.items()}
        m = xgb.XGBRegressor(**params())
        m.fit(Xtr, ly - np.array([mu[g] for g in gtr]))
        return m.predict(Xte)      # gecentreerd; alleen ordening telt

    if kind == "rangorde":
        # XGBRanker wil aaneengesloten groepen en een relevantielabel waarbij
        # HOOG = beter. Binnen elke zaal wordt de rangorde omgezet naar 0-31.
        order = np.argsort(gtr, kind="stable")
        Xs, ys, gs = Xtr[order], ytr[order], np.array(gtr)[order]
        sizes, labels = [], np.zeros(len(ys))
        start = 0
        for g in dict.fromkeys(gs):
            idx = np.where(gs == g)[0]
            sizes.append(len(idx))
            r = np.argsort(np.argsort(ys[idx]))          # 0 = laagste dist
            labels[idx] = 31 - np.floor(r / max(len(idx) - 1, 1) * 31)
            start += len(idx)
        m = xgb.XGBRanker(objective="rank:pairwise", **params())
        m.fit(Xs, labels, group=sizes)
        return -m.predict(Xte)     # hoog relevant -> lage dist, dus omdraaien

    raise ValueError(kind)


def evaluate(y, p):
    n = len(y)
    k = max(6, int(n * TOP_FRAC))
    top = np.argsort(y)[:k]
    best = y.min()
    return (spearmanr(y, p).statistic,
            spearmanr(y[top], p[top]).statistic,
            (y[int(np.argmin(p))] - best) / best * 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if CACHE.exists() and not args.rebuild:
        d = np.load(CACHE, allow_pickle=True)
        X, y, rk = d["X"], d["y"], list(d["rk"])
        print(f"cache geladen: {X.shape[0]} layouts, {X.shape[1]} features")
    else:
        print("features uitrekenen (dit duurt enkele minuten)…", flush=True)
        X, y, rk = build_cache(CACHE)
        print(f"cache gebouwd: {X.shape[0]} layouts, {X.shape[1]} features")

    # De diepe zalen zijn die met genoeg layouts om een kopgroep te hebben.
    counts = defaultdict(int)
    for k in rk:
        counts[k] += 1
    deep = sorted([k for k, c in counts.items() if c >= 400])
    print(f"{len(deep)} diep bemeten zalen om op achter te houden\n")

    kinds = ["basis", "offset", "per-zaal", "rangorde"]
    res = {k: defaultdict(list) for k in kinds}
    res["formule"] = defaultdict(list)

    rk_arr = np.array(rk)
    for f, room_key in enumerate(deep, 1):
        te = np.where(rk_arr == room_key)[0]
        tr = np.where(rk_arr != room_key)[0]
        name = json.loads(room_key)
        label = f"{name[0]} {name[1]}x{name[2]}"
        print(f"[{f}/{len(deep)}] {label}: {len(te)} test, {len(tr)} train", flush=True)

        # Referentie: de kale formule, zonder enige training.
        a, b, c = evaluate(y[te], X[te, WORK_SUM_COL])
        res["formule"]["rho"].append(a); res["formule"]["top"].append(b)
        res["formule"]["regret"].append(c)

        for kind in kinds:
            p = fit_predict(kind, X[tr], y[tr], list(rk_arr[tr]), X[te])
            a, b, c = evaluate(y[te], p)
            res[kind]["rho"].append(a); res[kind]["top"].append(b)
            res[kind]["regret"].append(c)
            print(f"      {kind:<10} rho {a:.3f}  kopgroep {b:.3f}  spijt {c:5.2f}%",
                  flush=True)

    print("\n" + "=" * 62)
    print("Mediaan over de acht achtergehouden zalen:\n")
    print("{:<12} {:>10} {:>13} {:>10}".format("", "rho alles", "rho kopgroep", "spijt"))
    for kind in ["formule"] + kinds:
        print("{:<12} {:>10.3f} {:>13.3f} {:>9.2f}%".format(
            kind, np.median(res[kind]["rho"]), np.median(res[kind]["top"]),
            np.median(res[kind]["regret"])))

    print("\nGepaard tegen 'basis' (acht zalen, Wilcoxon):")
    base = res["basis"]
    for kind in kinds[1:]:
        for metric, lbl in (("top", "kopgroep"), ("regret", "spijt")):
            a = np.array(base[metric]); b = np.array(res[kind][metric])
            better = int((b > a).sum()) if metric == "top" else int((b < a).sum())
            p = wilcoxon(b, a).pvalue if not np.allclose(a, b) else 1.0
            print(f"  {kind:<10} {lbl:<9} {np.mean(b - a):+.4f}"
                  f"   beter in {better}/8   p={p:.3f}")


if __name__ == "__main__":
    main()
