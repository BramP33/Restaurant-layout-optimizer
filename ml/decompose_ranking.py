"""
Waar zit de rangschikkingskracht binnen een zaal, en hoeveel is er nog te halen?

De aanleiding
-------------
96% van de feature-importance van het basismodel zit in twee sommen:
`work.sum = sum(stoelen * A*-padafstand)` en `paths.sum = sum(A*-padafstand)`.
Dat is de witte-doos formule uit fase 3 van het plan. Het model is dus in
hoofdzaak een natuurkundeschatting, met de overige 136 features als marge.

Dit script legt een ladder aan om te zien hoeveel elke stap toevoegt, gemeten
BINNEN een zaal -- want dat is het enige wat de optimizer doet:

    L1  work.sum alleen        de kale formule, één getal
    L2  lineair op padfeatures alleen A*-meetkunde, geen ruwe coordinaten
    L3  basismodel (138)       de volle featureset
    L4  frontier-model (152)   plus de tour-features
    --  plafond                wat een FOUTLOOS model haalt bij deze seedruis

Binnen een zaal is elke monotone transformatie van work.sum equivalent voor
rangschikking, dus L1 is meteen de eerlijke witte-doos baseline: a*work+b
ordent identiek.

Gemeten op de diepe set (8 zalen x ~600 layouts x 9 seeds), want daar ligt het
plafond hoog genoeg om verschillen te kunnen zien. Bij 3 seeds zat de kopgroep-
metriek op zijn plafond en kon geen enkele verbetering zichtbaar worden.

Gebruik:
    python3 decompose_ranking.py
"""

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

import pathgrid as pg
import train_surrogate as ts

ROOT = Path(__file__).parent.parent
SHARDS = str(ROOT / "data" / "shards" / "shard-diep9-*.json")
OOF = ROOT / "surrogate-oof.json"

WORK_SUM  = 17      # index in path_features
PATHS_SUM = 4
TOP_FRAC  = 0.10


def load_deep():
    """De diepe layouts met hun negen losse metingen."""
    by = {}
    for f in sorted(glob.glob(SHARDS)):
        for run in json.load(open(f)):
            var = [t for t in run["tables"] if t["size"] != "custom"]
            if not var:
                continue
            var.sort(key=lambda t: (-t["w"], t["x"], t["y"]))
            room = ts.run_room(run)
            key = ts.layout_key(var, room)
            e = by.setdefault(key, {"var": var, "room": room, "vals": {}})
            e["vals"][run["seeds"].index(run["seed"])] = run["metrics"]["waiterDist"]
    return [e for e in by.values() if len(e["vals"]) == 9]


def ceiling(groups, rng, n_rep=60):
    """Wat haalt een foutloos model bij deze seedruis en dit aantal seeds?"""
    ca, ct = [], []
    for vals in groups:
        A = np.array(vals, dtype=float)
        m = A.mean(1)
        within  = A.var(1, ddof=1).mean()
        between = max(m.var(ddof=1) - within / 9, 0.0)
        n = len(A)
        k = max(6, int(n * TOP_FRAC))
        for _ in range(n_rep):
            truth = rng.normal(0, between ** 0.5, n)
            obs   = truth + rng.normal(0, (within / 9) ** 0.5, n)
            ca.append(spearmanr(truth, obs).statistic)
            top = np.argsort(obs)[:k]
            ct.append(spearmanr(truth[top], obs[top]).statistic)
    return float(np.mean(ca)), float(np.mean(ct))


def scores(y, p):
    """rho over alles, rho in de kopgroep, en spijt van de keuze."""
    n = len(y)
    k = max(6, int(n * TOP_FRAC))
    top = np.argsort(y)[:k]
    best = y.min()
    return (spearmanr(y, p).statistic,
            spearmanr(y[top], p[top]).statistic,
            (y[int(np.argmin(p))] - best) / best * 100.0)


def main():
    rows = load_deep()
    print(f"{len(rows)} diepe layouts met negen seeds\n")

    y   = np.array([np.mean(list(e["vals"].values())) for e in rows])
    pf  = np.array([pg.path_features(e["var"], e["room"]) for e in rows])
    rk  = [json.dumps(ts.layout_key([], e["room"])[0]) for e in rows]

    # L2: lineair model op alleen de padfeatures, out-of-fold.
    g   = np.arange(len(y))
    l2  = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(pf, y, groups=g):
        m = RidgeCV(alphas=np.logspace(-3, 3, 13))
        m.fit(pf[tr], np.log(y[tr]))
        l2[te] = m.predict(pf[te])

    # L3/L4 uit de OOF-file, gekoppeld op (zaal, echte waarde).
    oof = json.loads(OOF.read_text())
    idx = defaultdict(dict)
    for yt, yp, fr, k in zip(oof["y_true"], oof["y_pred"],
                             oof["y_pred_frontier"], oof["room_key"]):
        idx[k][round(yt, 3)] = (yp, fr)
    l3 = np.full(len(y), np.nan)
    l4 = np.full(len(y), np.nan)
    for i, (k, yv) in enumerate(zip(rk, y)):
        hit = idx.get(k, {}).get(round(yv, 3))
        if hit:
            l3[i], l4[i] = hit
    ok = np.isfinite(l3)
    print(f"gekoppeld aan het getrainde model: {ok.sum()}/{len(y)}\n")

    byroom = defaultdict(list)
    for i, k in enumerate(rk):
        byroom[k].append(i)

    ladders = {
        "L1  work.sum (formule)": pf[:, WORK_SUM],
        "L2  lineair op padfeatures": l2,
        "L3  basismodel (138)": l3,
        "L4  frontier-model (152)": l4,
    }

    res = defaultdict(lambda: defaultdict(list))
    groups_vals = []
    for k, ii in byroom.items():
        ii = np.array([i for i in ii if ok[i]])
        if len(ii) < 100:
            continue
        groups_vals.append([[rows[i]["vals"][s] for s in range(9)] for i in ii])
        for name, pred in ladders.items():
            a, b, c = scores(y[ii], pred[ii])
            res[name]["rho"].append(a)
            res[name]["top"].append(b)
            res[name]["regret"].append(c)

    n_rooms = len(groups_vals)
    ca, ct = ceiling(groups_vals, np.random.default_rng(5))

    print(f"Binnen een zaal, {n_rooms} zalen, mediaan over de zalen:\n")
    print("{:<28} {:>10} {:>12} {:>10}".format(
        "", "rho alles", "rho kopgroep", "spijt"))
    prev = None
    for name in ladders:
        r  = np.median(res[name]["rho"])
        t  = np.median(res[name]["top"])
        rg = np.median(res[name]["regret"])
        delta = "" if prev is None else f"   (+{r - prev:.3f})"
        print("{:<28} {:>10.3f} {:>12.3f} {:>9.2f}%{}".format(name, r, t, rg, delta))
        prev = r
    print("{:<28} {:>10.3f} {:>12.3f}".format("--  plafond (foutloos)", ca, ct))
    print()

    l4r, l4t = np.median(res["L4  frontier-model (152)"]["rho"]), \
               np.median(res["L4  frontier-model (152)"]["top"])
    l1r, l1t = np.median(res["L1  work.sum (formule)"]["rho"]), \
               np.median(res["L1  work.sum (formule)"]["top"])
    print("Wat de 137 features boven de kale formule toevoegen:")
    print(f"   rho alles    {l1r:.3f} -> {l4r:.3f}   (+{l4r-l1r:.3f})")
    print(f"   rho kopgroep {l1t:.3f} -> {l4t:.3f}   (+{l4t-l1t:.3f})")
    print()
    print("Wat er daarna nog ligt tot het plafond:")
    print(f"   rho alles    {l4r:.3f} -> {ca:.3f}   ({ca-l4r:+.3f})")
    print(f"   rho kopgroep {l4t:.3f} -> {ct:.3f}   ({ct-l4t:+.3f})")


if __name__ == "__main__":
    main()
