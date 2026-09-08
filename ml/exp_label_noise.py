"""
Helpen schonere labels het model, of middelt het de ruis toch al weg?

De vraag
--------
De seedruis in deze simulator is groot ten opzichte van het signaal. Gemeten
op de volle dataset (418 brede zalen): seedruis sd 18.623 px per run, spreiding
tussen layouts binnen een zaal sd 19.678 px. Een target van drie seeds houdt
daarmee ~10.752 px labelruis over. Op de diepe set (4.920 layouts x 9 seeds)
ligt de ruis hoger: 21.891 px per run, 12.639 px op een 3-seed gemiddelde.

De modelfout is via Monte-Carlo-inversie op ~9.600 px geschat, dus labelruis en
modelfout zijn van dezelfde orde -- niet, zoals eerst gedacht, de labelruis
duidelijk groter. Die eerdere schatting (17.000 / 14.500 / 9.800 / 6.371 px)
kwam uit een steekproef van 240 layouts in 6 zalen en was structureel te laag.

Er staat een eerder negatief resultaat tegenover: op de eenzaal-dataset
rangschikten 1-seed targets even goed als 3-seed, omdat de leerder labelruis
uitmiddelt over 10.240 layouts. Dat was echter één zaal, globale rangschikking
en veel meer layouts per zaal. Deze test zit in het andere regime: ~590 layouts
per zaal en rangschikking BINNEN een zaal.

UITKOMST (8 sep 2026), zodat niemand hem verkeerd samenvat:
    rho   B - A = +0,0079  (95%-BI +0,0011 .. +0,0148, gepaarde t, p = 0,029)
    spijt B - A = -0,15 pp (95%-BI -0,36 .. +0,06, p = 0,13)  -- geen effect
    bij GELIJK simulatiebudget (halve layouts, dubbele seeds): beide nul
    (rho -0,003 p = 0,30; spijt -0,18 pp p = 0,17)
    onderscheidend vermogen (80%, n=8 zalen): 0,0095 rho of 0,24 pp spijt

Meer seeds helpt dus WEL een beetje voor de rangschikking, maar bij een vast
simulatiebudget wordt die winst precies opgegeten door de layouts die je ervoor
inlevert. Conclusie: 3 seeds aanhouden -- niet omdat extra seeds niets doen,
maar omdat ze niets opleveren per bestede simulatie.

De opzet
--------
Elke layout in de diepe run is met negen seeds gemeten, elk als eigen record.
Die negen worden gesplitst in drie disjuncte blokken:

    labels A  = gemiddelde van seed 1-3   (het huidige regime)
    labels B  = gemiddelde van seed 1-6   (dubbel zoveel seeds)
    waarheid  = gemiddelde van seed 7-9   (nooit gebruikt om te trainen)

Beide modellen worden afgerekend tegen dezelfde achtergehouden waarheid. Die
waarheid is zelf ruizig, dus de absolute rho valt laag uit -- maar de demping
is voor A en B identiek, dus het VERSCHIL is eerlijk. Dat is precies de
gepaarde opzet die dit project nodig heeft om kleine effecten te zien.

Gebruik:
    python3 exp_label_noise.py
"""

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

import train_surrogate as ts
from log_target import LogTargetModel

ROOT = Path(__file__).parent.parent
SHARDS = str(ROOT / "data" / "shards" / "shard-diep9-*.json")
CACHE = ROOT / "data" / "exp-label-noise-oof.npy"
K_POOL = 50      # grootte van de kandidatenlijst waaruit het model mag kiezen
N_DRAW = 400     # trekkingen per zaal


def load_per_seed():
    """Verzamelt per layout de negen metingen, in een vaste seedvolgorde."""
    by = {}
    for f in sorted(glob.glob(SHARDS)):
        for run in json.load(open(f)):
            var = [t for t in run["tables"] if t["size"] != "custom"]
            if not var:
                continue
            var.sort(key=lambda t: (-t["w"], t["x"], t["y"]))
            room = ts.run_room(run)
            key = ts.layout_key(var, room)
            # De volgorde moet aan de seed hangen en niet aan de volgorde
            # waarin shards toevallig ingelezen worden, anders zit er in blok
            # A soms een andere seed dan in blok B.
            order = run["seeds"].index(run["seed"])
            e = by.setdefault(key, {"var": var, "room": room, "vals": {}})
            e["vals"][order] = run["metrics"]["waiterDist"]
    return [e for e in by.values() if len(e["vals"]) == 9]


def main():
    rows = load_per_seed()
    print(f"{len(rows)} layouts met alle negen seeds\n")

    X, gA, gB, truth, rooms = [], [], [], [], []
    for e in rows:
        v = [e["vals"][i] for i in range(9)]
        X.append(ts.extract_features_from_list(e["var"], e["room"]))
        gA.append(np.mean(v[0:3]))
        gB.append(np.mean(v[0:6]))
        truth.append(np.mean(v[6:9]))
        rooms.append(json.dumps(ts.layout_key([], e["room"])[0]))

    X = np.array(X, dtype=np.float32)
    gA, gB, truth = map(lambda a: np.array(a, dtype=float), (gA, gB, truth))
    print(f"featurevector: {X.shape[1]} kolommen")
    print(f"labelruis A (3 seeds): sd {np.std(gA - truth):,.0f} px "
          f"| B (6 seeds): sd {np.std(gB - truth):,.0f} px  "
          "(t.o.v. de achtergehouden waarheid)\n")

    name = "XGBoost"
    groups = np.arange(len(truth))
    oof = {}
    for label, y in (("A: 3 seeds", gA), ("B: 6 seeds", gB)):
        p = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups=groups):
            m = LogTargetModel(clone(ts.build_models()[name]))
            m.fit(X[tr], y[tr])
            p[te] = m.predict(X[te])
        oof[label] = p

    byroom = defaultdict(list)
    for i, r in enumerate(rooms):
        byroom[r].append(i)

    print("Afgerekend tegen de ACHTERGEHOUDEN waarheid (seed 7-9), per zaal:\n")
    print("{:<12} {:>16} {:>16}".format("zaal", "A: 3 seeds", "B: 6 seeds"))
    pa, pb = [], []
    for r, idx in sorted(byroom.items(), key=lambda x: -len(x[1])):
        idx = np.array(idx)
        a = spearmanr(truth[idx], oof["A: 3 seeds"][idx]).statistic
        b = spearmanr(truth[idx], oof["B: 6 seeds"][idx]).statistic
        pa.append(a); pb.append(b)
        kind = json.loads(r)[0]
        print("{:<12} {:>16.3f} {:>16.3f}".format(kind[:12], a, b))

    pa, pb = np.array(pa), np.array(pb)
    print("\n{:<12} {:>16.3f} {:>16.3f}".format("gemiddeld", pa.mean(), pb.mean()))
    d = pb - pa
    print(f"\nverschil B - A: {d.mean():+.4f} (per zaal: "
          + ", ".join(f"{x:+.3f}" for x in d) + ")")
    if len(d) >= 6:
        try:
            print(f"Wilcoxon gepaard: p = {wilcoxon(pb, pa).pvalue:.4f}")
        except ValueError as exc:
            print(f"Wilcoxon niet te berekenen: {exc}")

    np.save(CACHE, np.vstack([truth, oof["A: 3 seeds"], oof["B: 6 seeds"]]))
    (CACHE.with_suffix(".rooms.json")).write_text(json.dumps(rooms))

    # Spijt telt zwaarder dan rho: dat is wat een misser de optimizer kost.
    # Eén keuze per zaal zijn acht getallen -- veel te weinig. Dus herhaald
    # trekken uit elke zaal, met VOOR BEIDE MODELLEN DEZELFDE deelverzameling,
    # zodat het verschil gepaard gemeten wordt. Dit bootst na wat de optimizer
    # doet: uit een kandidatenlijst de beste kiezen.
    print(f"\nSpijt bij een kandidatenlijst van {K_POOL} layouts, "
          f"{N_DRAW} trekkingen per zaal (gepaard):")
    rng = np.random.default_rng(3)
    ra, rb, rblind = [], [], []
    for r, idx in byroom.items():
        idx = np.array(idx)
        if len(idx) < K_POOL:
            continue
        t = truth[idx]
        for _ in range(N_DRAW):
            s = rng.choice(len(idx), K_POOL, replace=False)
            best = t[s].min()
            ra.append((t[s][int(np.argmin(oof["A: 3 seeds"][idx][s]))] - best) / best * 100)
            rb.append((t[s][int(np.argmin(oof["B: 6 seeds"][idx][s]))] - best) / best * 100)
            rblind.append((np.median(t[s]) - best) / best * 100)
    ra, rb = np.array(ra), np.array(rb)
    print(f"  {'A: 3 seeds':<14} {ra.mean():5.2f}%   (mediaan {np.median(ra):.2f}%)")
    print(f"  {'B: 6 seeds':<14} {rb.mean():5.2f}%   (mediaan {np.median(rb):.2f}%)")
    print(f"  {'blinde greep':<14} {np.mean(rblind):5.2f}%")
    d = ra - rb
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"\n  winst van B: {d.mean():+.3f} procentpunt "
          f"(95%-BI {d.mean()-1.96*se:+.3f} .. {d.mean()+1.96*se:+.3f})")
    print("  let op: de trekkingen binnen een zaal overlappen, dus dit BI is "
          "te smal;\n  behandel het als een indicatie, niet als een toets.")


if __name__ == "__main__":
    main()
