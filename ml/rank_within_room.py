"""
Rangschikking BINNEN één zaal.

Waarom dit apart gemeten moet worden
------------------------------------
De globale metrieken van train_surrogate.py lopen over alle zalen heen. Zodra
de dataset veel zaalvormen bevat, wordt die globale rho vooral een maat voor
"welke zaal is makkelijk" -- een grote open zaal met weinig tafels heeft nu
eenmaal kortere looproutes dan een smalle zaal met kolommen. Dat is echte
signaal, maar het is niet het signaal waar de optimizer iets aan heeft: die
krijgt EEN zaal aangereikt en moet daarbinnen indelingen ordenen.

Op de brede dataset (418 zalen x ~30 layouts) was het verschil groot:
    rho over alle zalen heen : 0.984
    rho binnen een zaal      : 0.738 (mediaan, plafond 0.844 -- zie hieronder)

De maat die het dichtst bij de praktijk ligt is niet rho maar SPIJT: laat het
model binnen een zaal zijn beste indeling kiezen, en kijk hoeveel slechter die
is dan de werkelijk beste in diezelfde zaal.

LEES DE SPIJT NIET ALS "WAT DE OPTIMIZER KOST"
----------------------------------------------
De referentie `yy.min()` is het minimum van RUIZIGE metingen en ligt daardoor
systematisch onder de echte beste waarde. Er zit dus een bodem in: ook een
FOUTLOOS model meet hier geen 0%. Gekalibreerd op de echte zalen met hun eigen
gemeten seedruis:

    ~30 layouts per zaal  -> bodem ~2,5%   (gemeten met dit model: 1,92%)
    ~600 layouts per zaal -> bodem ~4,8%

De bodem SCHAALT MEE met het aantal kandidaten per zaal, want hoe meer
metingen, hoe extremer het gunstigste toevalstreffertje. Diep bemeten zalen
(n~600) en brede zalen (n~30) zijn daarom NIET met elkaar te vergelijken in
dezelfde tabel -- de diepe zalen zien er ongeveer twee keer zo slecht uit
puur door de bodem. Vergelijk alleen bij gelijk n, of vergelijk twee modellen
op dezelfde zalen (dan valt de bodem weg in het verschil).

Dit is precies dezelfde fout als eerder met rho(top-deciel): selecteren en
afrekenen op dezelfde ruizige grootheid meet vooral zichzelf.

Gebruik:
    python3 rank_within_room.py
    python3 rank_within_room.py --min-n 100      # alleen diep bemeten zalen
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
OOF_FILE = ROOT / "surrogate-oof.json"
POOL_FILE = ROOT / "data" / "room-pool.json"


def pool_room_keys():
    """De zaalsleutels van de diepe pool, in hetzelfde formaat als de OOF-sleutel."""
    if not POOL_FILE.exists():
        return set()
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_surrogate import layout_key
    data = json.loads(POOL_FILE.read_text())
    entries = data["pool"] if isinstance(data, dict) else data   # oud formaat: kale lijst
    return {json.dumps(layout_key([], e["room"])[0]) for e in entries}


def group_stats(y, p, idx, decile_min=8):
    """Rangschikking en spijt voor één zaal."""
    yy, pp = y[idx], p[idx]
    out = {"n": len(idx), "rho": spearmanr(yy, pp).statistic}

    # Spijt: het model kiest zijn beste, wij kijken wat dat werkelijk kost.
    best_true = yy.min()
    picked = yy[int(np.argmin(pp))]
    out["regret"] = (picked - best_true) / best_true * 100.0
    # Referentie: een blinde greep uit dezelfde verzameling.
    out["regret_random"] = (np.median(yy) - best_true) / best_true * 100.0

    # In de kopgroep moet de optimizer het verschil maken, en daar is de
    # spreiding het kleinst -- dus daar is rangschikken het lastigst.
    #
    # De ondergrens van 8 layouts maakt dit bij kleine zalen GEEN deciel: bij
    # 30 layouts is het de beste 27%, bij 600 wel de beste 10%. Daarom heet het
    # hier "kopgroep" en niet "deciel", en staat het aandeel in de uitvoer --
    # anders staan er straks twee verschillende maten onder één kopje.
    k = max(decile_min, len(idx) // 10)
    if len(idx) >= 2 * decile_min:
        top = np.argsort(yy)[:k]
        out["rho_top"] = spearmanr(yy[top], pp[top]).statistic
        out["top_frac"] = k / len(idx)
    return out


def summarise(name, rows):
    if not rows:
        print(f"  {name}: geen zalen die aan de eis voldoen")
        return
    def med(key):
        v = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        return (np.median(v), np.percentile(v, 25), np.percentile(v, 75), len(v)) if v else None

    n_layouts = int(np.median([r["n"] for r in rows]))
    fr = [r["top_frac"] for r in rows if "top_frac" in r]
    kop = f"kopgroep ({np.median(fr):.0%})" if fr else "kopgroep"
    print(f"  {name}: {len(rows)} zalen, mediaan {n_layouts} layouts per zaal")
    for key, label in (("rho", "rho, alle layouts"),
                       ("rho_top", f"rho, {kop}"),
                       ("regret", "spijt van de modelkeuze"),
                       ("regret_random", "spijt van een blinde greep")):
        s = med(key)
        if s is None:
            print(f"    {label:<28} niet meetbaar")
            continue
        m, lo, hi, n = s
        unit = "%" if "regret" in key else ""
        note = "" if n == len(rows) else f"  (over {n} zalen)"
        print(f"    {label:<28} {m:6.3f}{unit}   (p25 {lo:.3f}, p75 {hi:.3f}){note}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-n", type=int, default=25,
                    help="Minimum aantal layouts voordat een zaal meetelt")
    ap.add_argument("--oof", default=str(OOF_FILE))
    args = ap.parse_args()

    oof = json.loads(Path(args.oof).read_text())
    if "room_key" not in oof:
        raise SystemExit(
            "Deze OOF-file bevat geen room_key. Train opnieuw met de huidige "
            "train_surrogate.py; zonder zaalsleutel is deze meting een gok.")

    y = np.array(oof["y_true"], dtype=float)
    preds = {"basismodel": np.array(oof["y_pred"], dtype=float)}
    if "y_pred_frontier" in oof:
        preds["frontier-model"] = np.array(oof["y_pred_frontier"], dtype=float)

    byroom = defaultdict(list)
    for i, k in enumerate(oof["room_key"]):
        byroom[k].append(i)

    pool = pool_room_keys()
    print(f"{len(y)} layouts in {len(byroom)} zalen "
          f"(pool bevat {len(pool)} zalen)\n")

    for pname, p in preds.items():
        print(f"=== {pname} ===")
        deep, broad = [], []
        for k, idx in byroom.items():
            if len(idx) < args.min_n:
                continue
            st = group_stats(y, p, np.array(idx))
            (deep if k in pool else broad).append(st)
        summarise("diep bemeten zalen", deep)
        summarise("overige zalen", broad)
        allrows = deep + broad
        if allrows:
            print(f"    ter vergelijking, rho over ALLE zalen heen: "
                  f"{spearmanr(y, p).statistic:.3f}")
        print()


if __name__ == "__main__":
    main()
