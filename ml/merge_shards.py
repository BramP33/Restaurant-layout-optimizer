"""
Voegt de shard-bestanden van collect_parallel.py samen tot één dataset.

Elke werker schrijft zijn eigen shard, zodat parallelle processen elkaar niet
overschrijven. Dit script plakt ze aan elkaar, gooit dubbele runs eruit en
rapporteert hoeveel er onbruikbaar waren.

Gebruik:
    python3 merge_shards.py
    python3 merge_shards.py --out ../restaurant-sim-clean.json --keep-invalid
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
SHARD_DIR = ROOT / "data" / "shards"


def layout_key(run):
    var = [t for t in run.get("tables", []) if t.get("size") != "custom"]
    return tuple(sorted((round(t["x"], 1), round(t["y"], 1), t["size"], t["rotation"])
                        for t in var))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "restaurant-sim-clean.json"))
    p.add_argument("--shards", default=str(SHARD_DIR))
    p.add_argument("--keep-invalid", action="store_true",
                   help="Behoud runs met een onbereikbare bar of tafel")
    args = p.parse_args()

    shard_dir = Path(args.shards)
    files = sorted(shard_dir.glob("shard*.json"))
    if not files:
        print(f"Geen shards gevonden in {shard_dir}")
        return

    runs, seen = [], set()
    invalid = dupes = failed = 0
    for f in files:
        try:
            entries = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"  overgeslagen (onvolledig geschreven): {f.name}")
            continue
        for r in entries:
            key = (layout_key(r), r.get("seed"))
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            m = r.get("metrics", {})
            # BatchRunner middelt metrics numeriek, dus false komt aan als 0.
            #
            # Twee filters, en de tweede is geen luxe. layoutValid is een
            # statische toets vooraf; waiterPathFailures telt wat er tijdens de
            # run echt misging. Een ober die geen route vindt blijft staan en
            # legt nul afstand af -- dat is de goedkoopste layout die er
            # bestaat, en precies de exploit die we dichttimmeren. Alles wat
            # een van beide toetsen niet haalt, hoort niet in de trainingsdata.
            if not m.get("layoutValid", 1):
                invalid += 1
                if not args.keep_invalid:
                    continue
            elif m.get("waiterPathFailures", 0):
                failed += 1
                if not args.keep_invalid:
                    continue
            runs.append(r)

    out = Path(args.out)
    out.write_text(json.dumps(runs))
    layouts = len({layout_key(r) for r in runs})

    print(f"Shards samengevoegd: {len(files)} bestanden")
    print(f"  runs bewaard      : {len(runs):,}  ({layouts:,} unieke layouts)")
    print(f"  duplicaten weg    : {dupes:,}")
    print(f"  ongeldige layouts : {invalid:,} "
          f"({'behouden' if args.keep_invalid else 'verwijderd'})")
    print(f"  oberroute mislukt : {failed:,} "
          f"({'behouden' if args.keep_invalid else 'verwijderd'})")
    print(f"  geschreven naar   : {out}")


if __name__ == "__main__":
    main()
