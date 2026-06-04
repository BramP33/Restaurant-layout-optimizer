"""
Voegt meerdere batch-JSON exports samen tot één bestand.

Gebruik:
    python3 merge_datasets.py                        # alle restaurant-sim-batch-*.json in huidige map
    python3 merge_datasets.py bestand1.json bestand2.json ...
    python3 merge_datasets.py --out gecombineerd.json
"""

import json
import sys
import glob
from pathlib import Path

HERE = Path(__file__).parent


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    args = sys.argv[1:]
    out_file = HERE / "restaurant-sim-merged.json"

    # --out overschrijft de uitvoernaam
    if "--out" in args:
        idx = args.index("--out")
        out_file = Path(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    # Bestanden uit argumenten of glob
    if args:
        files = [Path(a) for a in args]
    else:
        files = sorted(HERE.glob("restaurant-sim-batch-*.json"))

    if not files:
        print("Geen bestanden gevonden. Geef bestanden mee of zet ze in dezelfde map.")
        sys.exit(1)

    all_runs = []
    seen_keys = set()

    for f in files:
        if not f.exists():
            print(f"  SKIP (niet gevonden): {f}")
            continue
        runs = load(f)
        before = len(all_runs)
        for r in runs:
            # Dedupliceer op seed (eerste seed als identifier)
            key = r.get("seed") or r.get("seeds", [None])[0]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            all_runs.append(r)
        added = len(all_runs) - before
        print(f"  {f.name}: {len(runs)} runs, {added} toegevoegd ({len(runs)-added} dubbel)")

    print(f"\nTotaal: {len(all_runs)} unieke layouts")
    with open(out_file, "w") as f:
        json.dump(all_runs, f)          # compacte opslag (geen indent)
    print(f"Opgeslagen: {out_file}")


if __name__ == "__main__":
    main()
