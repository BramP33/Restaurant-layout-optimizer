"""
Overnight data collector voor Restaurant Simulator GNN training.

Genereert continu random layouts, simuleert ze via headless browser,
en voegt de resultaten toe aan restaurant-sim-merged.json.

Gebruik:
    python3 collect_overnight.py
    python3 collect_overnight.py --batch 150 --seeds 3 --hours 8
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent          # data en modelbestanden staan in de repo-root


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch",  type=int,   default=150,  help="Layouts per ronde")
    p.add_argument("--seeds",  type=int,   default=3,    help="Seeds per layout")
    p.add_argument("--hours",  type=float, default=8.0,  help="Maximale looptijd in uren")
    p.add_argument("--no-retrain", action="store_true",  help="Geen hertraining tussendoor")
    return p.parse_args()


def generate_random_batch(n: int, out_path: Path) -> int:
    """Gebruik optimize_layout.py's generate_batch om N random layouts te schrijven."""
    import numpy as np
    sys.path.insert(0, str(HERE))
    from optimize_layout import generate_batch, VARIABLE_TYPES, FIXED_TABLES, TABLE_TYPES, ROOM_W, ROOM_H

    rng = np.random.default_rng()
    all_layouts = []
    attempts = 0
    while len(all_layouts) < n and attempts < 20:
        layouts, hit = generate_batch(n * 3, rng)
        all_layouts.extend(layouts)
        attempts += 1

    layouts = all_layouts[:n]
    if not layouts:
        return 0

    # Schrijf in het formaat dat validate_headless.js verwacht
    results = []
    for i, layout in enumerate(layouts):
        results.append({
            "rank":                   i + 1,
            "predicted_waiterDist":   500_000,   # onbekend — dummy
            "predicted_score":        0,
            "config": {
                "roomW": ROOM_W, "roomH": ROOM_H,
                "guests": 49, "waiters": 3,
                "tSmall": 0, "tMedium": 6, "tLarge": 2,
                "partyType": "buffet",
            },
            "tables": layout,
        })

    with open(out_path, "w") as f:
        json.dump(results, f)

    return len(layouts)


def merge_into_dataset(val_path: Path, merged_path: Path) -> int:
    """Voeg validatieresultaten toe aan de gemerged dataset."""
    if not val_path.exists():
        return 0

    with open(val_path) as f:
        new_runs = json.load(f)

    if merged_path.exists():
        with open(merged_path) as f:
            merged = json.load(f)
    else:
        merged = []

    existing_seeds = {r.get("seed") for r in merged}
    added = 0
    for r in new_runs:
        candidates = r.get("trainingRuns") or [r]
        for entry in candidates:
            seed = entry.get("seed")
            if seed not in existing_seeds:
                merged.append(entry)
                existing_seeds.add(seed)
                added += 1

    with open(merged_path, "w") as f:
        json.dump(merged, f)

    return added


def retrain_gnn():
    """Hertraining van het GNN model."""
    print("  ↻ Hertrainen GNN…")
    t0 = time.time()
    result = subprocess.run(
        ["python3.12", str(HERE / "train_gnn.py"),
         "--epochs", "600", "--patience", "60", "--lr", "5e-4"],
        cwd=HERE, capture_output=True, text=True,
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        # Haal R² op uit output
        for line in result.stdout.split("\n"):
            if "Test-set:" in line:
                print(f"  ✓ GNN klaar in {elapsed/60:.1f} min — {line.strip()}")
                return
    else:
        print(f"  ✗ GNN training mislukt ({elapsed:.0f}s): {result.stderr[-200:]}")


def main():
    args = parse_args()

    batch_file  = HERE / "_batch_layouts.json"
    val_file    = HERE / "_batch_validated.json"
    merged_path = ROOT / "restaurant-sim-merged.json"

    deadline    = time.time() + args.hours * 3600
    total_added = 0
    round_num   = 0

    print(f"Overnight data collector gestart")
    print(f"  Batch-grootte:  {args.batch} layouts/ronde")
    print(f"  Seeds:          {args.seeds} per layout")
    print(f"  Max looptijd:   {args.hours}u")
    print(f"  Dataset:        {merged_path.name}")
    print()

    # Huidige dataset-grootte
    if merged_path.exists():
        with open(merged_path) as f:
            n_start = len(json.load(f))
        print(f"  Start: {n_start} runs in dataset")
    else:
        n_start = 0

    while time.time() < deadline:
        round_num += 1
        t_round = time.time()
        remaining = (deadline - time.time()) / 3600

        print(f"\n── Ronde {round_num}  ({remaining:.1f}u resterend) ─────────────────────")

        # 1. Genereer random layouts
        print(f"  Genereren {args.batch} random layouts…", end=" ", flush=True)
        n_gen = generate_random_batch(args.batch, batch_file)
        print(f"{n_gen} gegenereerd")

        if n_gen == 0:
            print("  Geen layouts gegenereerd, opnieuw proberen…")
            time.sleep(5)
            continue

        # 2. Simuleer via headless browser
        print(f"  Simuleren ({n_gen} layouts × {args.seeds} seeds)…")
        node_result = subprocess.run(
            ["node", str(HERE / "validate_headless.js"),
             "--input",  str(batch_file),
             "--top",    str(n_gen),
             "--seeds",  str(args.seeds),
             "--out",    str(val_file)],
            cwd=HERE, capture_output=True, text=True,
        )

        if node_result.returncode != 0:
            print(f"  ✗ Headless validator mislukt:\n{node_result.stderr[-500:]}")
            time.sleep(10)
            continue

        # 3. Toevoegen aan dataset
        added = merge_into_dataset(val_file, merged_path)
        total_added += added

        with open(merged_path) as f:
            n_total = len(json.load(f))

        elapsed_round = time.time() - t_round
        rate = added / elapsed_round * 3600
        print(f"  ✓ +{added} runs  →  totaal: {n_total}  ({elapsed_round:.0f}s, ~{rate:.0f}/uur)")

        # Opruimen
        batch_file.unlink(missing_ok=True)
        val_file.unlink(missing_ok=True)

        # 4. Optioneel: hertrain GNN elke ~500 nieuwe runs
        if not args.no_retrain and total_added > 0 and total_added % 500 < args.batch * args.seeds:
            retrain_gnn()

    # Eindresultaat
    print(f"\n══ Klaar! ══════════════════════════════")
    print(f"  Rondes voltooid:  {round_num}")
    print(f"  Totaal toegevoegd: {total_added} runs")
    with open(merged_path) as f:
        n_final = len(json.load(f))
    print(f"  Dataset eindgrootte: {n_final} runs (was {n_start})")
    print(f"\n  Start nu active learning:")
    print(f"    python3 train_gnn.py --epochs 1000 --patience 100")


if __name__ == "__main__":
    main()
