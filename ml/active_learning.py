"""
Actief leren loop voor de layout optimizer.

Workflow:
  1. Laad optimizer-results.json (gevalideerde resultaten uit de browser)
  2. Voeg toe aan de gemerged dataset
  3. Hertraining van het surrogate model
  4. Draai optimizer opnieuw voor betere kandidaten
  5. Herhaal

Gebruik na validatie in de browser:
    python3 active_learning.py --validated optimizer-results-validated.json
"""

import json
import sys
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent          # data en modelbestanden staan in de repo-root


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def main():
    args = sys.argv[1:]
    validated_path = None
    i = 0
    while i < len(args):
        if args[i] == "--validated" and i + 1 < len(args):
            validated_path = Path(args[i + 1])
            i += 2
        else:
            i += 1

    # De schone set is sinds de her-collectie de canonieke dataset, en
    # train_surrogate.py pakt hem ook als eerste. Naar restaurant-sim-merged
    # schrijven en daarna trainen betekende dat deze lus zijn eigen nieuwe
    # runs weggooide: hij vulde het ene bestand en trainde op het andere.
    dataset_path = ROOT / "restaurant-sim-clean.json"

    if validated_path and validated_path.exists():
        print(f"Stap 1 — gevalideerde resultaten toevoegen: {validated_path}")
        merged   = load_json(dataset_path)
        new_runs = load_json(validated_path)

        # Normaliseer: haal trainingRuns op als aanwezig (validate_headless.js formaat),
        # anders gebruik de runs direct (batch-export formaat).
        added = 0
        existing_seeds = {r.get("seed") for r in merged}
        for r in new_runs:
            candidates = r.get("trainingRuns") or [r]
            for entry in candidates:
                seed = entry.get("seed")
                if seed not in existing_seeds:
                    merged.append(entry)
                    existing_seeds.add(seed)
                    added += 1

        save_json(dataset_path, merged)
        print(f"  {added} nieuwe runs toegevoegd (totaal: {len(merged)})")
    else:
        print("Geen gevalideerd bestand opgegeven — alleen hertraining en optimizer.")

    print("\nStap 2 — model hertrainen…")
    # Probeer GNN training; fallback naar RF als gnn_layout niet beschikbaar
    train_script = HERE / "train_gnn.py"
    if not train_script.exists():
        train_script = HERE / "train_surrogate.py"
    try:
        import torch  # noqa — check of PyTorch beschikbaar is
    except ImportError:
        train_script = HERE / "train_surrogate.py"
    # Expliciet meegeven waar de lus zojuist naartoe geschreven heeft, zodat
    # een wijziging in de zoekvolgorde van _find_data() dit nooit meer stil
    # kan omleiden.
    train_args = ([] if train_script.name == "train_gnn.py"
                  else ["--data", str(dataset_path)])
    result = subprocess.run(
        [sys.executable, str(train_script), *train_args],
        cwd=HERE, capture_output=False
    )
    if result.returncode != 0:
        print("GNN trainingsfout, fallback naar RF…")
        result = subprocess.run(
            [sys.executable, str(HERE / "train_surrogate.py"),
             "--data", str(dataset_path)],
            cwd=HERE, capture_output=False
        )
        if result.returncode != 0:
            print("Trainingsfout.")
            return

    print("\nStap 3 — optimizer draaien…")
    subprocess.run(
        [sys.executable, str(HERE / "optimize_layout.py"), "--candidates", "500000"],
        cwd=HERE, capture_output=False
    )

    print("\nKlaar. Laad optimizer-results.json in de simulator om te valideren.")


if __name__ == "__main__":
    main()
