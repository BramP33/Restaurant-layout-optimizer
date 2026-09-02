#!/usr/bin/env bash
# Nachtelijke dataverzameling. Werkt vanuit elke map en gebruikt de venv van
# deze repo, zodat hetzelfde script lokaal en op een VM draait.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$REPO_DIR/.venv/bin/python"
WORKERS="${1:-$(( $(nproc) / 2 ))}"
HOURS="${2:-8}"

echo "=== Dataverzameling: $WORKERS werkers, $HOURS uur ==="
cd "$REPO_DIR/ml"
"$PYTHON" collect_parallel.py --workers "$WORKERS" --hours "$HOURS"
echo
echo "=== Shards samenvoegen ==="
"$PYTHON" merge_shards.py
