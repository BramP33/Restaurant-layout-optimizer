#!/usr/bin/env bash
# Zet een verse Ubuntu-VM klaar om de simulator headless te draaien.
#
# Eén keer draaien na het aanmaken van de machine:
#     bash ml/azure_setup.sh
#
# Installeert Node, Playwright met Chromium (inclusief systeembibliotheken die
# een kale server niet heeft) en een Python-venv met de pipeline-dependencies.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "== 1/4  Systeempakketten =="
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip curl ca-certificates

echo "== 2/4  Node.js 22 =="
if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
  sudo apt-get install -y -qq nodejs
fi
node --version

echo "== 3/4  Python-omgeving =="
[ -d .venv ] || python3 -m venv .venv
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -r requirements.txt
./.venv/bin/python -c "import numpy, sklearn, xgboost; print('  python ok')"

echo "== 4/4  Playwright + Chromium =="
cd ml
npm install --silent
# --with-deps haalt de systeembibliotheken op die Chromium nodig heeft;
# zonder dit start de browser op een kale server niet.
npx playwright install --with-deps chromium
cd "$REPO_DIR"

echo
echo "Klaar. Proefrun van 6 minuten:"
echo "    cd ml && ../.venv/bin/python collect_parallel.py --workers 2 --hours 0.1"
