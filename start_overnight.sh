#!/bin/bash
cd "/home/bram/Restaurant Simulator"
# Gebruik python3.12 (heeft numpy/torch) i.p.v. conda's python3
PYTHON=python3.12
echo "=== Overnight data collector ==="
echo "Verwacht ~19.000 nieuwe runs in 8 uur"
echo "Druk Ctrl+C om te stoppen"
echo ""
$PYTHON collect_overnight.py --batch 150 --seeds 3 --hours 8
echo ""
echo "=== Klaar! Hertraining starten ==="
$PYTHON train_gnn.py --epochs 1000 --patience 100 --lr 5e-4 --hidden 64
