"""
GNN trainingsloop voor Restaurant Simulator layout optimizer.

Laadt restaurant-sim-merged.json, bouwt PyG-grafen, traint RestaurantGNN op GPU
(GTX 1060 6GB) met AdamW + CosineAnnealingLR + Huber-loss + early stopping.

Gebruik:
    python3 train_gnn.py
    python3 train_gnn.py --epochs 500 --lr 0.001
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent          # data en modelbestanden staan in de repo-root


# ── Argumenten ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=500)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight-decay",type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=50)
    p.add_argument("--hidden",      type=int,   default=64)
    p.add_argument("--heads",       type=int,   default=4)
    p.add_argument("--layers",      type=int,   default=3)
    p.add_argument("--dropout",     type=float, default=0.25)
    p.add_argument("--data",        type=str,   default=None)
    p.add_argument("--out",         type=str,   default=None)
    return p.parse_args()


# ── Dataset laden ─────────────────────────────────────────────────────────────

def _find_data():
    merged = ROOT / "restaurant-sim-merged.json"
    if merged.exists():
        return merged
    files = sorted(ROOT.glob("restaurant-sim-batch-*.json"))
    if files:
        return files[-1]
    raise FileNotFoundError("Geen dataset gevonden (restaurant-sim-merged.json).")


def load_dataset(path: Path):
    with open(path) as f:
        runs = json.load(f)

    # Alleen multi-seed layouts (minder ruis)
    multi = [r for r in runs if len(r.get("seeds", [])) > 1]
    if len(multi) >= 100:
        runs = multi
        print(f"  Gefilterd op multi-seed: {len(runs)} layouts")

    records = []
    for run in runs:
        tables = run.get("tables", [])
        if not tables:
            continue
        y = run["metrics"]["waiterDist"]
        if y <= 0:
            continue
        config = run.get("config", {})
        records.append({"tables": tables, "config": config, "waiterDist": y})

    return records


# ── Datasets en DataLoaders ───────────────────────────────────────────────────

def make_splits(records, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records))
    n   = len(idx)
    n_val  = max(1, int(n * 0.10))
    n_test = max(1, int(n * 0.10))
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx, val_idx, test_idx


def _augment(tables, noise_px=15):
    """Kleine positie-jitter per tafel (behoudt schaal van waiterDist)."""
    import random
    aug = []
    for t in tables:
        if t.get("size") == "custom":
            aug.append(t)
            continue
        aug.append({**t,
                    "x": t["x"] + random.uniform(-noise_px, noise_px),
                    "y": t["y"] + random.uniform(-noise_px, noise_px)})
    return aug


class LayoutDataset:
    def __init__(self, records, indices, build_graph_fn, augment=False, cache=False):
        self.records     = records
        self.indices     = indices
        self.build_graph = build_graph_fn
        self.augment     = augment
        self._cache      = None
        if cache and not augment:
            self._cache = [
                build_graph_fn(records[idx]["tables"], records[idx]["config"],
                               target=records[idx]["waiterDist"])
                for idx in indices
            ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self._cache is not None:
            g = self._cache[i]
            return fast_augment_graph(g) if self.augment else g
        rec    = self.records[self.indices[i]]
        tables = rec["tables"]
        if self.augment:
            tables = _augment(tables)
        return self.build_graph(tables, rec["config"], target=rec["waiterDist"])


def collate_fn(batch):
    from torch_geometric.data import Batch as PyGBatch
    return PyGBatch.from_data_list(batch)


def fast_augment_graph(data, noise_px=15):
    """Perturb variable-table positions in a cached graph — pure tensor ops."""
    import torch
    data = data.clone()
    is_var = data.x[:, 8].bool()
    n_var  = int(is_var.sum())
    if n_var == 0:
        return data
    room_w = float(data.u[0, 0]) * 1000.0
    room_h = float(data.u[0, 1]) * 1000.0
    noise  = torch.empty(n_var, 2).uniform_(-1.0, 1.0)
    noise *= torch.tensor([noise_px / room_w, noise_px / room_h])
    data.x[is_var, :2] = (data.x[is_var, :2] + noise).clamp(0.0, 1.0)
    cx  = data.x[:, 0]
    cy  = data.x[:, 1]
    src, dst = data.edge_index
    ddx = cx[dst] - cx[src]
    ddy = cy[dst] - cy[src]
    d   = (ddx**2 + ddy**2).sqrt().clamp(min=1e-8)
    data.edge_attr[:, 0] = ddx
    data.edge_attr[:, 1] = ddy
    data.edge_attr[:, 2] = d / (2.0 ** 0.5)
    data.edge_attr[:, 3] = ddy / d
    data.edge_attr[:, 4] = ddx / d
    return data


def make_loader(dataset, batch_size, shuffle):
    import torch
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-10))


def _unscale(z_arr, log_mean, log_std):
    """Log z-score → waiterDist in px."""
    return np.exp(z_arr * log_std + log_mean)


def evaluate(model, loader, device, log_mean, log_std, criterion):
    import torch
    model.eval()
    total_loss = 0.0
    ys_log, yps_log = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).squeeze(-1)
            y    = batch.y
            loss = criterion(pred, y)
            total_loss += loss.item() * batch.num_graphs
            ys_log.append(y.cpu().numpy())
            yps_log.append(pred.cpu().numpy())

    ys_log  = np.concatenate(ys_log)
    yps_log = np.concatenate(yps_log)

    # MAE en R² in px-ruimte
    ys_px  = _unscale(ys_log,  log_mean, log_std)
    yps_px = _unscale(yps_log, log_mean, log_std)
    mae = float(np.abs(ys_px - yps_px).mean())
    r2  = r2_score(ys_px, yps_px)

    # R² ook in log-ruimte (beter voor evaluatie tijdens training)
    r2_log = r2_score(ys_log, yps_log)
    return total_loss / len(loader.dataset), mae, r2_log   # return log-R² voor monitoring


# ── Hoofdtraining ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Lazy imports na arg-parse (snellere --help)
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    try:
        from gnn_layout import build_graph, RestaurantGNN, Y_SCALE, LOG_MEAN, LOG_STD, NODE_DIM, EDGE_DIM, GLOBAL_DIM
    except ImportError as e:
        print(f"Kan gnn_layout niet importeren: {e}")
        print("Zorg dat torch en torch_geometric geïnstalleerd zijn.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data laden
    data_path = Path(args.data) if args.data else _find_data()
    print(f"\nDataset: {data_path.name}")
    records = load_dataset(data_path)
    print(f"  Geladen: {len(records)} layouts")

    if len(records) < 20:
        print("Te weinig data (min 20 layouts).")
        sys.exit(1)

    # Splits
    train_idx, val_idx, test_idx = make_splits(records)
    print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    y_vals = [r["waiterDist"] for r in records]
    print(f"  waiterDist: {min(y_vals):,.0f} → {max(y_vals):,.0f}  (med {np.median(y_vals):,.0f})")

    print(f"  Alle {len(records)} graphs pre-cachen…", end=" ", flush=True)
    train_ds = LayoutDataset(records, train_idx, build_graph, augment=True,  cache=True)
    val_ds   = LayoutDataset(records, val_idx,   build_graph, augment=False, cache=True)
    test_ds  = LayoutDataset(records, test_idx,  build_graph, augment=False, cache=True)
    print("klaar")

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader   = make_loader(val_ds,   args.batch_size, shuffle=False)
    test_loader  = make_loader(test_ds,  args.batch_size, shuffle=False)

    # Model
    model = RestaurantGNN(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        global_dim=GLOBAL_DIM,
        hidden=args.hidden,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: RestaurantGNN — {n_params:,} parameters")

    # Huber-loss in log-space: delta=0.2 (≈ relatieve fout van ~20%)
    criterion = nn.HuberLoss(delta=0.2)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20,
        min_lr=1e-6, threshold=0.001, threshold_mode='rel',
    )

    # Early stopping
    best_val_mae = 1e9
    best_state   = None
    patience_cnt = 0
    out_path     = Path(args.out) if args.out else ROOT / "gnn_model.pt"

    print(f"\nTraining {args.epochs} epochs (patience={args.patience}) …\n")
    header = f"{'Epoch':>6}  {'Train-loss':>12}  {'Val-MAE':>12}  {'Val-R²':>8}  {'LR':>10}  {'Time':>6}"
    print(header)
    print("-" * len(header))

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        t_ep = time.time()

        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch).squeeze(-1)
            loss = criterion(pred, batch.y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs

        train_loss /= len(train_loader.dataset)

        # ── Validate (elke epoch) ──
        _, val_mae, val_r2 = evaluate(model, val_loader, device, LOG_MEAN, LOG_STD, criterion)

        scheduler.step(val_mae)
        lr_now = optimizer.param_groups[0]['lr']
        ep_time = time.time() - t_ep

        if epoch % 10 == 0 or epoch <= 5:
            print(f"{epoch:>6}  {train_loss:>12.6f}  {val_mae:>10,.0f}px  {val_r2:>8.4f}  "
                  f"{lr_now:>10.2e}  {ep_time:>5.1f}s")

        # Early stopping: verbeter met >0.5% van beste MAE
        threshold = best_val_mae * 0.005
        if val_mae < best_val_mae - threshold:
            best_val_mae = val_mae
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\n  Early stopping op epoch {epoch} (best val MAE={best_val_mae:,.0f} px)")
                break

    total_time = time.time() - t_start
    print(f"\nTraining klaar in {total_time/60:.1f} min")

    # Laad beste gewichten
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test-evaluatie
    _, test_mae, test_r2 = evaluate(model, test_loader, device, LOG_MEAN, LOG_STD, criterion)
    print(f"\nTest-set: MAE={test_mae:,.0f} px  R²={test_r2:.4f}")

    # Vergelijk met RF baseline
    rf_r2  = 0.623
    rf_mae = 85_608
    print(f"RF baseline: MAE={rf_mae:,} px  R²={rf_r2:.3f}")
    delta_mae = rf_mae - test_mae
    delta_r2  = test_r2 - rf_r2
    print(f"GNN verbetering: MAE {delta_mae:+,.0f} px  R² {delta_r2:+.4f}")

    # Sla model op
    torch.save({
        "model_state":  best_state or model.state_dict(),
        "val_mae":      best_val_mae,
        "test_mae":     test_mae,
        "test_r2":      test_r2,
        "node_dim":     NODE_DIM,
        "edge_dim":     EDGE_DIM,
        "global_dim":   GLOBAL_DIM,
        "hidden":       args.hidden,
        "heads":        args.heads,
        "layers":       args.layers,
        "y_scale":      Y_SCALE,
        "log_mean":     LOG_MEAN,
        "log_std":      LOG_STD,
        "epochs_run":   epoch,
    }, out_path)
    print(f"\nModel opgeslagen: {out_path}  (R²={test_r2:.4f})")

    # Voorbeeldpredictie op 5 train-samples voor sanity check
    print("\n── Sanity check (5 train-samples) ──────────────────────────")
    model.eval()
    import torch
    from torch_geometric.data import Batch as PyGBatch
    with torch.no_grad():
        for i in range(min(5, len(train_ds))):
            g = train_ds[i]
            batch = PyGBatch.from_data_list([g]).to(device)
            pred_z = model(batch).item()
            pred   = float(np.exp(pred_z * LOG_STD + LOG_MEAN))
            actual = float(np.exp(g.y.item() * LOG_STD + LOG_MEAN))
            print(f"  voorspeld={pred:,.0f}  actueel={actual:,.0f}  fout={pred-actual:+,.0f}")


if __name__ == "__main__":
    main()
