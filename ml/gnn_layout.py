"""
GNN-gebaseerde layout optimizer voor Restaurant Simulator.

Bevat:
  - build_graph()                  : layout → PyG Data object
  - build_graph_differentiable()   : differentieerbare variant voor gradiëntoptimalisatie
  - RestaurantGNN                  : GATv2-gebaseerd regressiemodel
  - GNNSurrogate                   : sklearn-compatibele wrapper voor optimize_layout.py
  - project_positions()            : collision-projectie na gradiëntstap
  - gradient_optimize()            : gradiëntgebaseerde layoutoptimalisatie
"""

from __future__ import annotations

import math
import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

# ── Zaal-constanten (zelfde als optimize_layout.py) ──────────────────────────
ROOM_W   = 640
ROOM_H   = 640
WALL_M   = 20
MIN_CORR = 50

# Bar-node: synthetisch punt midden in bar-rect
# BAR_RECT = (ROOM_W - 90, 50, 70, ROOM_H - 100) → center = (550+35, 50+270) = (585, 320)
BAR_CX_DEFAULT   = ROOM_W - 90 + 35      # 585
BAR_CY_DEFAULT   = ROOM_H / 2            # 320
BAR_W_DEFAULT    = 70
BAR_H_DEFAULT    = ROOM_H - 100          # 540

# Ingang-node: center van entrance-rect
# ENTRANCE_RECT = (0, ROOM_H-90, 110, 90) → center = (55, 595)
ENT_CX_DEFAULT   = 55
ENT_CY_DEFAULT   = ROOM_H - 45           # 595
ENT_W_DEFAULT    = 110
ENT_H_DEFAULT    = 90

# Vaste obstakels voor collision-projectie
FIXED_TABLES_DEFAULT = [
    {"size": "custom", "x": 130.59, "y":  57.32, "rotation": 0, "w": 46.85,  "h":  97.69},
    {"size": "custom", "x": 347.91, "y":  60.81, "rotation": 0, "w": 44.86,  "h":  99.69},
    {"size": "custom", "x":  59.81, "y": 204.36, "rotation": 0, "w": 49.84,  "h": 145.55},
]

TABLE_TYPES = {
    "small":  {"w":  60, "h": 60, "seats": 2},
    "medium": {"w":  80, "h": 60, "seats": 4},
    "large":  {"w": 110, "h": 70, "seats": 6},
    "comb1":  {"w":  60, "h": 60, "seats": 2},
    "comb2":  {"w": 120, "h": 60, "seats": 4},
    "comb3":  {"w": 180, "h": 60, "seats": 6},
    "comb4":  {"w": 240, "h": 60, "seats": 8},
}

# Node dimensies
NODE_DIM   = 11
EDGE_DIM   = 6
GLOBAL_DIM = 8

# Target normalisatie: log-space voor skewed distributie
# log(waiterDist) heeft mean≈13.156, std≈0.351  →  z = (log(y) - LOG_MEAN) / LOG_STD  ≈ N(0,1)
Y_SCALE  = 100_000.0    # legacy constante (gebruikt in GNNSurrogate.Y_SCALE check)
LOG_MEAN = 13.156
LOG_STD  = 0.351


# ── Hulpfuncties ──────────────────────────────────────────────────────────────

def _aabb_dims(table: dict) -> tuple[float, float]:
    """Geeft effectieve (w, h) van een tafel rekening houdend met rotatie."""
    w, h = float(table["w"]), float(table["h"])
    if int(table.get("rotation", 0)) % 180 == 90:
        return h, w
    return w, h


def _edge_gap(cx_i, cy_i, w_i, h_i, cx_j, cy_j, w_j, h_j) -> float:
    """
    Berekent de minimale edge-to-edge afstand tussen twee rechthoeken.
    Positief = vrije ruimte; 0 = aanrakend of overlappend.
    """
    # Linker/rechter/boven/onder randen
    left_i,  right_i  = cx_i - w_i / 2, cx_i + w_i / 2
    top_i,   bot_i    = cy_i - h_i / 2, cy_i + h_i / 2
    left_j,  right_j  = cx_j - w_j / 2, cx_j + w_j / 2
    top_j,   bot_j    = cy_j - h_j / 2, cy_j + h_j / 2

    gap_x = max(left_j - right_i, left_i - right_j, 0.0)
    gap_y = max(top_j  - bot_i,   top_i  - bot_j,   0.0)

    # Als de rechthoeken overlappen in één as → de andere as geeft de corridor
    if gap_x == 0.0 and gap_y == 0.0:
        return 0.0   # overlap
    if gap_x == 0.0:
        return gap_y
    if gap_y == 0.0:
        return gap_x
    return min(gap_x, gap_y)   # diagonaal: kleinste

def _edge_gap_torch(
    cx_i, cy_i, w_i, h_i,
    cx_j, cy_j, w_j, h_j,
) -> torch.Tensor:
    """Torch-versie van _edge_gap voor differentieerbare graph-constructie."""
    half_wi, half_hi = w_i / 2, h_i / 2
    half_wj, half_hj = w_j / 2, h_j / 2

    gap_x = torch.clamp(
        torch.max(cx_j - half_wj - (cx_i + half_wi),
                  cx_i - half_wi - (cx_j + half_wj)),
        min=0.0,
    )
    gap_y = torch.clamp(
        torch.max(cy_j - half_hj - (cy_i + half_hi),
                  cy_i - half_hi - (cy_j + half_hj)),
        min=0.0,
    )

    zero = torch.zeros_like(gap_x)
    gap_x_zero = gap_x < 1e-6
    gap_y_zero = gap_y < 1e-6
    both_zero  = gap_x_zero & gap_y_zero

    result = torch.where(both_zero, zero,
             torch.where(gap_x_zero, gap_y,
             torch.where(gap_y_zero, gap_x,
             torch.minimum(gap_x, gap_y))))
    return result


# ── Graph-constructie ────────────────────────────────────────────────────────

def build_graph(tables: list[dict], config: dict, target: Optional[float] = None) -> "Data":
    """
    Zet een restaurant-layout om naar een PyG Data object.

    Args:
        tables : 11 tafel-dicts (3 custom + 8 variabel), willekeurige volgorde
        config : dict met roomW, roomH, guests, waiters
        target : waiterDist als float (None = geen label, voor inferentie)

    Returns:
        Data(x=[13,11], edge_index=[2,156], edge_attr=[156,6], u=[1,8], y=[1])
    """
    assert HAS_PYG, "torch_geometric niet geïnstalleerd"

    rW  = float(config.get("roomW",  ROOM_W))
    rH  = float(config.get("roomH",  ROOM_H))
    g   = float(config.get("guests",  49))
    w_n = float(config.get("waiters",  3))

    # Bar en ingang-posities schalen met zaalgrootte
    bar_cx = rW - 90 + 35
    bar_cy = rH / 2
    bar_w  = 70.0
    bar_h  = rH - 100

    ent_cx = 55.0
    ent_cy = rH - 45
    ent_w  = 110.0
    ent_h  = 90.0

    # ── Sorteer tafels: custom eerst, dan large, dan overige variabel
    custom_t   = [t for t in tables if t.get("size") == "custom"]
    large_t    = [t for t in tables if t.get("size") == "large"]
    other_var  = [t for t in tables if t.get("size") not in ("custom", "large")]

    ordered = custom_t + large_t + other_var   # maximaal 11 tafels

    # ── Bouw node-feature matrix [13, NODE_DIM]
    node_feats = []

    for i, t in enumerate(ordered):
        ew, eh = _aabb_dims(t)
        cx = (float(t["x"]) + ew / 2) / rW
        cy = (float(t["y"]) + eh / 2) / rH
        nw = ew / rW
        nh = eh / rH
        rot_rad = math.radians(float(t.get("rotation", 0)))
        sin_r = math.sin(rot_rad)
        cos_r = math.cos(rot_rad)
        seats = float(t.get("seats", 4)) / 10.0
        is_fix = 1.0 if t.get("size") == "custom" else 0.0
        is_var = 1.0 if t.get("size") != "custom" else 0.0
        node_feats.append([cx, cy, nw, nh, sin_r, cos_r, seats, is_fix, is_var, 0.0, 0.0])

    # Bar-node (index 11)
    node_feats.append([
        bar_cx / rW, bar_cy / rH, bar_w / rW, bar_h / rH,
        0.0, 1.0,   # rotation 0° → sin=0, cos=1
        0.0,        # seats
        1.0, 0.0,   # is_fixed, is_variable
        1.0, 0.0,   # is_bar, is_entrance
    ])

    # Ingang-node (index 12)
    node_feats.append([
        ent_cx / rW, ent_cy / rH, ent_w / rW, ent_h / rH,
        0.0, 1.0,
        0.0,
        1.0, 0.0,
        0.0, 1.0,   # is_bar=0, is_entrance=1
    ])

    x = torch.tensor(node_feats, dtype=torch.float32)   # [13, 11]

    # ── Volledig gericht graaf: alle 13×12 = 156 paren
    n = len(node_feats)   # 13
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)   # [2, 156]

    # ── Bouw centre-coördinaten en afmetingen per node voor edge-features
    # Gebruik de werkelijke pixel-waarden (niet genormaliseerd) voor gap-berekening
    centers_px = []
    dims_px    = []
    for t in ordered:
        ew, eh = _aabb_dims(t)
        centers_px.append((float(t["x"]) + ew / 2, float(t["y"]) + eh / 2))
        dims_px.append((ew, eh))
    centers_px.append((bar_cx, bar_cy))
    dims_px.append((bar_w, bar_h))
    centers_px.append((ent_cx, ent_cy))
    dims_px.append((ent_w, ent_h))

    diag = math.sqrt(rW**2 + rH**2)

    edge_feats = []
    for s, d in zip(src, dst):
        scx, scy = centers_px[s]
        dcx, dcy = centers_px[d]
        sw,  sh  = dims_px[s]
        dw,  dh  = dims_px[d]

        dx = (dcx - scx) / rW
        dy = (dcy - scy) / rH
        dist = math.sqrt((dcx - scx)**2 + (dcy - scy)**2) / diag
        angle = math.atan2(dcy - scy, dcx - scx)
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)
        gap = _edge_gap(scx, scy, sw, sh, dcx, dcy, dw, dh) / rW

        edge_feats.append([dx, dy, dist, sin_a, cos_a, gap])

    edge_attr = torch.tensor(edge_feats, dtype=torch.float32)   # [156, 6]

    # ── Globale context [1, GLOBAL_DIM]
    u = torch.tensor([[
        rW / 1000.0, rH / 1000.0,
        bar_cx / rW, bar_cy / rH,
        ent_cx / rW, ent_cy / rH,
        g  / 50.0,
        w_n / 6.0,
    ]], dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u)

    if target is not None:
        log_y = (math.log(max(target, 1.0)) - LOG_MEAN) / LOG_STD
        data.y = torch.tensor([log_y], dtype=torch.float32)

    return data


# ── GNN Model ─────────────────────────────────────────────────────────────────

class RestaurantGNN(nn.Module):
    """
    Graph Attention Network (GATv2) voor waiterDist regressie.

    Input : PyG Batch met x, edge_index, edge_attr, u
    Output: [B] tensor met geschaalde waiterDist voorspellingen (/ Y_SCALE)
    """

    def __init__(
        self,
        node_dim:   int = NODE_DIM,
        edge_dim:   int = EDGE_DIM,
        global_dim: int = GLOBAL_DIM,
        hidden:     int = 128,
        heads:      int = 4,
        layers:     int = 3,
        dropout:    float = 0.2,
    ):
        super().__init__()
        self.dropout = dropout

        # Input-projectie
        self.node_in = nn.Linear(node_dim, hidden)

        # GATv2 lagen met edge-features
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(GATv2Conv(
                in_channels=hidden,
                out_channels=hidden // heads,   # 32 per head
                heads=heads,
                edge_dim=edge_dim,
                concat=True,                    # output = heads * (hidden//heads) = hidden
                dropout=dropout,
                add_self_loops=False,
            ))
            self.norms.append(nn.LayerNorm(hidden))

        # MLP hoofd — output in log-space (geen activatie nodig)
        pool_out = 2 * hidden + global_dim   # mean + max + global
        self.mlp = nn.Sequential(
            nn.Linear(pool_out, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, data: "Batch") -> torch.Tensor:
        x          = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr
        batch      = data.batch
        u          = data.u   # [B, global_dim]

        # Projecteer naar hidden dimensie
        x = F.relu(self.node_in(x))

        # Message passing met residuele verbindingen
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x_new + x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Globale pooling
        x_mean = global_mean_pool(x, batch)               # [B, hidden]
        x_max  = global_max_pool(x, batch)                # [B, hidden]
        x_pool = torch.cat([x_mean, x_max, u], dim=-1)   # [B, 2*hidden+global_dim]

        return self.mlp(x_pool).squeeze(-1)   # [B]


# ── GNNSurrogate wrapper ──────────────────────────────────────────────────────

class GNNSurrogate:
    """
    Duck-type vervanging voor sklearn model.predict().
    Accepteert rauwe layout-lijsten i.p.v. feature-matrices.
    """

    Y_SCALE = Y_SCALE  # gebruik module-level constante

    def __init__(self, checkpoint_path: Path, device: Optional[str] = None):
        assert HAS_PYG, "torch_geometric niet geïnstalleerd"

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model = RestaurantGNN(
            node_dim=ckpt.get("node_dim",   NODE_DIM),
            edge_dim=ckpt.get("edge_dim",   EDGE_DIM),
            global_dim=ckpt.get("global_dim", GLOBAL_DIM),
            hidden=ckpt.get("hidden", 128),
            heads=ckpt.get("heads",   4),
            layers=ckpt.get("layers",  3),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self._val_r2  = ckpt.get("val_r2",  None)
        self._val_mae = ckpt.get("val_mae", None)

    @property
    def val_r2(self):  return self._val_r2
    @property
    def val_mae(self): return self._val_mae

    def predict(self, layouts: list[list[dict]], config: Optional[dict] = None) -> np.ndarray:
        """
        Voorspel waiterDist voor een lijst van layouts.

        layouts : list van tafel-lijsten (elk 11 tafels)
        config  : optioneel config-dict; gebruikt default 640×640 als None
        """
        if config is None:
            config = {"roomW": ROOM_W, "roomH": ROOM_H, "guests": 49, "waiters": 3}

        graphs = [build_graph(tables, config) for tables in layouts]
        batch  = Batch.from_data_list(graphs).to(self.device)
        with torch.no_grad():
            log_preds = self.model(batch).cpu().numpy()  # z-scores in log-space

        # Terugrekenen naar px: exp(z * LOG_STD + LOG_MEAN)
        return np.exp(log_preds * LOG_STD + LOG_MEAN)


# ── Differentieerbare graph-constructie ──────────────────────────────────────

def build_graph_differentiable(
    positions:    torch.Tensor,   # [N_var, 2]  vereist grad
    rotations:    list[int],
    sizes:        list[str],
    fixed_tables: list[dict],
    config:       dict,
) -> "Data":
    """
    Bouw een PyG Data object waarbij de nodefeatures van variabele tafels
    differentieerbaar zijn t.o.v. `positions`. Wordt gebruikt in gradient_optimize().
    """
    assert HAS_PYG
    device = positions.device

    rW  = float(config.get("roomW", ROOM_W))
    rH  = float(config.get("roomH", ROOM_H))
    g   = float(config.get("guests",  49))
    w_n = float(config.get("waiters",  3))

    bar_cx = torch.tensor(rW - 90 + 35, device=device)
    bar_cy = torch.tensor(rH / 2,       device=device)
    bar_w  = torch.tensor(70.0,         device=device)
    bar_h  = torch.tensor(rH - 100,     device=device)
    ent_cx = torch.tensor(55.0,         device=device)
    ent_cy = torch.tensor(rH - 45,      device=device)
    ent_w  = torch.tensor(110.0,        device=device)
    ent_h  = torch.tensor(90.0,         device=device)

    def _make_node_fixed(t: dict) -> torch.Tensor:
        ew, eh = _aabb_dims(t)
        cx = (float(t["x"]) + ew / 2) / rW
        cy = (float(t["y"]) + eh / 2) / rH
        rot = math.radians(float(t.get("rotation", 0)))
        return torch.tensor(
            [cx, cy, ew / rW, eh / rH,
             math.sin(rot), math.cos(rot),
             float(t.get("seats", 6)) / 10.0,
             1.0, 0.0, 0.0, 0.0],
            dtype=torch.float32, device=device,
        )

    node_list = []
    cx_list, cy_list, w_list, h_list = [], [], [], []

    # Vaste custom tafels
    for t in fixed_tables:
        ew, eh = _aabb_dims(t)
        node_list.append(_make_node_fixed(t))
        cx_list.append(torch.tensor((float(t["x"]) + ew/2), device=device))
        cy_list.append(torch.tensor((float(t["y"]) + eh/2), device=device))
        w_list.append(torch.tensor(ew, device=device))
        h_list.append(torch.tensor(eh, device=device))

    # Variabele tafels (differentieerbaar)
    for idx, (size, rot) in enumerate(zip(sizes, rotations)):
        td  = TABLE_TYPES[size]
        ew  = float(td["h"] if rot % 180 == 90 else td["w"])
        eh  = float(td["w"] if rot % 180 == 90 else td["h"])
        x_i = positions[idx, 0]
        y_i = positions[idx, 1]
        cx  = (x_i + ew / 2) / rW
        cy  = (y_i + eh / 2) / rH
        rot_rad = math.radians(float(rot))
        feat = torch.stack([
            cx, cy,
            torch.tensor(ew / rW, device=device),
            torch.tensor(eh / rH, device=device),
            torch.tensor(math.sin(rot_rad), device=device),
            torch.tensor(math.cos(rot_rad), device=device),
            torch.tensor(float(td["seats"]) / 10.0, device=device),
            torch.tensor(0.0, device=device),   # is_fixed
            torch.tensor(1.0, device=device),   # is_variable
            torch.tensor(0.0, device=device),   # is_bar
            torch.tensor(0.0, device=device),   # is_entrance
        ])
        node_list.append(feat)
        cx_list.append(x_i + ew / 2)
        cy_list.append(y_i + eh / 2)
        w_list.append(torch.tensor(ew, device=device))
        h_list.append(torch.tensor(eh, device=device))

    # Bar-node
    bar_feat = torch.stack([
        bar_cx / rW, bar_cy / rH, bar_w / rW, bar_h / rH,
        torch.tensor(0.0, device=device), torch.tensor(1.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device), torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device), torch.tensor(0.0, device=device),
    ])
    node_list.append(bar_feat)
    cx_list.append(bar_cx)
    cy_list.append(bar_cy)
    w_list.append(bar_w)
    h_list.append(bar_h)

    # Ingang-node
    ent_feat = torch.stack([
        ent_cx / rW, ent_cy / rH, ent_w / rW, ent_h / rH,
        torch.tensor(0.0, device=device), torch.tensor(1.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device), torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device), torch.tensor(1.0, device=device),
    ])
    node_list.append(ent_feat)
    cx_list.append(ent_cx)
    cy_list.append(ent_cy)
    w_list.append(ent_w)
    h_list.append(ent_h)

    x = torch.stack(node_list)   # [N, 11]

    n = len(node_list)
    # Volledig gericht graaf — vectorized
    idx = torch.arange(n, device=device)
    src_idx = idx.repeat_interleave(n - 1)
    # dst: voor elke src, alle nodes behalve zichzelf
    dst_idx = torch.cat([torch.cat([idx[:i], idx[i+1:]]) for i in range(n)])
    edge_index = torch.stack([src_idx, dst_idx], dim=0)   # [2, N*(N-1)]

    # Vectorized edge features
    cx_t = torch.stack(cx_list)   # [N]
    cy_t = torch.stack(cy_list)   # [N]
    w_t  = torch.stack(w_list)    # [N]
    h_t  = torch.stack(h_list)    # [N]

    scx = cx_t[src_idx];  scy = cy_t[src_idx]
    dcx = cx_t[dst_idx];  dcy = cy_t[dst_idx]
    sw  = w_t[src_idx];   sh  = h_t[src_idx]
    dw  = w_t[dst_idx];   dh  = h_t[dst_idx]

    diag    = math.sqrt(rW**2 + rH**2)
    ddx     = dcx - scx
    ddy     = dcy - scy
    eucl    = torch.sqrt(ddx**2 + ddy**2 + 1e-8)
    dx_norm = ddx / rW
    dy_norm = ddy / rH
    dist_n  = eucl / diag
    sin_a   = ddy / eucl
    cos_a   = ddx / eucl

    # Edge-to-edge gap (vectorized, conservative: alleen AABB overlap-as gap)
    ox  = torch.clamp(torch.minimum(scx + sw/2, dcx + dw/2) - torch.maximum(scx - sw/2, dcx - dw/2), min=0.0)
    oy  = torch.clamp(torch.minimum(scy + sh/2, dcy + dh/2) - torch.maximum(scy - sh/2, dcy - dh/2), min=0.0)
    gap = torch.where(
        (ox > 0) & (oy > 0),          # overlap → negatieve gap
        -torch.minimum(ox, oy) / rW,
        (torch.sqrt(ddx**2 + ddy**2 + 1e-8) - (sw + dw) / 2) / rW,
    )

    edge_attr = torch.stack([dx_norm, dy_norm, dist_n, sin_a, cos_a, gap], dim=-1)   # [N*(N-1), 6]

    u = torch.tensor([[
        rW / 1000.0, rH / 1000.0,
        (rW - 90 + 35) / rW, 0.5,
        55.0 / rW, (rH - 45) / rH,
        g / 50.0, w_n / 6.0,
    ]], dtype=torch.float32, device=device)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u)
    data.batch = torch.zeros(n, dtype=torch.long, device=device)
    return data


# ── Collision projectie ───────────────────────────────────────────────────────

def project_positions(
    positions:    torch.Tensor,   # [N_var, 2]
    rotations:    list[int],
    sizes:        list[str],
    config:       dict,
    fixed_tables: Optional[list[dict]] = None,
    n_iters:      int = 20,
) -> torch.Tensor:
    """
    Projecteert variabele tafelposities op de haalbare ruimte:
    1. Binnen wandmarges
    2. Geen overlap met vaste obstakels (bar, ingang, fixed tafels)
    3. Geen pairwise overlap + minimale corridor-breedte
    """
    if fixed_tables is None:
        fixed_tables = FIXED_TABLES_DEFAULT

    pos  = positions.clone()
    rW   = float(config.get("roomW", ROOM_W))
    rH   = float(config.get("roomH", ROOM_H))

    bar_rect = (rW - 90, 50.0, 70.0, rH - 100)   # (x, y, w, h)
    ent_rect = (0.0, rH - 90, 110.0, 90.0)

    def _aabb_wh(size: str, rot: int) -> tuple[float, float]:
        td = TABLE_TYPES[size]
        ew = float(td["h"] if rot % 180 == 90 else td["w"])
        eh = float(td["w"] if rot % 180 == 90 else td["h"])
        return ew, eh

    def _push_out(px, py, pw, ph, ox, oy, ow, oh, gap: float) -> tuple[float, float]:
        """Verplaats (px,py,pw,ph) weg van obstakel (ox,oy,ow,oh) met minimale verschuiving."""
        # Overlap berekenen
        over_x = (px + pw + gap) - ox
        over_x2 = (ox + ow + gap) - px
        over_y = (py + ph + gap) - oy
        over_y2 = (oy + oh + gap) - py

        if over_x <= 0 or over_x2 <= 0 or over_y <= 0 or over_y2 <= 0:
            return px, py   # geen overlap

        # Minimale translatie vector
        tx_r = -over_x   # push links
        tx_l =  over_x2  # push rechts
        ty_b = -over_y   # push omhoog
        ty_t =  over_y2  # push omlaag

        tx = tx_r if abs(tx_r) < abs(tx_l) else tx_l
        ty = ty_b if abs(ty_b) < abs(ty_t) else ty_t

        if abs(tx) <= abs(ty):
            return px + tx, py
        else:
            return px, py + ty

    for _ in range(n_iters):
        n_var = pos.shape[0]

        for i in range(n_var):
            ew, eh = _aabb_wh(sizes[i], rotations[i])
            px = float(pos[i, 0].item())
            py = float(pos[i, 1].item())

            # 1. Wandmarges
            px = max(WALL_M, min(px, rW - WALL_M - ew))
            py = max(WALL_M, min(py, rH - WALL_M - eh))

            # 2. Bar
            bx, by, bw, bh = bar_rect
            px, py = _push_out(px, py, ew, eh, bx, by, bw, bh, gap=25.0)

            # 3. Ingang
            ex, ey, ew2, eh2 = ent_rect
            px, py = _push_out(px, py, ew, eh, ex, ey, ew2, eh2, gap=50.0)

            # 4. Vaste tafels
            for ft in fixed_tables:
                fw, fh = _aabb_dims(ft)
                px, py = _push_out(px, py, ew, eh, float(ft["x"]), float(ft["y"]), fw, fh, gap=MIN_CORR)

            pos[i, 0] = px
            pos[i, 1] = py

        # 5. Pairwise overlap tussen variabele tafels
        for i in range(n_var):
            for j in range(i + 1, n_var):
                ew_i, eh_i = _aabb_wh(sizes[i], rotations[i])
                ew_j, eh_j = _aabb_wh(sizes[j], rotations[j])
                px_i, py_i = float(pos[i, 0].item()), float(pos[i, 1].item())
                px_j, py_j = float(pos[j, 0].item()), float(pos[j, 1].item())

                # Bereken overlap
                over_x  = (px_i + ew_i + MIN_CORR) - px_j
                over_x2 = (px_j + ew_j + MIN_CORR) - px_i
                over_y  = (py_i + eh_i + MIN_CORR) - py_j
                over_y2 = (py_j + eh_j + MIN_CORR) - py_i

                if over_x <= 0 or over_x2 <= 0 or over_y <= 0 or over_y2 <= 0:
                    continue   # geen overlap

                tx_r, tx_l = -over_x / 2, over_x2 / 2
                ty_b, ty_t = -over_y / 2, over_y2 / 2
                tx = tx_r if abs(tx_r) < abs(tx_l) else tx_l
                ty = ty_b if abs(ty_b) < abs(ty_t) else ty_t

                if abs(tx) <= abs(ty):
                    pos[i, 0] -= tx / 2
                    pos[j, 0] += tx / 2
                else:
                    pos[i, 1] -= ty / 2
                    pos[j, 1] += ty / 2

    # Clamp naar wandmarges
    for i in range(pos.shape[0]):
        ew, eh = _aabb_wh(sizes[i], rotations[i])
        pos[i, 0] = pos[i, 0].clamp(WALL_M, rW - WALL_M - ew)
        pos[i, 1] = pos[i, 1].clamp(WALL_M, rH - WALL_M - eh)

    return pos


# ── Gradiëntgebaseerde optimalisatie ─────────────────────────────────────────

def gradient_optimize(
    gnn_model:      RestaurantGNN,
    initial_layout: list[dict],
    config:         dict,
    fixed_tables:   Optional[list[dict]] = None,
    n_steps:        int   = 500,
    lr:             float = 1.0,
    proj_iters:     int   = 20,
    log_interval:   int   = 100,
) -> list[dict]:
    """
    Optimaliseer tafelposities via gradiëntafdaling door het GNN.

    Startpunt: initial_layout (lijst van tafeldicts)
    Terug: geoptimaliseerd layout (lijst van tafeldicts)
    """
    assert HAS_PYG
    if fixed_tables is None:
        fixed_tables = FIXED_TABLES_DEFAULT

    device = next(gnn_model.parameters()).device

    var_tables = [t for t in initial_layout if t.get("size") != "custom"]
    rotations  = [int(t.get("rotation", 0)) for t in var_tables]
    sizes      = [t["size"] for t in var_tables]

    # Beginposities als differentieerbare parameter
    init_pos = torch.tensor(
        [[float(t["x"]), float(t["y"])] for t in var_tables],
        dtype=torch.float32, device=device,
    )
    positions = nn.Parameter(init_pos.clone())
    optimizer = torch.optim.Adam([positions], lr=lr)

    gnn_model.eval()
    best_loss = float("inf")
    best_pos  = init_pos.clone()

    for step in range(n_steps):
        optimizer.zero_grad()

        graph = build_graph_differentiable(
            positions, rotations, sizes, fixed_tables, config,
        )
        # pred is log-z score; minimize exp(pred * LOG_STD + LOG_MEAN) ≈ waiterDist
        # Minimizing pred is equivalent (monotone transformation)
        pred = gnn_model(graph)
        loss = pred   # minimaliseer log-score (=monotoon met waiterDist)

        loss.backward()
        nn.utils.clip_grad_norm_([positions], max_norm=50.0)
        optimizer.step()

        # Projecteer terug naar haalbare ruimte (geen grad)
        with torch.no_grad():
            positions.data = project_positions(
                positions.data, rotations, sizes, config, fixed_tables, proj_iters,
            )
            # Zet loss om naar px voor logging
            pred_px = float(math.exp(float(pred.item()) * LOG_STD + LOG_MEAN))
            if pred_px < best_loss:
                best_loss = pred_px
                best_pos  = positions.data.clone()

        if log_interval > 0 and (step + 1) % log_interval == 0:
            print(f"  stap {step+1:>4}/{n_steps}  dist≈{pred_px:,.0f} px")

    # Bouw output-layout
    optimized = []
    for t in initial_layout:
        if t.get("size") == "custom":
            optimized.append(dict(t))
    for i, (size, rot) in enumerate(zip(sizes, rotations)):
        td = TABLE_TYPES[size]
        optimized.append({
            "size":     size,
            "x":        float(best_pos[i, 0].item()),
            "y":        float(best_pos[i, 1].item()),
            "rotation": rot,
            "w":        td["w"],
            "h":        td["h"],
            "seats":    td["seats"],
        })
    return optimized
