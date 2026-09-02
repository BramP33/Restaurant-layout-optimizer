"""
A*-afstanden over hetzelfde loopgrid als de simulator.

waiterDist is de totale looplengte van de obers: padlengte over een grid met
obstakels. De bestaande features meten hemelsbrede afstanden en kunnen dus per
definitie niet zien dat een tafel een doorgang blokkeert. Deze module bouwt
hetzelfde grid als `PathNavigator` in simulatie.html en levert echte
padafstanden.

Alles is bewust een 1-op-1 kopie van de simulator-logica; wijkt die af, dan
meten de features iets anders dan het target:

  - cel 18 px, blokkade als het CELMIDDELPUNT in een opgeblazen obstakel valt
  - tafels opgeblazen met 8 px, stoelen als 22x22 blok, bar met 4 px
  - 8-richtingen, diagonaalkosten 1.414, orthogonaal 1
  - een diagonale stap vervalt als een van beide orthogonale buren geblokkeerd
    is (anders snijden agents door meubelhoeken)
  - servicepunten op 26 px rond de ONGEROTEERDE tafelbox, geklemd op
    [16, room-16], daarna naar de dichtstbijzijnde vrije cel binnen 3 ringen

Aanname die afwijkt: voor de drie vaste "custom" tafels kent de dataset geen
stoelposities. Die krijgen hier de standaard rechthoek-stoelindeling. Ze staan
in elke layout identiek, dus een eventuele afwijking is een constante en
verandert de rangorde tussen layouts niet.
"""

import heapq
import math

import numpy as np

CELL     = 18
ROOM_W   = 640
ROOM_H   = 640
BAR_RECT = (ROOM_W - 90, 50, 70, ROOM_H - 100)
BAR_DOCK = (ROOM_W - 110, 80)          # simulatie.html:1153 — NIET het midden van de bar
ENTRANCE = (48, ROOM_H - 54)           # simulatie.html:1145

COLS = math.ceil(ROOM_W / CELL)
ROWS = math.ceil(ROOM_H / CELL)

_SQRT2 = 1.414                          # exact de constante uit de simulator


# ── Meubelgeometrie ──────────────────────────────────────────────────────────

def table_aabb(t):
    """AABB van de geroteerde tafel — zelfde formule als Table.rect()."""
    cx, cy = t["x"] + t["w"] / 2, t["y"] + t["h"] / 2
    rad = math.radians(t.get("rotation", 0))
    cos, sin = abs(math.cos(rad)), abs(math.sin(rad))
    bw = t["w"] * cos + t["h"] * sin
    bh = t["w"] * sin + t["h"] * cos
    return cx - bw / 2, cy - bh / 2, bw, bh


def table_chairs(t):
    """Stoelankers in wereldcoordinaten — zelfde formule als computeChairs()."""
    cx, cy = t["x"] + t["w"] / 2, t["y"] + t["h"] / 2
    rad = math.radians(t.get("rotation", 0))
    cos0, sin0 = math.cos(rad), math.sin(rad)
    n = int(t.get("seats", 4))
    margin = 14

    local = []
    top, bottom = math.ceil(n / 2), math.floor(n / 2)
    hy = t["h"] / 2 + margin
    for i in range(top):
        frac = i / (top - 1) if top > 1 else 0.5
        local.append(((frac - 0.5) * t["w"] * 0.8, -hy))
    for i in range(bottom):
        frac = i / (bottom - 1) if bottom > 1 else 0.5
        local.append(((frac - 0.5) * t["w"] * 0.8, hy))

    return [(cx + dx * cos0 - dy * sin0, cy + dx * sin0 + dy * cos0)
            for dx, dy in local]


def build_blocked(tables):
    """(ROWS, COLS) bool-array — True waar het celmiddelpunt in meubilair valt."""
    xs = (np.arange(COLS) * CELL + CELL / 2)[None, :]
    ys = (np.arange(ROWS) * CELL + CELL / 2)[:, None]
    blocked = np.zeros((ROWS, COLS), dtype=bool)

    def add(x, y, w, h):
        nonlocal blocked
        blocked |= (xs >= x) & (xs <= x + w) & (ys >= y) & (ys <= y + h)

    for t in tables:
        ax, ay, aw, ah = table_aabb(t)
        add(ax - 8, ay - 8, aw + 16, ah + 16)
        for cx, cy in table_chairs(t):
            add(cx - 11, cy - 11, 22, 22)

    bx, by, bw, bh = BAR_RECT
    add(bx - 4, by - 4, bw + 8, bh + 8)
    return blocked


# ── Afstandsveld ─────────────────────────────────────────────────────────────

def point_to_cell(p):
    return (min(max(int(p[1] // CELL), 0), ROWS - 1),
            min(max(int(p[0] // CELL), 0), COLS - 1))


def nearest_open(blocked, p, radius=3):
    """Dichtstbijzijnde vrije cel in ringen rond p — zoals nearestOpenPoint()."""
    r0, c0 = point_to_cell(p)
    if not blocked[r0, c0]:
        return r0, c0
    for r in range(1, radius + 1):
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if abs(dr) != r and abs(dc) != r:
                    continue
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < ROWS and 0 <= cc < COLS and not blocked[rr, cc]:
                    return rr, cc
    return None


def distance_field(blocked, start_point):
    """
    Dijkstra vanaf start_point over het vrije grid. Geeft (ROWS, COLS) in
    PIXELS, met inf voor onbereikbare cellen.

    Een enkele sweep geeft de afstand naar elke tafel tegelijk; dat is waarom
    dit betaalbaar is voor duizenden layouts.
    """
    dist = np.full((ROWS, COLS), np.inf)
    # Exacte cel, niet nearest_open: anders springt het startpunt over een
    # muur heen en lijkt een ingesloten dock bereikbaar. De cel zelf mag
    # geblokkeerd zijn (A* staat de startcel toe), maar de agent moet er via
    # een vrije buur uit kunnen.
    start = point_to_cell(start_point)

    dist[start] = 0.0
    heap = [(0.0, start[0], start[1])]
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r, c]:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < ROWS and 0 <= nc < COLS) or blocked[nr, nc]:
                    continue
                if dr != 0 and dc != 0:
                    # Hoekregel: diagonaal mag niet langs twee geblokkeerde buren
                    if blocked[r, nc] or blocked[nr, c]:
                        continue
                step = _SQRT2 if (dr and dc) else 1.0
                nd = d + step
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))

    return dist * CELL


# ── Servicepunten ────────────────────────────────────────────────────────────

def service_points(t):
    """De acht aanlooppunten rond een tafel — zoals _servicePoint()."""
    cx, cy = t["x"] + t["w"] / 2, t["y"] + t["h"] / 2
    g = 26
    x0, y0, w, h = t["x"], t["y"], t["w"], t["h"]
    cands = [
        (cx, y0 - g), (cx, y0 + h + g),
        (x0 - g, cy), (x0 + w + g, cy),
        (x0 - g, y0 - g), (x0 + w + g, y0 - g),
        (x0 - g, y0 + h + g), (x0 + w + g, y0 + h + g),
    ]
    return [(min(max(px, 16), ROOM_W - 16), min(max(py, 16), ROOM_H - 16))
            for px, py in cands]


def table_access(blocked, dist, t):
    """
    (padafstand, euclidische afstand) naar het best bereikbare servicepunt.

    De simulator kiest het servicepunt dat het dichtst bij de ober ligt; die
    positie varieert per moment. Als stabiele feature nemen we het punt met de
    kortste padafstand vanaf het startpunt van de sweep.
    """
    best_path, best_pt = np.inf, None
    for p in service_points(t):
        cell = nearest_open(blocked, p, radius=3)
        if cell is None:
            continue
        d = dist[cell]
        if d < best_path:
            best_path, best_pt = d, p
    return best_path, best_pt
