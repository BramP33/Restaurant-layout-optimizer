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

from rooms import CLASSIC, buffet_slot_points as _room_slot_points  # noqa: E402

CELL     = 18
# Terugvalzaal. Elke functie hieronder neemt een `room`; laat je hem weg dan
# krijg je de zaal waarop alle eerdere data verzameld is, zodat oude aanroepen
# ongewijzigd blijven werken.
ROOM_W   = CLASSIC["w"]
ROOM_H   = CLASSIC["h"]
BAR_RECT = (ROOM_W - 90, 50, 70, ROOM_H - 100)   # alleen nog documentatie
# Buffetlijn tegen de linkerwand -- moet exact overeenkomen met makeBuffet()
# in simulatie.html, anders meet de spiegel een andere vloer dan de simulator.
# Alleen aanwezig bij party-types met hasBuffet; de hele pipeline draait op
# "buffet", dus hier staat hij vast aan.
BUFFET_RECT = (20, 170, 60, max(160, ROOM_H - 340))
BAR_DOCK = (ROOM_W - 110, 80)          # simulatie.html:1153 — NIET het midden van de bar
ENTRANCE = (48, ROOM_H - 54)           # simulatie.html:1145

def grid_shape(room):
    return math.ceil(room["h"] / CELL), math.ceil(room["w"] / CELL)


COLS = math.ceil(ROOM_W / CELL)
ROWS = math.ceil(ROOM_H / CELL)

_SQRT2 = 1.414                          # exact de constante uit de simulator


# ── Meubelgeometrie ──────────────────────────────────────────────────────────

# Zoals DEFAULT_TABLE_TYPE_DEFS in simulatie.html:494. `new Table(id, size, ...)`
# negeert de w/h/seats die in een datasetrecord staan en haalt ze hieruit op.
# Records uit een sessie met aangepaste tafeltypes in localStorage dragen dus
# afmetingen die een verse browser nooit reproduceert; wie het record gelooft,
# rekent features op andere geometrie dan de target gemeten is.
TABLE_TYPE_DEFS = {
    "small":  {"w":  60, "h": 60, "seats": 2},
    "medium": {"w":  80, "h": 60, "seats": 4},
    "large":  {"w": 110, "h": 70, "seats": 6},
    "comb1":  {"w":  60, "h": 60, "seats": 2},
    "comb2":  {"w": 120, "h": 60, "seats": 4},
    "comb3":  {"w": 180, "h": 60, "seats": 6},
    "comb4":  {"w": 240, "h": 60, "seats": 8},
}


def normalise_table(t):
    """Afmetingen uit het tafeltype halen, niet uit het record."""
    d = TABLE_TYPE_DEFS.get(t.get("size"))
    if d is None:                       # "custom": alleen het record weet het
        return t
    if t.get("w") == d["w"] and t.get("h") == d["h"] and t.get("seats") == d["seats"]:
        return t
    out = dict(t)
    out.update(d)
    return out


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


def build_blocked(tables, room=CLASSIC):
    """(rows, cols) bool-array — True waar het celmiddelpunt in meubilair valt."""
    rows, cols = grid_shape(room)
    xs = (np.arange(cols) * CELL + CELL / 2)[None, :]
    ys = (np.arange(rows) * CELL + CELL / 2)[:, None]
    blocked = np.zeros((rows, cols), dtype=bool)

    def add(x, y, w, h):
        nonlocal blocked
        blocked |= (xs >= x) & (xs <= x + w) & (ys >= y) & (ys <= y + h)

    for t in map(normalise_table, tables):
        ax, ay, aw, ah = table_aabb(t)
        add(ax - 8, ay - 8, aw + 16, ah + 16)
        for cx, cy in table_chairs(t):
            add(cx - 11, cy - 11, 22, 22)

    b = room["bar"]
    add(b["x"] - 4, b["y"] - 4, b["w"] + 8, b["h"] + 8)

    f = room.get("buffet")
    if f:
        add(f["x"] - 4, f["y"] - 4, f["w"] + 8, f["h"] + 8)

    # Kolommen, podia en uitgesneden hoeken: alles wat vloer wegneemt zonder
    # meubilair te zijn. Zelfde inflatie als de bar, want een agent moet er
    # net zo goed omheen.
    for blk in room.get("blocks", []):
        add(blk["x"] - 4, blk["y"] - 4, blk["w"] + 8, blk["h"] + 8)
    return blocked


# ── Afstandsveld ─────────────────────────────────────────────────────────────

def point_to_cell(p, blocked):
    rows, cols = blocked.shape
    return (min(max(int(p[1] // CELL), 0), rows - 1),
            min(max(int(p[0] // CELL), 0), cols - 1))


def nearest_open(blocked, p, radius=3):
    """Dichtstbijzijnde vrije cel in ringen rond p — zoals nearestOpenPoint()."""
    rows, cols = blocked.shape
    r0, c0 = point_to_cell(p, blocked)
    if not blocked[r0, c0]:
        return r0, c0
    for r in range(1, radius + 1):
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if abs(dr) != r and abs(dc) != r:
                    continue
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < rows and 0 <= cc < cols and not blocked[rr, cc]:
                    return rr, cc
    return None


def components(blocked):
    """
    Labelt elke vrije cel met het nummer van zijn samenhangende component.
    Spiegel van PathNavigator.components() in simulatie.html.

    Geen enkele geblokkeerde cel doet mee, ook een startcel niet: findPath
    vrijwaart de startcel alleen zolang de route loopt, dus een agent kan er
    afstappen maar er nooit doorheen. Wie een geblokkeerde cel wel als
    doorgang telt, plakt gebieden aan elkaar die nooit verbonden zijn -- en
    daar kroop de optimizer in.
    """
    rows, cols = blocked.shape
    label = np.full((rows, cols), -1, dtype=np.int32)
    sizes = []
    for r0 in range(rows):
        for c0 in range(cols):
            if blocked[r0, c0] or label[r0, c0] >= 0:
                continue
            cid = len(sizes)
            label[r0, c0] = cid
            stack = [(r0, c0)]
            n = 0
            while stack:
                r, c = stack.pop()
                n += 1
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < rows and 0 <= nc < cols) or blocked[nr, nc]:
                            continue
                        if dr and dc and (blocked[r, nc] or blocked[nr, c]):
                            continue          # hoekregel, gelijk aan findPath
                        if label[nr, nc] >= 0:
                            continue
                        label[nr, nc] = cid
                        stack.append((nr, nc))
            sizes.append(n)
    return label, sizes


def entry_components(blocked, label, p):
    """De componenten waarin een agent op punt p terecht kan komen."""
    rows, cols = blocked.shape
    r0, c0 = point_to_cell(p, blocked)
    if not blocked[r0, c0]:
        return {int(label[r0, c0])}
    out = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r0 + dr, c0 + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or blocked[nr, nc]:
                continue
            if dr and dc:
                # De startcel telt hier als vrij, net als in walkable().
                a = (nr, c0) == (r0, c0) or not blocked[nr, c0]
                b = (r0, nc) == (r0, c0) or not blocked[r0, nc]
                if not (a and b):
                    continue
            out.add(int(label[nr, nc]))
    return out


def waiter_spawn(room, i):
    """
    Waar ober `i` begint.

    De obers staan naast elkaar LANGS de bar. Altijd verticaal stapelen ging
    goed zolang de bar tegen een zijwand stond, maar zet de bar onderaan en
    dan staan ober twee en drie middenin het meubel -- geen gemeenschappelijke
    loopvloer, en elke layout wordt ongeldig verklaard. Moet gelijk zijn aan
    waiterSpawn() in simulatie.html.
    """
    d = room["bar"]["dock"]
    b = room["bar"]
    if b["h"] >= b["w"]:                 # verticale bar: naast elkaar in y
        return (d["x"], d["y"] + i * 22)
    return (d["x"] + i * 22, d["y"])     # horizontale bar: naast elkaar in x


def waiter_floor(blocked, n_waiters=3, room=CLASSIC):
    """
    De gedeelde loopvloer van de obers: de grootste component waar ze alle
    drie in kunnen komen. None als die niet bestaat -- dan staat er minstens
    een ober opgesloten en is de layout ongeldig.
    """
    label, sizes = components(blocked)
    common = None
    for i in range(n_waiters):
        spawn = waiter_spawn(room, i)
        e = entry_components(blocked, label, spawn)
        common = e if common is None else (common & e)
        if not common:
            return label, sizes, None
    floor = max(common, key=lambda cid: sizes[cid])
    return label, sizes, floor


def layout_valid(blocked, tables, n_waiters=3, room=CLASSIC):
    """
    Spiegel van SimulationEngine._checkLayoutReachability(). Geeft
    (valid, unreachable_tables, trapped_waiters).
    """
    dock = (room["bar"]["dock"]["x"], room["bar"]["dock"]["y"])
    label, sizes, floor = waiter_floor(blocked, n_waiters, room)
    if floor is None:
        return False, len(tables), n_waiters
    trapped = 0
    for i in range(n_waiters):
        spawn = waiter_spawn(room, i)
        if floor not in entry_components(blocked, label, spawn):
            trapped += 1

    def in_floor(p):
        # Radius 3, net als nearestOpenPoint(p, 3) in simulatie.html voor
        # service- en buffetpunten. Met 5 snapte de spiegel verder dan de
        # simulator en keurde hij indelingen goed die de browser afkeurt --
        # altijd die kant op, dus precies de kant die de optimizerpoort
        # laat doorglippen.
        cell = nearest_open(blocked, p, radius=3)
        return cell is not None and int(label[cell]) == floor

    def dock_in_floor(p):
        cell = nearest_open(blocked, p, radius=5)     # de dock snapt wel op 5
        return cell is not None and int(label[cell]) == floor

    if sizes[floor] < 10 or not dock_in_floor(dock):
        return False, len(tables), trapped

    # Spiegel van de buffettoets in simulatie.html: een indeling die de
    # buffetlijn afsluit is ongeldig. Zonder deze regel keurt de zeef iets
    # anders goed dan de simulator, en dat verschil is precies waar een
    # optimizer op afgaat.
    if room.get("buffet") and not any(in_floor(p) for p in buffet_slot_points(room)):
        return False, len(tables), trapped

    unreachable = 0
    for t in tables:
        if not any(in_floor(p) for p in service_points(t, room)):
            unreachable += 1
    return unreachable == 0 and trapped == 0, unreachable, trapped


def distance_field(blocked, start_point):
    """
    Dijkstra vanaf start_point over het vrije grid. Geeft (ROWS, COLS) in
    PIXELS, met inf voor onbereikbare cellen.

    Een enkele sweep geeft de afstand naar elke tafel tegelijk; dat is waarom
    dit betaalbaar is voor duizenden layouts.
    """
    rows, cols = blocked.shape
    dist = np.full((rows, cols), np.inf)
    # De obers lopen naar nearestOpenPoint(dock, 5), niet naar de dockcel zelf;
    # dat punt is dus het eerlijke vertrekpunt. Een geblokkeerde cel mag nooit
    # het zaad zijn: dan sijpelt de sweep naar buiten via een cel waar niemand
    # doorheen kan.
    start = nearest_open(blocked, start_point, radius=5)
    if start is None:
        return dist

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
                if not (0 <= nr < rows and 0 <= nc < cols) or blocked[nr, nc]:
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

def service_points(t, room=CLASSIC):
    """De acht aanlooppunten rond een tafel — zoals _servicePoint()."""
    t = normalise_table(t)
    cx, cy = t["x"] + t["w"] / 2, t["y"] + t["h"] / 2
    g = 26
    x0, y0, w, h = t["x"], t["y"], t["w"], t["h"]
    cands = [
        (cx, y0 - g), (cx, y0 + h + g),
        (x0 - g, cy), (x0 + w + g, cy),
        (x0 - g, y0 - g), (x0 + w + g, y0 - g),
        (x0 - g, y0 + h + g), (x0 + w + g, y0 + h + g),
    ]
    return [(min(max(px, 16), room["w"] - 16), min(max(py, 16), room["h"] - 16))
            for px, py in cands]


def table_access(blocked, dist, t, room=CLASSIC):
    """
    (padafstand, euclidische afstand) naar het best bereikbare servicepunt.

    De simulator kiest het servicepunt dat het dichtst bij de ober ligt; die
    positie varieert per moment. Als stabiele feature nemen we het punt met de
    kortste padafstand vanaf het startpunt van de sweep.
    """
    best_path, best_pt = np.inf, None
    for p in service_points(t, room):
        cell = nearest_open(blocked, p, radius=3)
        if cell is None:
            continue
        d = dist[cell]
        if d < best_path:
            best_path, best_pt = d, p
    return best_path, best_pt


# ── Features voor de surrogate ───────────────────────────────────────────────

# LET OP -- de drie "custom" tafels in de datasetrecords zijn GEEN obstakels.
#
# Elk record draagt een `tables`-veld met 11 tafels, waarvan 3 custom. Het is
# verleidelijk die als vast meubilair in het grid te zetten, maar de simulatie
# die het target heeft gemeten kende ze niet:
#
#   - validate_headless.js:42 filtert custom weg en stuurt alleen de variabele
#     tafels als `forcedLayout` de pagina in;
#   - config.forcedLayout bevat in alle 30.720 runs precies 8 tafels en 0 custom;
#   - simulatie.html:1375 plaatst custom tafels uitsluitend uit
#     `engine.drawnTables`, en die wordt alleen uit localStorage gevuld
#     (simulatie.html:2791). Een verse headless browser heeft die niet.
#
# Het `tables`-veld is dus inerte metadata die collect_parallel meekopieert.
# Ze toch in het grid zetten meet een andere vloer dan waar het target vandaan
# komt: het sluit doorgangen af die in de simulatie openstaan.
N_PATH_FEATURES = 24


def buffet_slot_points(room=CLASSIC):
    """Aanlooppunten op de zaalkant van de buffetlijn -- zelfde formule als
    buffetSlotPoints() in simulatie.html. Gedeeld met rooms.py zodat er maar
    een definitie bestaat."""
    return _room_slot_points(room.get("buffet"))


MIN_TABLE_GAP = 50      # zelfde corridor als generate_batch (MIN_CORR)


def min_table_gap(tables):
    """
    Kleinste vrije ruimte tussen twee tafels. Negatief betekent overlap.

    generate_batch dwingt een corridor van 50 px af, maar local_refine
    verschoof 500 rondes lang tafels van +-35 px zonder enige toets. Drie
    tafels op elkaar stapelen verlaagt waiterDist met bijna 40% -- de obers
    lopen dan naar een punt in plaats van naar een zaal -- en niets keurde dat
    af. Vierde exploit van dit soort in dit project; deze toets sluit hem.
    """
    n = len(tables)
    if n < 2:
        return float("inf")
    boxes = [table_aabb(normalise_table(t)) for t in tables]
    worst = float("inf")
    for i in range(n):
        ax, ay, aw, ah = boxes[i]
        for j in range(i + 1, n):
            bx, by, bw, bh = boxes[j]
            gx = max(bx - (ax + aw), ax - (bx + bw))
            gy = max(by - (ay + ah), ay - (by + bh))
            # Overlappen ze op beide assen, dan is de "afstand" de diepste
            # doordringing (negatief); anders de grootste vrije as.
            worst = min(worst, max(gx, gy))
    return worst


def tables_ok(tables, min_gap=MIN_TABLE_GAP):
    return min_table_gap(tables) >= min_gap


def overlaps_buffet(tables, room=CLASSIC):
    """
    Staat er een tafel in de buffetlijn?

    layout_valid() vangt dit NIET: die toetst bereikbaarheid, en een tafel die
    half in het buffet staat blijft prima bereikbaar. Fysiek is het onzin, dus
    de optimizer moet er apart op filteren -- precies het soort gat waar de
    zoektocht eerder in kroop.
    """
    f = room.get("buffet")
    if not f:
        return False
    fx, fy, fw, fh = f["x"], f["y"], f["w"], f["h"]
    for t in tables:
        if t.get("size") == "custom":
            continue
        ax, ay, aw, ah = table_aabb(normalise_table(t))
        if ax < fx + fw and ax + aw > fx and ay < fy + fh and ay + ah > fy:
            return True
    return False


def path_features(variable, room=CLASSIC, fixed=()):
    """
    Padgebaseerde features uit een enkele Dijkstra-sweep vanaf de bardock.

    De bestaande features zijn Euclidisch terwijl het target een A*-padlengte
    is. Een tafel die een doorgang dichtzet ziet er hemelsbreed onschuldig uit
    en verdubbelt intussen de looproute -- dat verschil is precies wat hier
    gemeten wordt.

    De sterkste term is niet de afstand zelf maar `sum(seats * padafstand)`:
    obers lopen heen en weer per bestelling, en het aantal bestellingen aan een
    tafel schaalt met het aantal stoelen. Dat product is dus een directe
    natuurkundige schatting van de totale looplengte.

    Geeft altijd exact N_PATH_FEATURES waarden terug, ook als de layout
    onbereikbaar is -- de lengte van de featurevector moet vast liggen.
    """
    # `fixed` is standaard leeg: de simulator plaatst alleen de variabele
    # tafels (zie de toelichting bovenaan deze sectie). Alleen wie een sessie
    # met getekende tafels naspeelt, geeft hier iets mee.
    tables = list(fixed) + list(variable)

    dock    = (room["bar"]["dock"]["x"], room["bar"]["dock"]["y"])
    blocked = build_blocked(tables, room)
    dist    = distance_field(blocked, dock)

    free      = ~blocked
    reachable = np.isfinite(dist) & free

    paths, euclids, seats = [], [], []
    for t in variable:
        p, pt = table_access(blocked, dist, t, room)
        tn = normalise_table(t)
        cx, cy = tn["x"] + tn["w"] / 2, tn["y"] + tn["h"] / 2
        e = math.hypot(cx - dock[0], cy - dock[1])
        paths.append(p)
        euclids.append(e)
        seats.append(float(tn.get("seats", 4)))

    paths   = np.array(paths, dtype=float)
    euclids = np.array(euclids, dtype=float)
    seats   = np.array(seats, dtype=float)

    n_unreachable = int(np.isinf(paths).sum())
    # Onbereikbare tafels krijgen een eindige strafwaarde: inf maakt elke
    # afgeleide statistiek inf en het model kan er niets mee. De zeef in
    # collect_parallel houdt deze layouts sowieso buiten de trainingsdata,
    # maar de optimizer voert ze wel aan het model.
    # De straf moet boven ELKE echte padafstand liggen. `max(eindig) * 2` is
    # dat niet: in een krappe indeling loopt een bereikbare tafel tot ~1.800 px
    # terwijl een ruime layout op ~600 px zit, dus daar zou een afgesloten
    # tafel goedkoper uitvallen dan een ver-maar-bereikbare. Dat is precies de
    # vorm van de exploit die we dichttimmeren, nu in de featureruimte.
    # Bovengrens die per constructie klopt: geen kortste route kan langer zijn
    # dan alle vrije cellen achter elkaar. De oude waarde (10 x zaalbreedte)
    # was geijkt op 640x640; in een grotere zaal kon een onbereikbare tafel
    # daardoor goedkoper uitvallen dan een verre bereikbare -- precies de
    # exploit die dit commentaar hierboven beschrijft.
    penalty = float(free.sum()) * CELL + 1.0
    paths = np.where(np.isinf(paths), penalty, paths)

    detour = paths / np.maximum(euclids, 1.0)
    work   = seats * paths          # de proxy voor totale looplengte

    sorted_paths = np.sort(paths)
    if sorted_paths.size < 8:       # vaste lengte afdwingen
        sorted_paths = np.pad(sorted_paths, (0, 8 - sorted_paths.size),
                              constant_values=sorted_paths[-1] if sorted_paths.size else 0.0)
    else:
        sorted_paths = sorted_paths[:8]

    floor_d = dist[reachable]

    return [
        # Padafstand tot de bar
        float(paths.mean()), float(paths.min()), float(paths.max()),
        float(paths.std()),  float(paths.sum()),
        *[float(v) for v in sorted_paths],                 # 8
        # Omweg ten opzichte van hemelsbreed: hoeveel blokkeert de indeling?
        float(detour.mean()), float(detour.max()),
        float(detour.std()),  float(detour.min()),
        # Werk-proxy: stoelen maal padafstand
        float(work.sum()), float(work.mean()), float(work.max()),
        # Hoe open is de vloer, en hoe ver ligt hij gemiddeld van de bar?
        float(reachable.sum()),
        float(floor_d.mean()) if floor_d.size else 0.0,
        float(np.percentile(floor_d, 90)) if floor_d.size else 0.0,
        float(n_unreachable),
    ]


# ── Tour-features ────────────────────────────────────────────────────────────

N_TOUR_FEATURES = 14


def tour_features(variable, room=CLASSIC):
    """
    Wat de ober werkelijk loopt: de kosten van een rit langs meerdere tafels.

    De ober draagt tot acht drankjes en bouwt een plan van meerdere tafels
    (simulatie.html:2061). Hij kettingt die tafels aan elkaar op EUCLIDISCHE
    afstand tot de ankertafel, maar loopt vervolgens A*-paden. Twee tafels die
    hemelsbreed naast elkaar liggen met een obstakel ertussen belanden dus in
    dezelfde rit en kosten een omweg -- en geen enkele bestaande feature ziet
    dat, want die meten alleen de afstand tot de bar en Euclidische
    paarafstanden.

    Vandaar de mismatch-features (pad gedeeld door hemelsbreed per tafelpaar):
    ze meten precies waar de kettingheuristiek van de simulator zichzelf in de
    voet schiet.

    Duur: acht extra Dijkstra-sweeps, ~34 ms per layout tegen ~4 ms voor
    path_features. Te duur voor een brede zoektocht, de moeite waard om de
    kopgroep te herordenen -- en daar helpen ze ook het meest.
    """
    blocked = build_blocked(variable, room)
    n       = len(variable)
    BIG     = float((~blocked).sum()) * CELL + 1.0   # zie path_features

    bar = distance_field(blocked, (room["bar"]["dock"]["x"], room["bar"]["dock"]["y"]))

    # Het anker per tafel MOET op de obervloer liggen. Het eerste servicepunt
    # met een vrije cel in de buurt pakken is niet genoeg: dat kan een
    # afgesloten nis naast de vloer zijn, en dan meet het afstandsveld vanaf
    # dat punt een gebied waar geen ober ooit komt. Op de trainingsset gaf dat
    # 15% van de layouts een fantoomstraf van BIG px -- allemaal volledig
    # geldige indelingen -- en dat draaide het teken van de features om.
    # table_access kiest via het barveld en is daarmee wel vloerbewust: een
    # onbereikbaar servicepunt heeft afstand inf en wint nooit.
    fields, pts = [], []
    for t in variable:
        _d, bp = table_access(blocked, bar, t, room)
        pts.append(bp)
        fields.append(distance_field(blocked, bp) if bp is not None else None)

    def at(field, p):
        if field is None or p is None:
            return BIG
        c = nearest_open(blocked, p, radius=3)
        v = field[c] if c is not None else np.inf
        return BIG if not np.isfinite(v) else float(v)

    M = np.full((n, n), BIG)      # padafstand tafel -> tafel
    E = np.zeros((n, n))          # hemelsbreed, wat de simulator gebruikt
    for i in range(n):
        M[i, i] = 0.0
        for j in range(n):
            if i == j:
                continue
            M[i, j] = at(fields[i], pts[j])
            E[i, j] = math.hypot(variable[i]["x"] - variable[j]["x"],
                                 variable[i]["y"] - variable[j]["y"])
    bd = np.array([at(bar, p) for p in pts])

    off  = ~np.eye(n, dtype=bool)
    pair = M[off]
    mism = pair / np.maximum(E[off], 1.0)

    # Greedy rit: bar -> steeds de dichtstbijzijnde ongeziene tafel -> bar.
    cur  = int(np.argmin(bd))
    tour = bd[cur]
    unvisited = set(range(n)) - {cur}
    while unvisited:
        nxt   = min(unvisited, key=lambda j: M[cur, j])
        tour += M[cur, nxt]
        cur   = nxt
        unvisited.discard(cur)
    tour += bd[cur]

    # Getest en verworpen: een rit in de volgorde die de simulator werkelijk
    # kiest (Euclidisch sorteren vanaf de bar, simulatie.html:2091) in plaats
    # van de greedy volgorde hieronder. Over 9 metingen leverde die extra
    # feature niets op -- drho +0,037 tegen +0,039 zonder -- dus hij is er
    # weer uit. De greedy rit blijft, als maat voor hoe duur de vloer is.
    nn = np.array([M[i][np.arange(n) != i].min() for i in range(n)])
    # De drie tafels die de simulator zou aanketenen: Euclidisch het dichtst.
    chain = np.array([M[i, np.argsort(E[i] + np.eye(n)[i] * 1e9)[:3]].mean()
                      for i in range(n)])

    return [
        float(pair.mean()), float(pair.min()), float(pair.max()), float(pair.std()),
        float(tour),
        float(tour / max(2.0 * bd.sum(), 1.0)),   # winst van ketenen t.o.v. losse ritten
        float(mism.mean()), float(mism.max()),    # waar de heuristiek misgrijpt
        float(nn.mean()), float(nn.max()),
        float(chain.mean()), float(chain.std()),
        float((mism > 1.5).sum()), float((mism > 2.0).sum()),
    ]
