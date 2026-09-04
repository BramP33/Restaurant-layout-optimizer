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
# Buffetlijn tegen de linkerwand -- moet exact overeenkomen met makeBuffet()
# in simulatie.html, anders meet de spiegel een andere vloer dan de simulator.
# Alleen aanwezig bij party-types met hasBuffet; de hele pipeline draait op
# "buffet", dus hier staat hij vast aan.
BUFFET_RECT = (20, 170, 60, max(160, ROOM_H - 340))
BAR_DOCK = (ROOM_W - 110, 80)          # simulatie.html:1153 — NIET het midden van de bar
ENTRANCE = (48, ROOM_H - 54)           # simulatie.html:1145

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


def build_blocked(tables):
    """(ROWS, COLS) bool-array — True waar het celmiddelpunt in meubilair valt."""
    xs = (np.arange(COLS) * CELL + CELL / 2)[None, :]
    ys = (np.arange(ROWS) * CELL + CELL / 2)[:, None]
    blocked = np.zeros((ROWS, COLS), dtype=bool)

    def add(x, y, w, h):
        nonlocal blocked
        blocked |= (xs >= x) & (xs <= x + w) & (ys >= y) & (ys <= y + h)

    for t in map(normalise_table, tables):
        ax, ay, aw, ah = table_aabb(t)
        add(ax - 8, ay - 8, aw + 16, ah + 16)
        for cx, cy in table_chairs(t):
            add(cx - 11, cy - 11, 22, 22)

    bx, by, bw, bh = BAR_RECT
    add(bx - 4, by - 4, bw + 8, bh + 8)

    fx, fy, fw, fh = BUFFET_RECT
    add(fx - 4, fy - 4, fw + 8, fh + 8)
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
    label = np.full((ROWS, COLS), -1, dtype=np.int32)
    sizes = []
    for r0 in range(ROWS):
        for c0 in range(COLS):
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
                        if not (0 <= nr < ROWS and 0 <= nc < COLS) or blocked[nr, nc]:
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
    r0, c0 = point_to_cell(p)
    if not blocked[r0, c0]:
        return {int(label[r0, c0])}
    out = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r0 + dr, c0 + dc
            if not (0 <= nr < ROWS and 0 <= nc < COLS) or blocked[nr, nc]:
                continue
            if dr and dc:
                # De startcel telt hier als vrij, net als in walkable().
                a = (nr, c0) == (r0, c0) or not blocked[nr, c0]
                b = (r0, nc) == (r0, c0) or not blocked[r0, nc]
                if not (a and b):
                    continue
            out.add(int(label[nr, nc]))
    return out


def waiter_floor(blocked, n_waiters=3):
    """
    De gedeelde loopvloer van de obers: de grootste component waar ze alle
    drie in kunnen komen. None als die niet bestaat -- dan staat er minstens
    een ober opgesloten en is de layout ongeldig.
    """
    label, sizes = components(blocked)
    common = None
    for i in range(n_waiters):
        spawn = (BAR_DOCK[0], BAR_DOCK[1] + i * 22)
        e = entry_components(blocked, label, spawn)
        common = e if common is None else (common & e)
        if not common:
            return label, sizes, None
    floor = max(common, key=lambda cid: sizes[cid])
    return label, sizes, floor


def layout_valid(blocked, tables, n_waiters=3):
    """
    Spiegel van SimulationEngine._checkLayoutReachability(). Geeft
    (valid, unreachable_tables, trapped_waiters).
    """
    label, sizes, floor = waiter_floor(blocked, n_waiters)
    if floor is None:
        return False, len(tables), n_waiters
    trapped = 0
    for i in range(n_waiters):
        spawn = (BAR_DOCK[0], BAR_DOCK[1] + i * 22)
        if floor not in entry_components(blocked, label, spawn):
            trapped += 1

    def in_floor(p):
        cell = nearest_open(blocked, p, radius=5)
        return cell is not None and int(label[cell]) == floor

    if sizes[floor] < 10 or not in_floor(BAR_DOCK):
        return False, len(tables), trapped

    unreachable = 0
    for t in tables:
        if not any(in_floor(p) for p in service_points(t)):
            unreachable += 1
    return unreachable == 0 and trapped == 0, unreachable, trapped


def distance_field(blocked, start_point):
    """
    Dijkstra vanaf start_point over het vrije grid. Geeft (ROWS, COLS) in
    PIXELS, met inf voor onbereikbare cellen.

    Een enkele sweep geeft de afstand naar elke tafel tegelijk; dat is waarom
    dit betaalbaar is voor duizenden layouts.
    """
    dist = np.full((ROWS, COLS), np.inf)
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


def overlaps_buffet(tables):
    """
    Staat er een tafel in de buffetlijn?

    layout_valid() vangt dit NIET: die toetst bereikbaarheid, en een tafel die
    half in het buffet staat blijft prima bereikbaar. Fysiek is het onzin, dus
    de optimizer moet er apart op filteren -- precies het soort gat waar de
    zoektocht eerder in kroop.
    """
    fx, fy, fw, fh = BUFFET_RECT
    for t in tables:
        if t.get("size") == "custom":
            continue
        ax, ay, aw, ah = table_aabb(normalise_table(t))
        if ax < fx + fw and ax + aw > fx and ay < fy + fh and ay + ah > fy:
            return True
    return False


def path_features(variable, fixed=()):
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

    blocked = build_blocked(tables)
    dist    = distance_field(blocked, BAR_DOCK)

    free      = ~blocked
    reachable = np.isfinite(dist) & free

    paths, euclids, seats = [], [], []
    for t in variable:
        p, pt = table_access(blocked, dist, t)
        tn = normalise_table(t)
        cx, cy = tn["x"] + tn["w"] / 2, tn["y"] + tn["h"] / 2
        e = math.hypot(cx - BAR_DOCK[0], cy - BAR_DOCK[1])
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
    penalty = 10.0 * ROOM_W          # 6.400 px, ruim boven de langste route
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


def tour_features(variable):
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
    blocked = build_blocked(variable)
    n       = len(variable)
    BIG     = 10.0 * ROOM_W

    bar = distance_field(blocked, BAR_DOCK)

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
        _d, bp = table_access(blocked, bar, t)
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
