"""
Zaalgenerator.

De simulator genereert zijn zaal NIET zelf. Die krijgt hem aangereikt via
`config.room`, en pathgrid.py rekent op datzelfde object. Dat is een bewuste
keuze: zouden beide kanten zelf een zaal trekken, dan moeten twee losse
RNG-implementaties exact hetzelfde doen, en juist dat soort stille afwijking
tussen simulator en spiegel is in dit project al drie keer misgegaan.

Een zaal is een plat woordenboek:

    {
      "kind":     "kolommen",              # welk archetype
      "w": 640, "h": 640,                  # buitenmaten in px
      "bar":      {"x","y","w","h","dock":{"x","y"}},
      "buffet":   {"x","y","w","h","slots"} of None,
      "entrance": {"x","y"},
      "blocks":   [{"x","y","w","h"}, ...] # kolommen, podia, uitgesneden hoeken
    }

`blocks` is de generieke bak voor alles wat vloer wegneemt. Een L-vorm is een
rechthoekige zaal met een groot blok in een hoek; een kolommenzaal heeft een
handvol kleine blokken. Zo hoeft het loopgrid maar een ding te kennen.

Schaal: ongeveer 67 px per meter (een tafel voor vier is 80 px breed en in het
echt ~1,20 m). Zie onderzoek/zaalvormen.md.
"""

import math

CELL         = 18      # loopgrid, moet gelijk zijn aan PathNavigator.cell
WALL_MARGIN  = 20      # tafels blijven zo ver van de muur
PX_PER_M     = 67

KINDS = ["rechthoek", "langwerpig", "kolommen", "l-vorm", "nis", "podium"]


# ── Bouwstenen ───────────────────────────────────────────────────────────────

def _bar_on_wall(w, h, wall, rng):
    """Bar tegen een wand, met de dock net ervoor in de zaal."""
    depth = 70
    if wall in ("links", "rechts"):
        length = h - 100
        y      = 50
        x      = 20 if wall == "links" else w - 90
        # De dock ligt aan de zaalkant van de bar, boven in plaats van in het
        # midden -- zo staat hij in de simulator ook (simulatie.html:1153).
        dock   = {"x": x + depth + 20 if wall == "links" else x - 20,
                  "y": y + 30}
    else:
        length = w - 100
        x      = 50
        y      = 20 if wall == "boven" else h - 90
        dock   = {"x": x + 30,
                  "y": y + depth + 20 if wall == "boven" else y - 20}
    if wall in ("links", "rechts"):
        return {"x": x, "y": y, "w": depth, "h": length, "dock": dock}
    return {"x": x, "y": y, "w": length, "h": depth, "dock": dock}


def _buffet_on_wall(w, h, wall, rng):
    """Buffetlijn tegen een wand; enkelzijdig, dus aanschuiven vanaf de zaal."""
    depth = 60
    if wall in ("links", "rechts"):
        length = max(160, int(h * 0.45))
        y      = int((h - length) / 2)
        x      = 20 if wall == "links" else w - 80
    else:
        length = max(160, int(w * 0.45))
        x      = int((w - length) / 2)
        y      = 20 if wall == "boven" else h - 80
    if wall in ("links", "rechts"):
        return {"x": x, "y": y, "w": depth, "h": length, "slots": 3,
            "dwellRange": [45, 90], "wall": wall}
    return {"x": x, "y": y, "w": length, "h": depth, "slots": 3,
            "dwellRange": [45, 90], "wall": wall}


def buffet_slot_points(buffet):
    """Aanlooppunten op de zaalkant van de lijn. Moet gelijk zijn aan
    buffetSlotPoints() in simulatie.html."""
    if not buffet:
        return []
    n, gap = buffet["slots"], 26
    wall = buffet.get("wall", "links")
    pts = []
    for i in range(n):
        f = (i + 0.5) / n
        if wall == "links":
            pts.append((buffet["x"] + buffet["w"] + gap, buffet["y"] + f * buffet["h"]))
        elif wall == "rechts":
            pts.append((buffet["x"] - gap,               buffet["y"] + f * buffet["h"]))
        elif wall == "boven":
            pts.append((buffet["x"] + f * buffet["w"],   buffet["y"] + buffet["h"] + gap))
        else:  # onder
            pts.append((buffet["x"] + f * buffet["w"],   buffet["y"] - gap))
    return pts


def _entrance_on_wall(w, h, wall):
    if wall == "links":  return {"x": 48,     "y": int(h * 0.75)}
    if wall == "rechts": return {"x": w - 48, "y": int(h * 0.75)}
    if wall == "boven":  return {"x": int(w * 0.25), "y": 48}
    return {"x": int(w * 0.25), "y": h - 48}


# ── Archetypen ───────────────────────────────────────────────────────────────

def _overlap(a, b, pad=0):
    return (a["x"] - pad < b["x"] + b["w"] and a["x"] + a["w"] + pad > b["x"] and
            a["y"] - pad < b["y"] + b["h"] and a["y"] + a["h"] + pad > b["y"])


def _fits(block, bar, buffet, entrance, pad=20):
    """Botst dit blok met meubilair of de deur?"""
    door = {"x": entrance["x"] - 55, "y": entrance["y"] - 45, "w": 110, "h": 90}
    dock = {"x": bar["dock"]["x"] - 30, "y": bar["dock"]["y"] - 30, "w": 60, "h": 60}
    if _overlap(block, bar, pad):        return False
    if buffet and _overlap(block, buffet, pad): return False
    if _overlap(block, door):            return False
    if _overlap(block, dock):            return False
    return True


def _first_fitting(cands, bar, buffet, entrance):
    """
    Eerste plaatsing die niet botst, anders niets.

    Op wandniveau redeneren was te grof: staan bar en buffet tegenover elkaar,
    dan blijft er geen hoek over met twee vrije wanden en werd 59% van de
    L-vormen alsnog een kale rechthoek. Direct op de meubelrechthoeken toetsen
    laat veel meer plaatsingen toe.
    """
    for c in cands:
        if _fits(c, bar, buffet, entrance):
            return [c]
    return []


def _blocks_for(kind, w, h, rng, bar, buffet, entrance):
    """
    De vloer die dit archetype wegneemt, geplaatst waar het meubilair niet staat.

    Een zaal snijdt zijn hoek in het echt ook niet weg op de plek waar de tap
    zit, en een podium staat niet in de deuropening.
    """
    if kind == "kolommen":
        # Dragende kolommen in een raster: midden in de zaal, verplaatsen zich
        # niet. Het meest genoemde praktijkprobleem. Botsende kolommen vallen
        # gewoon af; de rest blijft staan.
        cols, rows = int(rng.integers(2, 4)), int(rng.integers(1, 3))
        size = int(rng.integers(26, 40))
        out = [{"x": int(w * (i + 1) / (cols + 1) - size / 2),
                "y": int(h * (j + 1) / (rows + 1) - size / 2),
                "w": size, "h": size}
               for i in range(cols) for j in range(rows)]
        return [b for b in out if _fits(b, bar, buffet, entrance, pad=10)]

    if kind == "l-vorm":
        # Past de grote hoek niet, probeer een kleinere voordat we het opgeven:
        # een ondiepe knik is nog altijd een L-vorm, een kale rechthoek niet.
        for scale in (1.0, 0.75, 0.55):
            bw = int(w * rng.uniform(0.24, 0.34) * scale)
            bh = int(h * rng.uniform(0.24, 0.34) * scale)
            corners = [{"x": 0, "y": 0, "w": bw, "h": bh},
                       {"x": w - bw, "y": 0, "w": bw, "h": bh},
                       {"x": 0, "y": h - bh, "w": bw, "h": bh},
                       {"x": w - bw, "y": h - bh, "w": bw, "h": bh}]
            rng.shuffle(corners)
            got = _first_fitting(corners, bar, buffet, entrance)
            if got:
                return got
        return []

    if kind == "nis":
        t = 40
        cands = [{"x": 0, "y": int(h * f), "w": int(w * 0.55), "h": t} for f in (0.30, 0.62)]
        cands += [{"x": int(w * 0.45), "y": int(h * f), "w": int(w * 0.55), "h": t} for f in (0.30, 0.62)]
        cands += [{"x": int(w * f), "y": 0, "w": t, "h": int(h * 0.55)} for f in (0.30, 0.62)]
        cands += [{"x": int(w * f), "y": int(h * 0.45), "w": t, "h": int(h * 0.55)} for f in (0.30, 0.62)]
        rng.shuffle(cands)
        return _first_fitting(cands, bar, buffet, entrance)

    if kind == "podium":
        d = int(min(w, h) * rng.uniform(0.16, 0.24))
        cands = [{"x": int(w*0.25), "y": 0,     "w": int(w*0.50), "h": d},
                 {"x": int(w*0.25), "y": h - d, "w": int(w*0.50), "h": d},
                 {"x": 0,     "y": int(h*0.25), "w": d, "h": int(h*0.50)},
                 {"x": w - d, "y": int(h*0.25), "w": d, "h": int(h*0.50)}]
        rng.shuffle(cands)
        return _first_fitting(cands, bar, buffet, entrance)

    return []


def make_room(rng, kind=None):
    """Trekt een zaal. `rng` is een numpy Generator, zodat de seed doorwerkt."""
    kind = kind or KINDS[int(rng.integers(0, len(KINDS)))]

    if kind == "langwerpig":
        long_side = int(rng.integers(720, 900))
        ratio     = rng.uniform(1.6, 2.2)
        if rng.random() < 0.5:
            w, h = long_side, int(long_side / ratio)
        else:
            w, h = int(long_side / ratio), long_side
    else:
        w = int(rng.integers(520, 760))
        h = int(rng.integers(520, 760))

    walls = ["links", "rechts", "boven", "onder"]
    bar_wall = walls[int(rng.integers(0, 4))]
    rest     = [x for x in walls if x != bar_wall]

    has_buffet  = rng.random() < 0.75
    buffet_wall = rest[int(rng.integers(0, len(rest)))] if has_buffet else None
    ent_choices = [x for x in rest if x != buffet_wall] or rest
    ent_wall    = ent_choices[int(rng.integers(0, len(ent_choices)))]

    bar      = _bar_on_wall(w, h, bar_wall, rng)
    buffet   = _buffet_on_wall(w, h, buffet_wall, rng) if has_buffet else None
    entrance = _entrance_on_wall(w, h, ent_wall)
    # Een smalle langwerpige zaal kan bar en buffet laten overlappen; dan gaat
    # het buffet eruit in plaats van er half in te staan.
    if buffet and _overlap(bar, buffet, pad=10):
        buffet = None
    blocks = _blocks_for(kind, w, h, rng, bar, buffet, entrance)
    return {
        "kind":     kind,
        "w":        w,
        "h":        h,
        "bar":      bar,
        "buffet":   buffet,
        "entrance": entrance,
        "blocks":   blocks,
        "bar_wall": bar_wall,
    }


# Vrije vloer per zitplaats, geijkt op de klassieke zaal: 349.272 px2 vrije
# vloer voor 36 zitplaatsen. Dat is 2,2 m2 (23 sq ft) per gast -- hoger dan de
# cateringnorm van 10-15 sq ft, omdat bar en buffet wel vloer innemen maar geen
# zitplaats opleveren.
AREA_PER_SEAT = 9700


def table_mix_for(free_area):
    """
    Hoeveel tafels en gasten horen er in een zaal met deze vrije vloer?

    Zonder dit krijgt elke zaal dezelfde acht tafels, en dan is de vergelijking
    tussen zalen oneerlijk in twee richtingen tegelijk: een grote zaal krijgt
    onnodig lange looproutes, en een kleine zaal wint automatisch omdat alles
    dicht bij elkaar staat. Het aantal gasten hoort met de zaal mee te schalen,
    precies zoals een cateraar het zou plannen.

    De verhouding grote/middelgrote tafels blijft die van de klassieke zaal
    (2 large op 6 medium), en het aantal gasten blijft 36% boven het aantal
    zitplaatsen -- ook dat is hoe de bestaande dataset is opgebouwd.
    """
    seats = int(round(free_area / AREA_PER_SEAT))
    seats = max(16, min(72, seats))
    n_large  = max(1, int(round(seats / 18)))          # 2 large per 36 zitplaatsen
    rest     = seats - 6 * n_large
    n_medium = max(1, int(round(rest / 4)))
    total    = 6 * n_large + 4 * n_medium
    return {"tLarge": n_large, "tMedium": n_medium, "tSmall": 0,
            "seats": total, "guests": int(round(total * 49 / 36))}


# De klassieke zaal waarop alles tot nu toe is verzameld. Blijft de default,
# zodat oude data en oude aanroepen ongewijzigd blijven werken.
CLASSIC = {
    "kind": "rechthoek", "w": 640, "h": 640,
    "bar":      {"x": 550, "y": 50, "w": 70, "h": 540, "dock": {"x": 530, "y": 80}},
    "buffet":   {"x": 20, "y": 170, "w": 60, "h": 300, "slots": 3,
                 "dwellRange": [45, 90], "wall": "links"},
    "entrance": {"x": 48, "y": 586},
    "blocks":   [],
    "bar_wall": "rechts",
}
