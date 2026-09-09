"""
Vertaling tussen het app-model (meters, vrije vorm) en het simulatiemodel
(pixels, 67 per meter, rechthoekige zaal met blokken).

Het app-model:
    rects      lijst rechthoeken in meters; de vereniging is de vloer
    bar        {"wall","x","y","w","h"} tegen een wandsegment
    buffet     idem, of None
    entrance   {"x","y"} net binnen de wand
    blocks     losse kolommen/podia in meters
    tables     {"small": n, "medium": n, "large": n, ...}
    currentLayout  eigen indeling: [{"type","x","y","rotation"}] in meters

Het simulatiemodel staat beschreven in ml/rooms.py. Alles wat hier gebeurt
is meetkunde plus eenheden; gedrag van gasten en obers blijft in de
simulator zelf.
"""

from __future__ import annotations

PX_PER_M = 67          # zie ml/rooms.py
BAR_DEPTH_PX = 70      # diepte van een bar tegen de wand (rooms._bar_on_wall)
BUFFET_DEPTH_PX = 60   # diepte van een buffetlijn (rooms._buffet_on_wall)

# Vaste tafeltypes van de simulator (simulatie.html DEFAULT_TABLE_TYPE_DEFS).
TABLE_TYPES_PX = {
    "small":  {"w": 60,  "h": 60, "seats": 2},
    "medium": {"w": 80,  "h": 60, "seats": 4},
    "large":  {"w": 110, "h": 70, "seats": 6},
}
GENERATABLE = ("large", "medium", "small")


def m2px(v: float) -> float:
    return v * PX_PER_M


def px2m(v: float) -> float:
    return v / PX_PER_M


# ── Vorm: vereniging van rechthoeken -> omhullende + blokken ─────────────

def _bbox(rects):
    x0 = min(r["x"] for r in rects)
    y0 = min(r["y"] for r in rects)
    x1 = max(r["x"] + r["w"] for r in rects)
    y1 = max(r["y"] + r["h"] for r in rects)
    return x0, y0, x1, y1


def complement_blocks(rects, eps=1e-6):
    """
    Rechthoeken die de omhullende wel bedekt maar de vloer niet.

    Rasterdecompositie op alle randcoordinaten: elke cel is ofwel helemaal
    vloer ofwel helemaal geen vloer. Niet-vloercellen worden per rij
    horizontaal samengevoegd, en gelijke rijen daarna verticaal.
    """
    if not rects:
        return []
    xs = sorted({v for r in rects for v in (r["x"], r["x"] + r["w"])})
    ys = sorted({v for r in rects for v in (r["y"], r["y"] + r["h"])})

    def covered(cx, cy):
        return any(r["x"] - eps <= cx <= r["x"] + r["w"] + eps and
                   r["y"] - eps <= cy <= r["y"] + r["h"] + eps for r in rects)

    rows = []
    for j in range(len(ys) - 1):
        cy = (ys[j] + ys[j + 1]) / 2
        spans, start = [], None
        for i in range(len(xs) - 1):
            cx = (xs[i] + xs[i + 1]) / 2
            free = not covered(cx, cy)
            if free and start is None:
                start = xs[i]
            if not free and start is not None:
                spans.append((start, xs[i]))
                start = None
        if start is not None:
            spans.append((start, xs[-1]))
        rows.append((ys[j], ys[j + 1], spans))

    # Verticaal samenvoegen van identieke spans in opeenvolgende rijen.
    out, active = [], {}                  # (x0,x1) -> y-start
    for y0, y1, spans in rows:
        cur = set(spans)
        for s in [s for s in active if s not in cur]:
            ystart = active.pop(s)
            out.append({"x": s[0], "y": ystart, "w": s[1] - s[0], "h": y0 - ystart})
        for s in cur:
            active.setdefault(s, y0)
    for s, ystart in active.items():
        out.append({"x": s[0], "y": ystart, "w": s[1] - s[0], "h": ys[-1] - ystart})
    return [b for b in out if b["w"] > eps and b["h"] > eps]


# ── Meubilair ─────────────────────────────────────────────────────────────

def _bar_px(bar_m, ox, oy):
    """Bar in px, met de dock op dezelfde plek als rooms._bar_on_wall."""
    x = m2px(bar_m["x"] - ox)
    y = m2px(bar_m["y"] - oy)
    w = m2px(bar_m["w"])
    h = m2px(bar_m["h"])
    wall = bar_m["wall"]
    if wall == "links":
        dock = {"x": x + w + 20, "y": y + 30}
    elif wall == "rechts":
        dock = {"x": x - 20, "y": y + 30}
    elif wall == "boven":
        dock = {"x": x + 30, "y": y + h + 20}
    else:
        dock = {"x": x + 30, "y": y - 20}
    return {"x": round(x), "y": round(y), "w": round(w), "h": round(h),
            "dock": {"x": round(dock["x"]), "y": round(dock["y"])}}


def _buffet_px(bf_m, ox, oy):
    return {"x": round(m2px(bf_m["x"] - ox)), "y": round(m2px(bf_m["y"] - oy)),
            "w": round(m2px(bf_m["w"])), "h": round(m2px(bf_m["h"])),
            "slots": int(bf_m.get("slots", 3)), "dwellRange": [45, 90],
            "wall": bf_m["wall"]}


def _table_px(t_m, ox, oy, table_types):
    typ = t_m["type"]
    if typ in TABLE_TYPES_PX:
        d = TABLE_TYPES_PX[typ]
        return {"size": typ, "x": m2px(t_m["x"] - ox), "y": m2px(t_m["y"] - oy),
                "rotation": int(t_m.get("rotation", 0)),
                "w": d["w"], "h": d["h"], "seats": d["seats"]}
    d = table_types.get(typ, {})
    return {"size": "custom", "x": m2px(t_m["x"] - ox), "y": m2px(t_m["y"] - oy),
            "rotation": int(t_m.get("rotation", 0)),
            "w": round(m2px(d.get("w", 1.2)), 2), "h": round(m2px(d.get("h", 0.8)), 2),
            "seats": int(d.get("seats", 4)), "name": d.get("name", typ)}


# ── Hoofdconversie ────────────────────────────────────────────────────────

def project_to_job(project: dict) -> dict:
    """
    Zet een app-project om in de invoer voor run_optimizer.py.

    Geeft een dict met `room` (px), `types` (tafellijst voor de generator),
    `config` (simulatie-instellingen), `current` (eigen indeling in px, of
    None), `origin` (verschuiving in meters) en `warnings`.
    """
    warnings = []
    rects = project.get("rects") or []
    if not rects:
        raise ValueError("Teken eerst de vloer van de zaal.")
    if not project.get("bar"):
        raise ValueError("Plaats een bar tegen een wand.")
    if not project.get("entrance"):
        raise ValueError("Plaats een ingang.")
    party = project.get("party") or {}
    has_buffet = bool(party.get("hasBuffet"))
    if has_buffet and not project.get("buffet"):
        raise ValueError("Het feest heeft een buffet, maar er staat geen buffetlijn in de zaal.")

    x0, y0, x1, y1 = _bbox(rects)
    w_px, h_px = round(m2px(x1 - x0)), round(m2px(y1 - y0))
    if w_px < 300 or h_px < 300:
        raise ValueError("De zaal is te klein: minimaal ongeveer 4,5 bij 4,5 meter.")
    if w_px > 2400 or h_px > 2400:
        raise ValueError("De zaal is te groot: maximaal ongeveer 35 bij 35 meter.")

    blocks = []
    for b in complement_blocks(rects):
        blocks.append({"x": round(m2px(b["x"] - x0)), "y": round(m2px(b["y"] - y0)),
                       "w": round(m2px(b["w"])), "h": round(m2px(b["h"]))})
    for b in project.get("blocks") or []:
        blocks.append({"x": round(m2px(b["x"] - x0)), "y": round(m2px(b["y"] - y0)),
                       "w": round(m2px(b["w"])), "h": round(m2px(b["h"]))})

    bar = _bar_px(project["bar"], x0, y0)
    buffet = _buffet_px(project["buffet"], x0, y0) if (has_buffet and project.get("buffet")) else None
    if project.get("buffet") and not has_buffet:
        warnings.append("Er staat een buffetlijn getekend, maar het feest heeft geen buffet. De lijn wordt weggelaten.")
    ent = project["entrance"]
    entrance = {"x": round(m2px(ent["x"] - x0)), "y": round(m2px(ent["y"] - y0))}

    room = {
        "kind": "eigen",
        "w": w_px, "h": h_px,
        "bar": bar,
        "buffet": buffet,
        "entrance": entrance,
        "blocks": blocks,
        "bar_wall": project["bar"]["wall"],
    }

    counts = project.get("tables") or {}
    types = []
    for typ in GENERATABLE:
        types += [typ] * int(counts.get(typ, 0) or 0)
    if not types:
        raise ValueError("Geef minstens één kleine, middelgrote of grote tafel op.")
    if len(types) > 40:
        raise ValueError("Meer dan 40 tafels wordt nog niet ondersteund.")
    custom_n = sum(int(v or 0) for k, v in counts.items() if k not in GENERATABLE)
    if custom_n:
        warnings.append(f"{custom_n} eigen tafel(s) worden bewaard en getekend, maar de generatie kent ze nog niet en laat ze weg.")
    if counts.get("small"):
        warnings.append("Kleine tafels zitten nauwelijks in de trainingsdata; de voorspelling daarvoor is minder betrouwbaar.")

    seats = sum(TABLE_TYPES_PX[t]["seats"] for t in types)
    guests = int(party.get("guests") or round(seats * 49 / 36))
    party_type = party.get("type") or "buffet"
    if party_type not in ("receptie", "buffet", "feestavond"):
        party_type = "buffet"
    duration_h = float(party.get("durationH") or 0)
    pop = project.get("population") or {}
    config = {
        "room": room,
        "roomW": w_px, "roomH": h_px,
        "guests": guests,
        "waiters": int(party.get("waiters") or 3),
        "tSmall": 0, "tMedium": 0, "tLarge": 0,
        "partyType": party_type,
        "gridSize": 24,
        "population": {
            "avgAge":      float(pop.get("avgAge", 40)),
            "ageSpread":   float(pop.get("ageSpread", 15)),
            "thirst":      float(pop.get("thirst", 1.0)),
            "appetite":    float(pop.get("appetite", 1.0)),
            "waiterSkill": float(pop.get("waiterSkill", 0.5)),
            "skillSpread": 0.2,
        },
    }
    if duration_h > 0:
        config["durationSec"] = int(duration_h * 3600)

    current = None
    cur = project.get("currentLayout") or []
    if cur:
        tt = project.get("tableTypes") or {}
        current = [_table_px(t, x0, y0, tt) for t in cur]

    return {
        "room": room,
        "types": types,
        "seats": seats,
        "config": config,
        "current": current,
        "origin": {"x": x0, "y": y0},
        "warnings": warnings,
    }


def tables_to_m(tables_px, origin):
    """Tafels uit de simulator terug naar app-coordinaten (meters, linksboven)."""
    out = []
    for t in tables_px:
        out.append({
            "type": t.get("size", "medium"),
            "x": px2m(t["x"]) + origin["x"],
            "y": px2m(t["y"]) + origin["y"],
            "rotation": int(t.get("rotation", 0)),
            "w": px2m(t.get("w", 80)), "h": px2m(t.get("h", 60)),
            "seats": int(t.get("seats", 4)),
        })
    return out
