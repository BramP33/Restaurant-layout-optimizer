"""
Kruiscontrole: geeft de Python-spiegel (pathgrid.py) hetzelfde oordeel als de
simulator zelf?

pathgrid.py bouwt hetzelfde loopraster in Python, zodat features (A*-afstanden)
en de voorzeef in collect_parallel.py niet elke keer een browser nodig hebben.
Die spiegel is alleen bruikbaar als hij exact hetzelfde zegt als simulatie.html.
Deze test vergelijkt het geldigheidsoordeel over de hele dataset.

Gebruik:
    python3 test_pathgrid.py                       # tegen restaurant-sim-merged.json
    python3 test_pathgrid.py --data ../foo.json --limit 500

Vereist dat ml/test_reachability.js eerst is gedraaid? Nee — deze test roept de
browser zelf aan via node, zodat beide kanten in één keer bepaald worden.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import pathgrid as pg  # noqa: E402


def layout_key(tables):
    return "|".join(sorted(
        f'{t["size"]},{round(t["x"])},{round(t["y"])},{round(t.get("rotation") or 0)}'
        for t in tables if t.get("size") != "custom"))


NODE_SNIPPET = r"""
const { chromium } = require('playwright');
const fs = require('fs'); const path = require('path');
(async () => {
  const layouts = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto('file://' + process.argv[3], { waitUntil: 'networkidle' });
  const out = await page.evaluate(list => list.map(item => {
    const e = window.__engine;
    const room = item.room;
    e._batchStart({ room, roomW: room.w, roomH: room.h, guests: 49, waiters: 3,
      tSmall: 0, tMedium: 0, tLarge: 0, partyType: 'buffet', gridSize: 24,
      forcedLayout: item.tables });
    const r = e._layoutReach();
    return { valid: r.valid, unreach: r.unreachable, trapped: r.trappedWaiters };
  }), layouts);
  fs.writeFileSync(process.argv[4], JSON.stringify(out));
  await browser.close();
})();
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "restaurant-sim-merged.json"))
    ap.add_argument("--limit", type=int, default=0, help="0 = alles")
    ap.add_argument("--rooms", type=int, default=0,
                    help="aantal gevarieerde zalen; 0 = alleen de klassieke zaal")
    args = ap.parse_args()

    raw = json.loads(Path(args.data).read_text())
    runs = raw if isinstance(raw, list) else raw.get("runs", raw.get("data", []))
    uniq = {}
    for r in runs:
        if not r.get("tables") or not r.get("metrics"):
            continue
        uniq.setdefault(layout_key(r["tables"]), r["tables"])
    layouts = [[t for t in tabs if t.get("size") != "custom"] for tabs in uniq.values()]
    if args.limit:
        layouts = layouts[:args.limit]
    import numpy as np
    import rooms as rm
    from optimize_layout import generate_batch

    items = [{"room": rm.CLASSIC, "tables": tabs} for tabs in layouts]

    if args.rooms:
        # Gevarieerde zalen zijn de eigenlijke reden dat deze test bestaat: de
        # spiegel is per definitie in orde op de zaal waarop hij geijkt is, en
        # breekt pas zodra de geometrie verandert.
        from optimize_layout import fit_table_mix
        rng = np.random.default_rng(20260907)
        for _ in range(args.rooms):
            room = rm.make_room(rng)
            types, _mix = fit_table_mix(room, rng)
            batch, _ = generate_batch(12, rng, room=room, types=types)
            for lay in batch[:6]:
                items.append({"room": room,
                              "tables": [t for t in lay if t.get("size") != "custom"]})
        print(f"  + {args.rooms} gevarieerde zalen")

    print(f"{len(runs):,} runs -> {len(items):,} te toetsen indelingen")

    # Het hulpscript moet naast node_modules staan, anders vindt require()
    # playwright niet: node zoekt vanaf de map van het bestand, niet vanaf cwd.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in.json").write_text(json.dumps(items))
        runner = HERE / "_pathgrid_xcheck.tmp.js"
        runner.write_text(NODE_SNIPPET)
        try:
            proc = subprocess.run(
                ["node", str(runner), str(tmp / "in.json"),
                 str(ROOT / "simulatie.html"), str(tmp / "out.json")],
                cwd=HERE, capture_output=True, text=True,
            )
        finally:
            runner.unlink(missing_ok=True)
        if proc.returncode != 0:
            print("browser-oordeel mislukt:\n" + (proc.stderr or "")[-2000:])
            return 2
        js = json.loads((tmp / "out.json").read_text())

    agree = 0
    mismatches = []
    for i, (item, v) in enumerate(zip(items, js)):
        tabs, room = item["tables"], item["room"]
        valid, unreach, trapped = pg.layout_valid(
            pg.build_blocked(tabs, room), tabs, room=room)
        if bool(valid) == bool(v["valid"]):
            agree += 1
        elif len(mismatches) < 10:
            mismatches.append((i, valid, v["valid"], unreach, v["unreach"], trapped, v["trapped"]))

    dis = len(items) - agree
    print(f"{len(items):,} indelingen: {agree:,} eens, {dis:,} oneens "
          f"({100 * dis / max(1, len(items)):.2f}%)")
    for i, pv, jv, pu, ju, pt, jt in mismatches:
        print(f"  #{i}: py valid={pv} (onbereikbaar {pu}, opgesloten {pt}) "
              f"vs js valid={jv} (onbereikbaar {ju}, opgesloten {jt})")
    if dis:
        print("\nFAAL: de Python-spiegel wijkt af van de simulator.")
        return 1
    print("\nOK: spiegel en simulator zijn het overal eens.")
    return 0


def test_tour_features():
    """
    Tour-features op geldige layouts mogen nooit de strafwaarde raken.

    De eerste versie koos per tafel het eerste servicepunt met een vrije cel
    in de buurt, zonder te toetsen of dat punt op de obervloer lag. Een
    afgesloten nis telde dus mee, 15% van de geldige layouts kreeg een
    fantoomstraf van BIG px, en de correlaties draaiden van teken om. Deze
    test bewaakt precies dat: geldige layout in, eindige afstanden uit.
    """
    import json
    import numpy as np
    import pathgrid as pg

    # De ZAAL moet mee. Sinds de dataset meerdere zaalvormen bevat, sloeg deze
    # test elke layout tegen pathgrid.CLASSIC aan -- een indeling uit een zaal
    # van 840x508 werd getoetst in een zaal van 640x640, en viel dan terecht om
    # op "testset moet geldig zijn". De test mat dus niet de tour-features maar
    # de verkeerde zaal.
    from train_surrogate import run_room

    runs = json.loads((ROOT / "restaurant-sim-clean.json").read_text())
    seen, layouts = set(), []
    for r in runs:
        var = [t for t in r["tables"] if t["size"] != "custom"]
        if not var:
            continue
        var.sort(key=lambda t: (-t["w"], t["x"], t["y"]))
        room = run_room(r)
        key = (room["kind"], room["w"], room["h"],
               tuple((round(t["x"], 1), round(t["y"], 1), t["size"], t["rotation"])
                     for t in var))
        if key in seen:
            continue
        seen.add(key)
        layouts.append((var, room))
        if len(layouts) >= 300:
            break

    bad = 0
    for lay, room in layouts:
        # De strafwaarde is zaalafhankelijk (alle vrije cellen achter elkaar),
        # dus hij hoort per zaal berekend te worden en niet uit ROOM_W.
        #
        # MET de tafels erin, precies zoals tour_features hem zelf berekent.
        # Op een LEGE zaal ligt de drempel mediaan 28% te hoog, en dan valt de
        # helft van de wacht dood: een fantoomstraf op de vier
        # paarafstandsfeatures (12.349 px) zou onder een lege-zaaldrempel van
        # 15.174 px doorglippen. De test vuurde dan alleen nog op de ritkosten,
        # die toevallig ~2x de straf optellen.
        big = float((~pg.build_blocked(lay, room)).sum()) * pg.CELL
        assert pg.layout_valid(pg.build_blocked(lay, room), lay, room=room)[0], \
            "testset moet geldig zijn"
        f = pg.tour_features(lay, room)
        assert len(f) == pg.N_TOUR_FEATURES, f"{len(f)} != {pg.N_TOUR_FEATURES}"
        if max(f[:4]) >= big or f[4] >= big:
            bad += 1

    print(f"tour_features: {len(layouts)} geldige layouts, {bad} met strafwaarde")
    assert bad == 0, f"{bad} geldige layouts kregen een fantoomstraf"
    print("OK: tour-features raken de strafwaarde niet op geldige layouts.")


if __name__ == "__main__":
    rc = main()
    if rc == 0:
        test_tour_features()
    sys.exit(rc)
