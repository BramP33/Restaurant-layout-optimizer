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
  const out = await page.evaluate(list => list.map(tables => {
    const e = window.__engine;
    e._batchStart({ roomW: 640, roomH: 640, guests: 49, waiters: 3,
      tSmall: 0, tMedium: 0, tLarge: 0, partyType: 'buffet', gridSize: 24,
      forcedLayout: tables });
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
    print(f"{len(runs):,} runs -> {len(layouts):,} unieke layouts")

    # Het hulpscript moet naast node_modules staan, anders vindt require()
    # playwright niet: node zoekt vanaf de map van het bestand, niet vanaf cwd.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in.json").write_text(json.dumps(layouts))
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
    for i, (tabs, v) in enumerate(zip(layouts, js)):
        valid, unreach, trapped = pg.layout_valid(pg.build_blocked(tabs), tabs)
        if bool(valid) == bool(v["valid"]):
            agree += 1
        elif len(mismatches) < 10:
            mismatches.append((i, valid, v["valid"], unreach, v["unreach"], trapped, v["trapped"]))

    dis = len(layouts) - agree
    print(f"{len(layouts):,} layouts: {agree:,} eens, {dis:,} oneens "
          f"({100 * dis / max(1, len(layouts)):.2f}%)")
    for i, pv, jv, pu, ju, pt, jt in mismatches:
        print(f"  #{i}: py valid={pv} (onbereikbaar {pu}, opgesloten {pt}) "
              f"vs js valid={jv} (onbereikbaar {ju}, opgesloten {jt})")
    if dis:
        print("\nFAAL: de Python-spiegel wijkt af van de simulator.")
        return 1
    print("\nOK: spiegel en simulator zijn het overal eens.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
