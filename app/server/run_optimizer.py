"""
Draait de bestaande optimizer voor een app-job, als los proces.

Gebruik:
    python run_optimizer.py <jobmap>

Leest <jobmap>/input.json (uitvoer van convert.project_to_job plus een
`quality`-blok), schrijft <jobmap>/optimizer-results.json in het formaat dat
ml/validate_headless.js verwacht, en meldt voortgang op stdout als regels die
met `@@ ` beginnen gevolgd door JSON. Alle andere uitvoer is logtekst van de
optimizer zelf en wordt door de server doorgegeven als log.

Waarom een los proces: de optimizer rekent minutenlang met numpy en houdt de
GIL vast; in het serverproces zou de API dan niet meer antwoorden. En een
crash in de optimizer mag de server niet meenemen.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "ml"))

import joblib                                   # noqa: E402
import optimize_layout as opt                   # noqa: E402
import pathgrid as pg                           # noqa: E402


def progress(stage, pct, msg, **extra):
    d = {"stage": stage, "pct": round(float(pct), 1), "msg": msg}
    d.update(extra)
    print("@@ " + json.dumps(d), flush=True)


QUALITY = {
    # kandidaten, verfijn top-k, verfijnrondes, validatie-kopgroep. Geijkt op
    # deze machine: de brede zoektocht haalt ~40 kandidaten per seconde (het
    # basismodel rekent tour-features mee), de verfijning ~2 s per ronde bij
    # top-20.
    #
    # validate_top > top: het model scheidt slechte van goede layouts goed
    # (rho ~0,86) maar ordent BINNEN de goede groep zwak (kopgroep-rho ~0,52,
    # tegen een plafond van ~0,70 -- zie ml/decompose_ranking.py). Wie alleen
    # de top-`top` modelkandidaten simuleert, mist dus vaak de echte beste
    # layout. Simuleer daarom een bredere kopgroep en laat de simulator zelf
    # kiezen welke `top` daarvan aan de gebruiker getoond worden.
    "snel":    {"candidates": 4_000,  "refine_top": 15, "refine_rounds": 40,  "validate_top": 10},
    "normaal": {"candidates": 12_000, "refine_top": 25, "refine_rounds": 120, "validate_top": 16},
    "grondig": {"candidates": 40_000, "refine_top": 40, "refine_rounds": 300, "validate_top": 24},
}


def main(job_dir):
    job_dir = Path(job_dir)
    inp = json.loads((job_dir / "input.json").read_text())
    room, types, cfg = inp["room"], inp["types"], inp["config"]
    q = QUALITY.get(inp.get("quality", "normaal"), QUALITY["normaal"])
    top_n = int(inp.get("top", 3))
    # Zie de toelichting bij QUALITY: exporteer een bredere kopgroep dan de
    # gebruiker uiteindelijk te zien krijgt, zodat de validatiestap zo
    # dadelijk uit meer kandidaten kan kiezen dan alleen het modeloordeel.
    validate_top = max(top_n, q["validate_top"])

    progress("laden", 1, "Model laden")
    saved = joblib.load(ROOT / "surrogate_model.pkl")
    model, feat_len = saved["model"], len(saved["feature_names"])
    model_frontier = saved.get("model_frontier")
    feat_len_frontier = saved.get("feat_len_frontier")
    print(f"Model: {saved['model_name']} — {feat_len} features")
    print(f"Zaal: {room['w']}x{room['h']} px, bar {room.get('bar_wall')}, "
          f"{len(room['blocks'])} blokken, buffet {'ja' if room.get('buffet') else 'nee'}")
    print(f"Tafels: {len(types)} ({', '.join(types)})")

    # Past het uberhaupt? Een korte proef voordat we minuten gaan rekenen.
    rng = np.random.default_rng(42)
    probe, geom_yield = opt.generate_batch(300, rng, room=room, types=types)
    print(f"Proefplaatsing: {geom_yield*100:.1f}% van de pogingen past")
    if not probe:
        progress("fout", 100, "De tafels passen niet in deze zaal. Haal tafels weg of maak de zaal groter.",
                 error=True)
        sys.exit(2)
    if geom_yield < 0.02:
        print("Waarschuwing: de zaal zit erg vol; de zoektocht wordt daardoor smal.")

    # ── Fase 1: brede zoektocht in stappen, zodat we voortgang kunnen melden ──
    n_total = q["candidates"]
    steps = 8
    all_layouts, all_scores = [], []
    t0 = time.time()
    for i in range(steps):
        n = n_total // steps
        L, S = opt.random_search(model, feat_len, n, rng, batch=min(n, 25_000),
                                 room=room, types=types)
        all_layouts += L
        all_scores += S
        best = min(all_scores) if all_scores else float("nan")
        progress("zoeken", 5 + 45 * (i + 1) / steps,
                 f"Brede zoektocht: {(i+1)*n:,} kandidaten, beste {best:,.0f} px".replace(",", "."),
                 best_px=best, n=(i + 1) * n)
    if not all_layouts:
        progress("fout", 100, "Geen enkele bereikbare indeling gevonden. Controleer bar, ingang en buffet.",
                 error=True)
        sys.exit(2)
    order = np.argsort(all_scores)
    layouts = [all_layouts[i] for i in order]
    scores = [all_scores[i] for i in order]
    print(f"Fase 1 totaal {time.time()-t0:.0f}s")

    # ── Fase 2: verfijning in stappen ──
    m2, f2, frontier = (model_frontier, feat_len_frontier, True) if model_frontier is not None \
        else (model, feat_len, False)
    chunks = 5
    rounds = max(1, q["refine_rounds"] // chunks)
    for i in range(chunks):
        layouts, scores = opt.local_refine(m2, f2, layouts, scores,
                                           top_k=q["refine_top"], n_rounds=rounds,
                                           rng=rng, frontier=frontier, room=room)
        progress("verfijnen", 50 + 40 * (i + 1) / chunks,
                 f"Verfijning ronde {(i+1)*rounds} van {rounds*chunks}, beste {scores[0]:,.0f} px".replace(",", "."),
                 best_px=float(scores[0]))

    # ── Uitvoer in het formaat van validate_headless.js ──
    results = []
    for rank, (layout, sc) in enumerate(zip(layouts[:validate_top], scores[:validate_top])):
        results.append({
            "rank": rank + 1,
            "predicted_waiterDist": round(float(sc)),
            "predicted_score": round(-float(sc) * 0.02 + 421 * 12, 1),
            "config": cfg,
            "tables": layout,
        })
    current = inp.get("current")
    if current:
        var = [t for t in current if t.get("size") != "custom"]
        blocked = pg.build_blocked(var, room)
        ok, unreachable, _trapped = pg.layout_valid(blocked, var, room=room)
        placement = opt.placement_ok(current, room) if var else False
        results.append({
            "rank": 0,
            "is_current": True,
            "predicted_waiterDist": None,
            "predicted_score": None,
            "layout_valid_precheck": bool(ok),
            "placement_ok_precheck": bool(placement),
            "unreachable_precheck": int(unreachable) if unreachable is not None else 0,
            "config": cfg,
            "tables": current,
        })
    (job_dir / "optimizer-results.json").write_text(json.dumps(results, indent=1))
    progress("optimizer-klaar", 90, f"{len(results)} indelingen klaar voor validatie")


if __name__ == "__main__":
    main(sys.argv[1])
