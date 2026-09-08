"""
Parallelle dataverzamelaar — bedoeld voor een VM met veel CPU-kernen.

De bestaande collect_overnight.py draait één browser sequentieel en schrijft
direct in de gedeelde dataset. Op een machine met 16 kernen laat dat 15 kernen
onbenut, en parallel draaien zou races op dat ene bestand geven.

Deze versie start N werkers die elk hun eigen headless browser draaien en hun
resultaten in een eigen shard-bestand schrijven. Niets wordt gedeeld, dus er
valt niets te racen. Achteraf voegt merge_shards.py alles samen.

Elke run krijgt de bereikbaarheidsvlaggen mee (layoutValid, unreachableTables,
trappedWaiters, pathFailures, waiterPathFailures) zodat onbruikbare indelingen
later gefilterd kunnen worden in plaats van als goedkope layouts in de training
te belanden.

Kandidaten worden bovendien vooraf gezeefd met pathgrid.layout_valid(), de
Python-spiegel van de bereikbaarheidscheck in de simulator. Op de bestaande
dataset is ongeveer twee derde van de willekeurige indelingen onbruikbaar; die
vooraf weggooien scheelt evenveel rekentijd, en de zeef kost milliseconden
tegenover seconden voor een simulatie.

Gebruik:
    python3 collect_parallel.py --workers 8 --hours 8
    python3 collect_parallel.py --workers 8 --hours 0.1   # korte proefrun
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
SHARD_DIR = ROOT / "data" / "shards"
POOL_PATH = ROOT / "data" / "room-pool.json"
MAX_FAILS = 12       # mislukte werkerrondes voordat de verzamelaar opgeeft


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int,   default=max(1, (os.cpu_count() or 2) // 2),
                   help="Aantal parallelle browsers (vuistregel: helft van de kernen)")
    p.add_argument("--batch",   type=int,   default=40,   help="Layouts per ronde per werker")
    p.add_argument("--seeds",   type=int,   default=3,    help="Seeds per layout")
    p.add_argument("--hours",   type=float, default=8.0,  help="Maximale looptijd in uren")
    p.add_argument("--tag",     type=str,   default="",   help="Label in de shard-bestandsnaam")
    p.add_argument("--rooms",   type=int,   default=0,
                   help="Verzamel diep in een vaste pool van N zalen in plaats van "
                        "elke ronde een nieuwe zaal (0 = nieuwe zaal per ronde)")
    p.add_argument("--pool-seed", type=int, default=20260907,
                   help="Seed voor de zaal-pool; dezelfde seed geeft dezelfde zalen")
    return p.parse_args()


WALLS = ["links", "rechts", "boven", "onder"]


def build_pool(n, seed, probe_n=120, min_yield=0.15, min_tables=6):
    """
    Trekt n zalen en legt ze vast in data/room-pool.json.

    Waarom een pool en niet elke ronde een verse zaal: met ~30 layouts per zaal
    is de rangschikking BINNEN een zaal niet te meten, en dat is precies wat de
    optimizer doet. Breed verzamelen leert het model welke zaal makkelijk is
    (rho 0.98); diep verzamelen moet het leren welke indeling beter is.

    De pool gaat naar een bestand in plaats van dat elke werker hem opnieuw
    afleidt: dan kan er geen verschil tussen processen ontstaan. Dat is in dit
    project al twee keer misgegaan.

    De pool wordt bewust GESTUURD in plaats van blind getrokken. Een blinde
    trekking gaf acht zalen die allemaal een buffet hadden (terwijl een kwart
    van de zalen er geen heeft), vijf keer dezelfde barwand, en twee zalen met
    maar 3-4 tafels. In zo'n zaal valt niets te ordenen, dus daar is diep
    verzamelen weggegooid budget: een kwart van de looptijd.
    """
    import numpy as np
    sys.path.insert(0, str(HERE))
    from optimize_layout import generate_batch, fit_table_mix
    import rooms as rm

    pool, attempt = [], 0
    while len(pool) < n and attempt < n * 400:
        i = len(pool)
        # Elk slot heeft een doelprofiel: archetype, barwand en wel/geen
        # buffet rouleren onafhankelijk van elkaar. Bij n=8 levert dat elk
        # archetype minstens één keer, alle vier de barwanden, en twee zalen
        # zonder buffet.
        want_kind   = rm.KINDS[i % len(rm.KINDS)]
        want_wall   = WALLS[i % len(WALLS)]
        want_buffet = (i % 4) != 3

        rng = np.random.default_rng(seed * 1000 + attempt)
        attempt += 1
        room = rm.make_room(rng, kind=want_kind)
        # Eerst de goedkope eisen; de probe hieronder kost een seconde.
        if room["bar_wall"] != want_wall:
            continue
        if (room.get("buffet") is not None) != want_buffet:
            continue

        types, mix = fit_table_mix(room, rng)
        # Een zaal met drie tafels heeft nauwelijks indelingen om tussen te
        # kiezen; het model kan daar niets leren wat de optimizer helpt.
        if len(types) < min_tables:
            continue
        # Een zaal waar bijna niets in past levert een ronde lang niets op.
        # Dit is de MEETKUNDIGE opbrengst (voor de dock-nazeef): een zaal
        # afkeuren omdat veel indelingen daar een krappe bardock hebben zou
        # goede zalen uit de pool gooien.
        batch, yield_ = generate_batch(probe_n, rng, room=room, types=types)
        if yield_ < min_yield:
            continue
        pool.append({"room": room, "types": types, "mix": mix,
                     "yield": round(yield_, 3), "attempt": attempt - 1,
                     "n_tables": len(types)})

    if len(pool) < n:
        raise SystemExit(f"kon maar {len(pool)}/{n} bruikbare zalen trekken "
                         f"in {attempt} pogingen")

    POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Stempel erbij, zodat een werker kan merken dat het bestand onder hem
    # veranderd is in plaats van stilzwijgend in een andere zaal te verzamelen.
    payload = {"seed": seed, "n": n, "pool": pool}
    tmp = POOL_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(POOL_PATH)          # atomair, anders leest een werker een half bestand
    return pool


def load_pool(expect_seed=None, expect_n=None):
    if not POOL_PATH.exists():
        raise SystemExit(f"zaal-pool ontbreekt: {POOL_PATH}")
    data = json.loads(POOL_PATH.read_text())
    if not isinstance(data, dict) or "pool" not in data:
        raise SystemExit(f"zaal-pool heeft een oud formaat: {POOL_PATH}")
    if expect_seed is not None and data["seed"] != expect_seed:
        raise SystemExit(f"zaal-pool is van seed {data['seed']}, verwacht {expect_seed} "
                         "-- draait er een tweede verzamelaar?")
    if expect_n is not None and data["n"] != expect_n:
        raise SystemExit(f"zaal-pool heeft {data['n']} zalen, verwacht {expect_n}")
    return data["pool"]


def generate_batch_file(n, out_path, rng_seed, pool_entry=None):
    """Schrijft n willekeurige geldige layouts in het formaat van validate_headless.js."""
    import numpy as np
    sys.path.insert(0, str(HERE))
    from optimize_layout import generate_batch, fit_table_mix

    import pathgrid as pg
    import rooms as rm

    rng = np.random.default_rng(rng_seed)

    if pool_entry is not None:
        # Diepe modus: de zaal ligt vast, alleen de indelingen variëren. De rng
        # blijft willekeurig, zodat opeenvolgende rondes andere layouts in
        # dezelfde zaal opleveren.
        room, types, mix = pool_entry["room"], pool_entry["types"], pool_entry["mix"]
    else:
        # Elke ronde een andere zaal. Zonder dit leert het model een zaal in
        # plaats van een indelingsprincipe. fit_room laat de zaal groeien tot
        # het meubilair er redelijk in past, want zes kolommen in een kleine
        # zaal leveren nul plaatsbare layouts.
        room = rm.make_room(rng)
        # Tafels en gasten schalen met de zaal. Een vast aantal zou de
        # vergelijking tussen zalen scheeftrekken: een grote zaal krijgt dan
        # onnodig lange looproutes en een kleine zaal wint automatisch.
        types, mix = fit_table_mix(room, rng)
    print(f"  zaal: {room['kind']} {room['w']}x{room['h']}, bar {room['bar_wall']}, "
          f"{len(room['blocks'])} blokken -> {mix['tLarge']}L+{mix['tMedium']}M "
          f"({mix['seats']} zitplaatsen, {mix['guests']} gasten)", flush=True)

    layouts, seen, rejected = [], 0, 0
    for _ in range(20):
        batch, _hit = generate_batch(n * 3, rng, room=room, types=types)
        for layout in batch:
            seen += 1
            # Zeef vooraf: een indeling waarin een ober opgesloten staat of een
            # tafel onbereikbaar is, hoeft niet gesimuleerd te worden. De
            # uitkomst zou toch weggegooid worden door merge_shards.
            #
            # Toets de wereld die de simulator opbouwt: alleen de variabele
            # tafels. De custom tafels die generate_batch meelevert komen nooit
            # op de vloer terecht (validate_headless.js stuurt ze niet mee, en
            # simulatie.html plaatst ze alleen uit localStorage) -- zie de
            # toelichting in pathgrid.py. Ze toch als obstakel meerekenen keurt
            # indelingen af die in de simulatie prima lopen.
            var = [t for t in layout if t.get("size") != "custom"]
            valid, _unreach, _trapped = pg.layout_valid(
                pg.build_blocked(var, room), var, room=room)
            if valid:
                layouts.append(layout)
            else:
                rejected += 1
        if len(layouts) >= n:
            break
    layouts = layouts[:n]
    if seen:
        print(f"  zeef: {rejected}/{seen} kandidaten vooraf afgekeurd", flush=True)
    if not layouts:
        return 0

    payload = [{
        "rank": i + 1,
        "predicted_waiterDist": 0,
        "predicted_score": 0,
        "config": {"room": room, "roomW": room["w"], "roomH": room["h"],
                   "guests": mix["guests"], "waiters": 3,
                   "tSmall": 0, "tMedium": mix["tMedium"], "tLarge": mix["tLarge"],
                   "partyType": "buffet"},
        "tables": layout,
    } for i, layout in enumerate(layouts)]

    out_path.write_text(json.dumps(payload))
    return len(layouts)


def worker_round(worker_id, args, round_num, pool_entry=None):
    """Eén ronde voor één werker: genereer, simuleer, schrijf shard."""
    tmp_in  = SHARD_DIR / f"_in-w{worker_id}.json"
    tmp_out = SHARD_DIR / f"_out-w{worker_id}.json"

    n = generate_batch_file(args.batch, tmp_in, rng_seed=random.randrange(2**32),
                            pool_entry=pool_entry)
    if n == 0:
        return 0, "geen layouts gegenereerd"

    proc = subprocess.run(
        ["node", str(HERE / "validate_headless.js"),
         "--input", str(tmp_in), "--top", str(n),
         "--seeds", str(args.seeds), "--out", str(tmp_out)],
        cwd=HERE, capture_output=True, text=True,
    )
    if proc.returncode != 0 or not tmp_out.exists():
        return 0, (proc.stderr or "onbekende fout")[-300:]

    results = json.loads(tmp_out.read_text())
    entries = []
    for r in results:
        entries.extend(r.get("trainingRuns") or [])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    tag   = f"-{args.tag}" if args.tag else ""
    shard = SHARD_DIR / f"shard{tag}-w{worker_id}-r{round_num}-{stamp}.json"
    shard.write_text(json.dumps(entries))

    tmp_in.unlink(missing_ok=True)
    tmp_out.unlink(missing_ok=True)
    return len(entries), None


def main():
    args = parse_args()
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    deadline = time.time() + args.hours * 3600
    print(f"Parallelle verzamelaar")
    print(f"  Werkers:   {args.workers} (machine heeft {os.cpu_count()} kernen)")
    print(f"  Batch:     {args.batch} layouts x {args.seeds} seeds per werker per ronde")
    print(f"  Looptijd:  {args.hours} uur")
    print(f"  Shards:    {SHARD_DIR}")

    pool = None
    if args.rooms:
        print(f"  Zalen:     vaste pool van {args.rooms} (seed {args.pool_seed})", flush=True)
        pool = build_pool(args.rooms, args.pool_seed)
        for i, e in enumerate(pool):
            r, m = e["room"], e["mix"]
            print(f"    [{i}] {r['kind']:<11} {r['w']}x{r['h']}  bar {r['bar_wall']:<7} "
                  f"{len(r['blocks'])} blokken  {m['tLarge']}L+{m['tMedium']}M  "
                  f"{m['guests']} gasten  (opbrengst {e['yield']:.0%})")
        print(f"    pool: {POOL_PATH}")
    else:
        print(f"  Zalen:     nieuwe zaal per ronde")
    print(flush=True)

    # Elke werker is een los proces van dit script in --worker-modus; hier
    # draaien we ze als subprocessen zodat een crash de rest niet meesleept.
    running, totals, round_num = {}, {}, {}
    t_start = time.time()
    total_runs = 0
    # Rondes gaan om de beurt naar elke zaal, zodat het budget gelijk verdeeld
    # wordt in plaats van dat de snelste zaal de dataset domineert.
    dispatched, per_room, fails = 0, {}, 0

    while time.time() < deadline:
        for wid in range(args.workers):
            if wid in running:
                continue
            round_num[wid] = round_num.get(wid, 0) + 1
            env = {**os.environ, "RS_WORKER_ID": str(wid),
                   "RS_ROUND": str(round_num[wid])}
            if pool:
                idx = dispatched % len(pool)
                env["RS_ROOM_INDEX"] = str(idx)
                per_room[idx] = per_room.get(idx, 0) + 1
            else:
                # Een blijven hangende variabele uit de omgeving zou een brede
                # run stilzwijgend diep maken.
                env.pop("RS_ROOM_INDEX", None)
            dispatched += 1
            running[wid] = subprocess.Popen(
                [sys.executable, str(HERE / "collect_parallel.py"),
                 "--workers", "1", "--batch", str(args.batch),
                 "--seeds", str(args.seeds), "--hours", "0",
                 "--rooms", str(args.rooms), "--pool-seed", str(args.pool_seed),
                 "--tag", args.tag or "run"],
                env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )

        time.sleep(2)
        for wid, proc in list(running.items()):
            if proc.poll() is None:
                continue
            out, err = proc.communicate()
            done = 0
            for line in (out or "").splitlines():
                if line.startswith("SHARD_RUNS="):
                    done = int(line.split("=", 1)[1])
            if proc.returncode != 0:
                print(f"  werker {wid}: fout — {(err or '').strip()[-200:]}", flush=True)
                fails += 1
                # Zonder rem blijft de lus uren lang processen starten die
                # allemaal meteen omvallen, en dat ziet er in het log uit als
                # een run die gewoon draait.
                if fails >= MAX_FAILS:
                    print(f"\n{fails} mislukte rondes op rij-ish — gestopt. "
                          "Zie de fout hierboven.", flush=True)
                    deadline = 0
            else:
                fails = 0
            total_runs += done
            totals[wid] = totals.get(wid, 0) + done
            del running[wid]

        elapsed = time.time() - t_start
        if elapsed > 0 and int(elapsed) % 60 < 2:
            rate = total_runs / elapsed * 3600
            print(f"  [{elapsed/60:5.1f} min] {total_runs:>6} runs  "
                  f"~{rate:,.0f}/uur  resterend {(deadline-time.time())/3600:.1f}u",
                  flush=True)

    for proc in running.values():
        proc.terminate()
    print(f"\nKlaar: {total_runs} runs in {(time.time()-t_start)/3600:.2f} uur")
    if pool:
        print("Rondes per zaal: " +
              ", ".join(f"[{i}] {per_room.get(i, 0)}" for i in range(len(pool))))
    print(f"Voeg samen met:  python3 merge_shards.py")


def run_single():
    """Werker-modus: één ronde en klaar. Aangeroepen door de hoofdlus."""
    args = parse_args()
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    wid   = int(os.environ.get("RS_WORKER_ID", "0"))
    rnd   = int(os.environ.get("RS_ROUND", "1"))
    entry = None
    if os.environ.get("RS_ROOM_INDEX") is not None:
        # De pool komt uit het bestand dat de hoofdlus schreef, niet uit een
        # herberekening hier: dezelfde zaal moet in elk proces bit voor bit
        # dezelfde zaal zijn. De stempel vangt af dat een tweede verzamelaar
        # het bestand onder deze run vandaan heeft geschreven.
        entry = load_pool(expect_seed=args.pool_seed,
                          expect_n=args.rooms or None)[int(os.environ["RS_ROOM_INDEX"])]
    elif args.rooms:
        raise SystemExit("--rooms gezet maar RS_ROOM_INDEX ontbreekt; "
                         "deze werker zou een willekeurige zaal pakken")
    n, err = worker_round(wid, args, rnd, pool_entry=entry)
    if err:
        print(f"fout: {err}", file=sys.stderr)
        sys.exit(1)
    print(f"SHARD_RUNS={n}")


if __name__ == "__main__":
    # --hours 0 betekent: ik ben een werker, doe één ronde.
    if "--hours" in sys.argv and sys.argv[sys.argv.index("--hours") + 1] == "0":
        run_single()
    else:
        main()
