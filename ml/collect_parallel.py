"""
Parallelle dataverzamelaar — bedoeld voor een VM met veel CPU-kernen.

De bestaande collect_overnight.py draait één browser sequentieel en schrijft
direct in de gedeelde dataset. Op een machine met 16 kernen laat dat 15 kernen
onbenut, en parallel draaien zou races op dat ene bestand geven.

Deze versie start N werkers die elk hun eigen headless browser draaien en hun
resultaten in een eigen shard-bestand schrijven. Niets wordt gedeeld, dus er
valt niets te racen. Achteraf voegt merge_shards.py alles samen.

Elke run krijgt de bereikbaarheidsvlaggen mee (layoutValid, unreachableTables,
pathFailures) zodat onbruikbare indelingen later gefilterd kunnen worden in
plaats van als goedkope layouts in de training te belanden.

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int,   default=max(1, (os.cpu_count() or 2) // 2),
                   help="Aantal parallelle browsers (vuistregel: helft van de kernen)")
    p.add_argument("--batch",   type=int,   default=40,   help="Layouts per ronde per werker")
    p.add_argument("--seeds",   type=int,   default=3,    help="Seeds per layout")
    p.add_argument("--hours",   type=float, default=8.0,  help="Maximale looptijd in uren")
    p.add_argument("--tag",     type=str,   default="",   help="Label in de shard-bestandsnaam")
    return p.parse_args()


def generate_batch_file(n, out_path, rng_seed):
    """Schrijft n willekeurige geldige layouts in het formaat van validate_headless.js."""
    import numpy as np
    sys.path.insert(0, str(HERE))
    from optimize_layout import generate_batch, ROOM_W, ROOM_H

    rng = np.random.default_rng(rng_seed)
    layouts = []
    for _ in range(20):
        batch, _hit = generate_batch(n * 3, rng)
        layouts.extend(batch)
        if len(layouts) >= n:
            break
    layouts = layouts[:n]
    if not layouts:
        return 0

    payload = [{
        "rank": i + 1,
        "predicted_waiterDist": 0,
        "predicted_score": 0,
        "config": {"roomW": ROOM_W, "roomH": ROOM_H, "guests": 49, "waiters": 3,
                   "tSmall": 0, "tMedium": 6, "tLarge": 2, "partyType": "buffet"},
        "tables": layout,
    } for i, layout in enumerate(layouts)]

    out_path.write_text(json.dumps(payload))
    return len(layouts)


def worker_round(worker_id, args, round_num):
    """Eén ronde voor één werker: genereer, simuleer, schrijf shard."""
    tmp_in  = SHARD_DIR / f"_in-w{worker_id}.json"
    tmp_out = SHARD_DIR / f"_out-w{worker_id}.json"

    n = generate_batch_file(args.batch, tmp_in, rng_seed=random.randrange(2**32))
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
    print(flush=True)

    # Elke werker is een los proces van dit script in --worker-modus; hier
    # draaien we ze als subprocessen zodat een crash de rest niet meesleept.
    running, totals, round_num = {}, {}, {}
    t_start = time.time()
    total_runs = 0

    while time.time() < deadline:
        for wid in range(args.workers):
            if wid in running:
                continue
            round_num[wid] = round_num.get(wid, 0) + 1
            running[wid] = subprocess.Popen(
                [sys.executable, str(HERE / "collect_parallel.py"),
                 "--workers", "1", "--batch", str(args.batch),
                 "--seeds", str(args.seeds), "--hours", "0",
                 "--tag", args.tag or "run"],
                env={**os.environ, "RS_WORKER_ID": str(wid),
                     "RS_ROUND": str(round_num[wid])},
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
    print(f"Voeg samen met:  python3 merge_shards.py")


def run_single():
    """Werker-modus: één ronde en klaar. Aangeroepen door de hoofdlus."""
    args = parse_args()
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    wid   = int(os.environ.get("RS_WORKER_ID", "0"))
    rnd   = int(os.environ.get("RS_ROUND", "1"))
    n, err = worker_round(wid, args, rnd)
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
