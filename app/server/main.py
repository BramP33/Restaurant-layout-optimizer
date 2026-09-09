"""
Zaalplanner — server.

Serveert de browser-app en biedt een kleine job-API die de bestaande
optimizer (ml/optimize_layout.py) en validator (ml/validate_headless.js)
aanstuurt. Draait lokaal; later kan hetzelfde proces op een server staan,
de browser-app hoeft daar niets voor te weten.

Starten:
    .venv/bin/uvicorn app.server.main:app --port 8765 --reload
of via app/start.sh
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
APP = HERE.parent
ROOT = APP.parent
WEB = APP / "web"
JOBS = HERE / "jobs"
JOBS.mkdir(exist_ok=True)

sys.path.insert(0, str(HERE))
import convert  # noqa: E402

PYTHON = sys.executable
NODE = shutil.which("node") or "node"

# Validatieseeds per kwaliteitsniveau. Meer seeds = smaller
# betrouwbaarheidsinterval, maar elke seed is een volledige feestsimulatie.
SEEDS = {"snel": 3, "normaal": 7, "grondig": 15}

# Aantal browser-pages dat validate_headless.js gelijktijdig gebruikt. Elke
# page is een aparte engine-instantie (geen gedeelde state), dus dit schaalt
# vrijwel lineair met de kernen die er zijn -- maar de optimizer (fase 1)
# draait op datzelfde moment niet meer, dus de volle machine mag hieraan.
VALIDATE_PARALLEL = max(1, (os.cpu_count() or 4) - 1)

app = FastAPI(title="Zaalplanner")


# ── Jobs ──────────────────────────────────────────────────────────────────

class JobRequest(BaseModel):
    project: dict
    quality: str = "normaal"
    top: int = 3


_jobs: dict[str, dict] = {}
_procs: dict[str, subprocess.Popen] = {}
_lock = threading.Lock()


def _status_path(job_id):
    return JOBS / job_id / "status.json"


def _save(job):
    with _lock:
        _status_path(job["id"]).write_text(json.dumps(job, indent=1))


def _update(job, **kw):
    job.update(kw)
    job["updatedAt"] = time.time()
    _save(job)


def _log(job, line):
    job["log"].append(line.rstrip()[:400])
    if len(job["log"]) > 400:
        del job["log"][:-400]


def _run_proc(job, cmd, cwd, on_line):
    p = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    _procs[job["id"]] = p
    for line in p.stdout:
        if job.get("cancelled"):
            p.kill()
            break
        on_line(line)
    p.wait()
    _procs.pop(job["id"], None)
    return p.returncode


def _worker(job):
    job_dir = JOBS / job["id"]
    try:
        # ── 1. Optimizer ──
        _update(job, stage="zoeken", pct=2, msg="Optimizer starten")

        def on_opt_line(line):
            if line.startswith("@@ "):
                try:
                    d = json.loads(line[3:])
                except json.JSONDecodeError:
                    return
                if d.get("error"):
                    job["error"] = d["msg"]
                _update(job, stage=d["stage"], pct=d["pct"], msg=d["msg"],
                        best_px=d.get("best_px", job.get("best_px")))
            else:
                _log(job, line)
                _save(job)

        rc = _run_proc(job, [PYTHON, str(HERE / "run_optimizer.py"), str(job_dir)], ROOT / "ml",
                       on_opt_line)
        if job.get("cancelled"):
            _update(job, status="geannuleerd", msg="Geannuleerd")
            return
        if rc != 0:
            _update(job, status="fout", pct=100,
                    msg=job.get("error") or f"Optimizer stopte met code {rc}. Zie het logboek.")
            return

        # ── 2. Validatie met de echte simulator ──
        results = json.loads((job_dir / "optimizer-results.json").read_text())
        seeds = SEEDS.get(job["quality"], 7)
        n = len(results)
        _update(job, stage="valideren", pct=91,
                msg=f"Simulatie: {n} indelingen x {seeds} seeds")
        done = {"n": 0}

        def on_val_line(line):
            _log(job, line)
            if "Layout #" in line and "dist=" in line:
                done["n"] += 1
                _update(job, pct=91 + 8 * done["n"] / max(1, n),
                        msg=f"Simulatie: {done['n']} van {n} indelingen gevalideerd")
            else:
                _save(job)

        rc = _run_proc(job, [NODE, str(ROOT / "ml" / "validate_headless.js"),
                             "--input", str(job_dir / "optimizer-results.json"),
                             "--out", str(job_dir / "validation-results.json"),
                             "--top", str(n), "--seeds", str(seeds),
                             "--parallel", str(VALIDATE_PARALLEL)],
                       ROOT / "ml", on_val_line)
        if job.get("cancelled"):
            _update(job, status="geannuleerd", msg="Geannuleerd")
            return
        if rc != 0 or not (job_dir / "validation-results.json").exists():
            _update(job, status="fout", pct=100,
                    msg="De simulatie-validatie mislukte. Zie het logboek.")
            return

        # ── 3. Samenvatten voor de app ──
        val = json.loads((job_dir / "validation-results.json").read_text())
        inp = json.loads((job_dir / "input.json").read_text())
        origin = inp["origin"]
        pxm = convert.PX_PER_M
        by_rank = {r["rank"]: r for r in results}
        cands, current = [], None
        for r in val:
            dists = r.get("dist_per_seed") or []
            mean = sum(dists) / len(dists) if dists else r.get("actual_dist", 0)
            sd = (sum((d - mean) ** 2 for d in dists) / (len(dists) - 1)) ** 0.5 if len(dists) > 1 else 0.0
            sem = sd / len(dists) ** 0.5 if dists else 0.0
            src = by_rank.get(r["rank"], {})
            item = {
                "rank": r["rank"],
                "is_current": bool(src.get("is_current")),
                "predicted_m": (src.get("predicted_waiterDist") or 0) / pxm if src.get("predicted_waiterDist") else None,
                "actual_m": mean / pxm,
                "sem_m": sem / pxm,
                "ci95_m": [(mean - 1.96 * sem) / pxm, (mean + 1.96 * sem) / pxm],
                "dist_per_seed_m": [d / pxm for d in dists],
                "avg_wait_s": (sum(r.get("wait_per_seed") or [0]) / max(1, len(r.get("wait_per_seed") or [0]))),
                "served": (sum(r.get("served_per_seed") or [0]) / max(1, len(r.get("served_per_seed") or [0]))),
                "impatient": (sum(r.get("impatient_per_seed") or [0]) / max(1, len(r.get("impatient_per_seed") or [0]))),
                "layout_valid": bool(r.get("layout_valid", True)),
                "unreachable_tables": r.get("unreachable_tables", 0),
                "seeds": r.get("seeds") or [],
                "tables": convert.tables_to_m(r["tables"], origin),
                "tables_px": r["tables"],
            }
            if item["is_current"]:
                item["placement_ok"] = bool(src.get("placement_ok_precheck", True))
                current = item
            else:
                cands.append(item)
        # run_optimizer.py exporteert en valideert een BREDERE kopgroep dan
        # de gebruiker te zien krijgt (zie QUALITY.validate_top aldaar) --
        # juist omdat het modeloordeel waarop die kopgroep gekozen is de
        # kandidaten binnenin niet betrouwbaar ordent. Sorteer dus eerst op
        # de echte, gesimuleerde afstand en kap dan pas af tot wat gevraagd
        # is; zo ziet de gebruiker de beste `top` van de VALIDATIE, niet de
        # eerste `top` van het model.
        cands.sort(key=lambda c: c["actual_m"])
        cands = cands[:inp.get("top", len(cands))]
        for i, c in enumerate(cands):
            c["rank"] = i + 1
        summary = {
            "candidates": cands,
            "current": current,
            "room_px": inp["room"],
            "config": inp["config"],
            "origin": origin,
            "pxPerM": pxm,
            "seats": inp["seats"],
            "quality": job["quality"],
            "seeds": seeds,
            "warnings": inp.get("warnings", []),
            "finishedAt": time.time(),
        }
        (job_dir / "result.json").write_text(json.dumps(summary, indent=1))
        _update(job, status="klaar", stage="klaar", pct=100, msg="Klaar", result=summary)
    except Exception as e:  # noqa: BLE001
        _log(job, f"Interne fout: {e!r}")
        _update(job, status="fout", pct=100, msg=f"Interne fout: {e}")


@app.post("/api/jobs")
def create_job(req: JobRequest):
    try:
        converted = convert.project_to_job(req.project)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if any(j.get("status") == "bezig" for j in _jobs.values()):
        raise HTTPException(status_code=409,
                            detail="Er loopt al een berekening. Wacht tot die klaar is of stop hem eerst.")
    quality = req.quality if req.quality in SEEDS else "normaal"
    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    job_dir = JOBS / job_id
    job_dir.mkdir()
    converted["quality"] = quality
    converted["top"] = max(1, min(8, req.top))
    (job_dir / "input.json").write_text(json.dumps(converted, indent=1))
    job = {"id": job_id, "status": "bezig", "stage": "wachten", "pct": 0, "msg": "In de wachtrij",
           "quality": quality, "log": [], "warnings": converted["warnings"],
           "createdAt": time.time(), "updatedAt": time.time(),
           "projectName": req.project.get("name", "")}
    _jobs[job_id] = job
    _save(job)
    threading.Thread(target=_worker, args=(job,), daemon=True).start()
    return {"id": job_id, "warnings": converted["warnings"], "seats": converted["seats"],
            "room_px": converted["room"]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        p = _status_path(job_id)
        if not p.exists():
            raise HTTPException(status_code=404, detail="Job onbekend")
        job = json.loads(p.read_text())
        if job.get("status") == "bezig":
            job["status"] = "fout"
            job["msg"] = "De server is herstart terwijl deze job liep."
    return job


@app.get("/api/jobs")
def list_jobs():
    out = []
    for p in sorted(JOBS.glob("*/status.json"), reverse=True)[:30]:
        try:
            j = json.loads(p.read_text())
            out.append({k: j.get(k) for k in ("id", "status", "stage", "pct", "msg", "quality",
                                               "createdAt", "projectName")})
        except json.JSONDecodeError:
            continue
    return out


@app.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job onbekend")
    job["cancelled"] = True
    p = _procs.get(job_id)
    if p:
        p.kill()
    _update(job, status="geannuleerd", msg="Geannuleerd")
    return {"ok": True}


@app.post("/api/preview")
def preview(req: JobRequest):
    """Alleen de conversie, voor de app om de simulatiezaal te tonen zonder te rekenen."""
    try:
        converted = convert.project_to_job(req.project)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return converted


@app.get("/api/health")
def health():
    model = (ROOT / "surrogate_model.pkl").exists()
    node = shutil.which("node") is not None
    pw = (ROOT / "node_modules" / "playwright").exists()
    return {"ok": model and node and pw, "model": model, "node": node, "playwright": pw,
            "python": PYTHON}


# ── Statisch ──────────────────────────────────────────────────────────────

@app.get("/sim/simulatie.html")
def sim_page():
    return FileResponse(ROOT / "simulatie.html")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=str(WEB)), name="web")
