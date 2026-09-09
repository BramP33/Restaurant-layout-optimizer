// Zaalplanner — hoofdmodule. Bindt panelen, editor, opslag, jobs en export.

import * as G from "./geometry.js";
import * as M from "./model.js";
import { api, projectForServer } from "./api.js";
import { Editor } from "./editor.js";
import { drawPlan, COLORS } from "./render.js";
import { renderResults, pickCandidate, fmtKm } from "./results.js";
import { Playback } from "./playback.js";

const $ = (id) => document.getElementById(id);
const nl = (v, d = 1) => Number(v).toLocaleString("nl-NL", { minimumFractionDigits: d, maximumFractionDigits: d });

// ── State ──────────────────────────────────────────────────────────────────
let projects = M.loadAll();
if (!projects.length) projects = [M.newProject("Mijn zaal")];
let P = projects[0];
let saveTimer = null;
let pollTimer = null;

const editor = new Editor($("plan"), () => P, {
  onChange: () => { touch(); syncRoomInfo(); syncChecklist(); syncProps(); },
  onSelect: () => syncProps(),
  onStatus: (t) => { $("status").textContent = t; },
  onCursor: (w) => { $("cursorPos").textContent = `x ${nl(w.x, 2)} m   y ${nl(w.y, 2)} m`; $("zoomLabel").textContent = `${Math.round(editor.view.scale)} px/m`; },
  onRectAdded: (i) => { editor.setTool("select"); editor.select({ kind: "rect", index: i }); },
  onTool: (t) => { document.querySelectorAll(".toolbar .tool[data-tool]").forEach(x => x.classList.toggle("active", x.dataset.tool === t)); },
});
const playback = new Playback();

function toast(msg, ms = 3200) {
  const t = $("toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(t._t); t._t = setTimeout(() => { t.hidden = true; }, ms);
}

function touch() {
  P.updatedAt = Date.now();
  $("saveState").textContent = "Wijzigingen…";
  clearTimeout(saveTimer);
  saveTimer = setTimeout(save, 500);
}
function save() {
  const ok = M.saveAll(projects);
  $("saveState").textContent = ok ? "Opgeslagen" : "Niet opgeslagen: te groot voor de browseropslag";
  if (!ok) toast("Opslaan mislukt. Waarschijnlijk is de foto te groot; verwijder hem of gebruik een kleinere.");
}

// ── Projecten ──────────────────────────────────────────────────────────────
function syncProjectSelect() {
  const s = $("projectSelect");
  s.innerHTML = "";
  for (const p of projects) {
    const o = document.createElement("option");
    o.value = p.id; o.textContent = p.name || "Zonder naam";
    if (p === P) o.selected = true;
    s.appendChild(o);
  }
  $("projectName").value = P.name;
}
function switchProject(p) {
  P = p;
  editor.select(null);
  editor.overlay.candidateTables = null;
  syncAll();
  editor.fit();
  if (P.job && P.job.status === "bezig") startPolling();
}
$("projectSelect").addEventListener("change", e => switchProject(projects.find(p => p.id === e.target.value)));
$("projectName").addEventListener("input", e => { P.name = e.target.value; touch(); syncProjectSelect(); });
$("btnNewProject").addEventListener("click", () => { const p = M.newProject("Nieuwe zaal"); projects.unshift(p); switchProject(p); touch(); });
$("btnDuplicate").addEventListener("click", () => {
  const c = M.upgrade(JSON.parse(JSON.stringify(P))); c.id = Math.random().toString(36).slice(2, 10); c.name = P.name + " (kopie)"; c.job = null;
  projects.unshift(c); switchProject(c); touch();
});
$("btnDeleteProject").addEventListener("click", () => {
  if (!confirm(`"${P.name}" verwijderen? Dit kan niet ongedaan worden gemaakt.`)) return;
  projects = projects.filter(p => p !== P);
  if (!projects.length) projects = [M.newProject("Mijn zaal")];
  switchProject(projects[0]); touch();
});

// ── Zaal ───────────────────────────────────────────────────────────────────
$("photoInput").addEventListener("change", async e => {
  const f = e.target.files[0]; if (!f) return;
  try {
    const { dataUrl, aspect } = await M.loadPhotoFile(f);
    const w = 20;
    P.photo = { dataUrl, x: 0, y: 0, w, h: w * aspect, opacity: 0.6 };
    touch(); syncPhoto(); editor.fit();
    if (!P.rects.length) { editor.setTool("rect"); toast("Teken nu de vloer over de foto. Kies daarna de rechthoek om de echte maten in te vullen."); }
  } catch (err) { toast(err.message); }
  e.target.value = "";
});
$("btnRemovePhoto").addEventListener("click", () => { P.photo = null; touch(); syncPhoto(); editor.draw(); });
$("photoWidth").addEventListener("input", e => {
  if (!P.photo) return;
  const w = Math.max(1, parseFloat(e.target.value) || 1);
  P.photo.h = P.photo.h / P.photo.w * w; P.photo.w = w; touch(); editor.draw();
});
$("photoOpacity").addEventListener("input", e => { if (P.photo) { P.photo.opacity = parseFloat(e.target.value); touch(); editor.draw(); } });

function syncPhoto() {
  const has = !!P.photo;
  $("btnRemovePhoto").hidden = !has; $("photoControls").hidden = !has;
  if (has) { $("photoWidth").value = nl(P.photo.w, 1).replace(",", "."); $("photoOpacity").value = P.photo.opacity; }
}

function syncRoomInfo() {
  const bb = G.bbox(P.rects);
  const dl = $("roomInfo");
  if (!bb) { dl.innerHTML = ""; return; }
  const area = P.rects.length === 1 ? bb.w * bb.h : bb.w * bb.h - G.complementBlocks(P.rects).reduce((s, b) => s + b.w * b.h, 0);
  const seats = G.totalSeats(P.tables, P.tableTypes);
  dl.innerHTML = `
    <dt>Buitenmaten</dt><dd>${nl(bb.w, 1)} × ${nl(bb.h, 1)} m</dd>
    <dt>Vloer</dt><dd>${nl(area, 0)} m²</dd>
    <dt>Per zitplaats</dt><dd>${seats ? nl(area / seats, 1) + " m²" : "–"}</dd>
    <dt>Bar</dt><dd>${P.bar ? nl(Math.max(P.bar.w, P.bar.h), 1) + " m, wand " + P.bar.wall : "nog niet geplaatst"}</dd>
    <dt>Buffet</dt><dd>${P.buffet ? nl(Math.max(P.buffet.w, P.buffet.h), 1) + " m, wand " + P.buffet.wall : "geen"}</dd>
    <dt>Ingang</dt><dd>${P.entrance ? "wand " + (P.entrance.wall || "") : "nog niet geplaatst"}</dd>`;
}

// ── Tafels ─────────────────────────────────────────────────────────────────
function activeTypes() {
  const out = ["small", "medium", "large"];
  for (const [k, v] of Object.entries(P.tableTypes)) if (v.enabled) out.push(k);
  return out;
}
function syncTables() {
  const box = $("tableCounts"); box.innerHTML = "";
  for (const type of activeTypes()) {
    const d = G.tableDef(type, P.tableTypes);
    const row = document.createElement("div"); row.className = "table-row";
    row.innerHTML = `<div><span class="swatch ${d.fixed ? "" : "custom"}"></span>${d.name || "Eigen tafel"}</div>
      <div class="meta">${nl(d.w, 2)} × ${nl(d.h, 2)} m · ${d.seats} st.</div>
      <input type="number" min="0" max="40" step="1" value="${P.tables[type] || 0}" data-type="${type}" />`;
    row.querySelector("input").addEventListener("input", e => { P.tables[type] = Math.max(0, parseInt(e.target.value, 10) || 0); touch(); syncSeats(); syncRoomInfo(); syncChecklist(); });
    box.appendChild(row);
  }
  syncSeats();
  // Eigen types
  const list = $("customTypes"); list.innerHTML = "";
  let n = 0;
  for (const [k, v] of Object.entries(P.tableTypes)) {
    if (!v.enabled) continue;
    n++;
    const row = document.createElement("div"); row.className = "custom-row";
    row.innerHTML = `<input type="text" placeholder="Naam" value="${v.name || ""}" data-f="name" />
      <input type="number" step="0.05" min="0.3" max="6" value="${v.w}" data-f="w" title="Breedte (m)" />
      <input type="number" step="0.05" min="0.3" max="6" value="${v.h}" data-f="h" title="Diepte (m)" />
      <input type="number" step="1" min="1" max="20" value="${v.seats}" data-f="seats" title="Stoelen" />
      <button class="x" title="Slot leegmaken">×</button>
      <div class="sub"><span>breedte</span><span>diepte</span><span>stoelen</span></div>`;
    row.querySelectorAll("input").forEach(inp => inp.addEventListener("input", e => {
      const f = e.target.dataset.f; v[f] = f === "name" ? e.target.value : parseFloat(e.target.value) || v[f];
      touch(); if (f !== "name") { syncSeats(); editor.draw(); } else syncTables();
    }));
    row.querySelector(".x").addEventListener("click", () => {
      P.tableTypes[k] = { name: "", w: 1.2, h: 0.8, seats: 4, enabled: false }; delete P.tables[k];
      P.currentLayout = P.currentLayout.filter(t => t.type !== k); touch(); syncTables(); editor.draw();
    });
    list.appendChild(row);
  }
  $("customCount").textContent = n ? `${n} / ${M.CUSTOM_SLOTS}` : "";
  $("btnAddCustom").disabled = n >= M.CUSTOM_SLOTS;
}
function syncSeats() { $("seatTotal").textContent = G.totalSeats(P.tables, P.tableTypes); }
$("btnAddCustom").addEventListener("click", () => {
  const free = Object.keys(P.tableTypes).find(k => !P.tableTypes[k].enabled);
  if (!free) return;
  P.tableTypes[free] = { name: "Eigen tafel", w: 1.2, h: 0.8, seats: 4, enabled: true }; P.tables[free] = 0;
  touch(); syncTables();
  $("customDetails").open = true;
});

// ── Feest en gezelschap ────────────────────────────────────────────────────
function syncParty() {
  $("partyType").value = P.party.type; $("durationH").value = P.party.durationH; $("guests").value = P.party.guests;
  $("hasBuffet").checked = P.party.hasBuffet; $("waiters").value = P.party.waiters;
  $("buffetHint").textContent = P.party.hasBuffet && !P.buffet ? "(plaats nog een buffetlijn)" : "";
  for (const k of ["avgAge", "ageSpread", "thirst", "appetite", "waiterSkill"]) {
    $(k).value = P.population[k]; $(k + "Val").textContent = k === "avgAge" || k === "ageSpread" ? P.population[k] : nl(P.population[k], 2);
  }
  $("quality").value = P.generation.quality; $("topN").value = P.generation.top;
  $("qualityHint").textContent = M.QUALITIES[P.generation.quality].hint;
}
$("partyType").addEventListener("change", e => {
  const preset = M.PARTY_PRESETS[e.target.value];
  P.party.type = e.target.value; P.party.durationH = preset.durationH; P.party.hasBuffet = preset.hasBuffet;
  touch(); syncParty(); syncChecklist(); editor.draw();
});
$("durationH").addEventListener("input", e => { P.party.durationH = Math.max(0.5, parseFloat(e.target.value) || 1); touch(); });
$("guests").addEventListener("input", e => { P.party.guests = Math.max(1, parseInt(e.target.value, 10) || 1); touch(); syncChecklist(); });
$("waiters").addEventListener("input", e => { P.party.waiters = Math.max(1, parseInt(e.target.value, 10) || 1); touch(); });
$("hasBuffet").addEventListener("change", e => { P.party.hasBuffet = e.target.checked; touch(); syncParty(); syncChecklist(); editor.draw(); });
for (const k of ["avgAge", "ageSpread", "thirst", "appetite", "waiterSkill"]) {
  $(k).addEventListener("input", e => { P.population[k] = parseFloat(e.target.value); touch(); syncParty(); });
}
$("quality").addEventListener("change", e => { P.generation.quality = e.target.value; touch(); syncParty(); });
$("topN").addEventListener("change", e => { P.generation.top = parseInt(e.target.value, 10); touch(); });

// ── Checklist en genereren ─────────────────────────────────────────────────
let serverOk = false;
function checklistItems() {
  const items = [];
  const seats = G.totalSeats(P.tables, P.tableTypes);
  const gen = ["small", "medium", "large"].reduce((s, k) => s + (P.tables[k] || 0), 0);
  items.push({ ok: P.rects.length > 0, text: P.rects.length ? "Vloer getekend" : "Teken de vloer van de zaal" });
  items.push({ ok: !!P.bar, text: P.bar ? "Bar geplaatst" : "Plaats een bar tegen een wand" });
  items.push({ ok: !!P.entrance, text: P.entrance ? "Ingang geplaatst" : "Plaats een ingang" });
  if (P.party.hasBuffet) items.push({ ok: !!P.buffet, text: P.buffet ? "Buffetlijn geplaatst" : "Plaats een buffetlijn, of zet het buffet uit" });
  items.push({ ok: gen > 0, text: gen ? `${gen} tafels om te plaatsen, ${seats} zitplaatsen` : "Geef minstens één tafel op" });
  if (seats && P.party.guests > seats * 1.6) items.push({ warn: true, text: "Veel meer gasten dan zitplaatsen; bij een receptie is dat normaal" });
  if (seats && P.party.guests < seats * 0.6) items.push({ warn: true, text: "Veel minder gasten dan zitplaatsen" });
  items.push({ ok: serverOk, text: serverOk ? "Rekenserver bereikbaar" : "Rekenserver niet bereikbaar. Start app/start.sh" });
  return items;
}
function syncChecklist() {
  $("buffetHint").textContent = P.party.hasBuffet && !P.buffet ? "(plaats nog een buffetlijn)" : "";
  const ul = $("checklist"); ul.innerHTML = "";
  let ready = true;
  for (const it of checklistItems()) {
    const li = document.createElement("li");
    li.className = it.warn ? "warn" : it.ok ? "ok" : "bad";
    li.textContent = it.text; ul.appendChild(li);
    if (!it.warn && !it.ok) ready = false;
  }
  $("btnGenerate").disabled = !ready || (P.job && P.job.status === "bezig");
}

$("btnGenerate").addEventListener("click", async () => {
  try {
    $("btnGenerate").disabled = true;
    const res = await api.createJob(projectForServer(P), P.generation.quality, P.generation.top);
    P.job = { id: res.id, status: "bezig", pct: 0, msg: "Gestart", log: [], startedAt: Date.now() };
    touch(); syncJob(); startPolling();
    if (res.warnings?.length) toast(res.warnings.join(" "), 6000);
  } catch (e) {
    toast(e.message, 6000); syncChecklist();
  }
});
$("btnCancel").addEventListener("click", async () => { if (P.job) { await api.cancel(P.job.id).catch(() => {}); } });

function startPolling() {
  clearInterval(pollTimer);
  const job = P.job; if (!job) return;
  const proj = P;
  pollTimer = setInterval(async () => {
    try {
      const j = await api.job(job.id);
      proj.job = { ...proj.job, ...j };
      if (proj === P) syncJob();
      if (j.status !== "bezig") {
        clearInterval(pollTimer);
        if (j.status === "klaar" && j.result) {
          proj.result = j.result; proj.view.candidate = 1;
          if (proj === P) { showCandidate(1); renderResults(P, showCandidate); toast("Klaar: " + j.result.candidates.length + " indelingen gevalideerd."); }
        } else if (j.status === "fout") {
          toast("Mislukt: " + j.msg, 8000);
        }
        touch(); if (proj === P) syncChecklist();
      }
    } catch (e) {
      // Server tijdelijk weg: blijven proberen.
    }
  }, 2000);
}
function syncJob() {
  const j = P.job;
  const busy = j && j.status === "bezig";
  $("jobProgress").hidden = !j;
  $("btnCancel").hidden = !busy;
  $("btnGenerate").textContent = busy ? "Bezig…" : "Genereer indelingen";
  if (!j) return;
  $("jobBar").style.width = (j.pct || 0) + "%";
  const mins = j.startedAt ? Math.round((Date.now() - j.startedAt) / 60000) : null;
  $("jobMsg").textContent = (j.msg || "") + (busy && mins ? ` · ${mins} min` : "") + (j.status && !busy ? ` · ${j.status}` : "");
  $("jobLog").textContent = (j.log || []).slice(-60).join("\n");
  syncChecklist();
}

// ── Resultaten ─────────────────────────────────────────────────────────────
function showCandidate(rank) {
  P.view.candidate = rank;
  const c = pickCandidate(P);
  editor.overlay.candidateTables = c && !c.is_current ? c.tables : null;
  editor.overlay.showCurrent = $("showCurrent").checked || (c && c.is_current);
  renderResults(P, showCandidate);
  editor.draw(); touch();
}
$("showCurrent").addEventListener("change", () => { P.view.showCurrent = $("showCurrent").checked; showCandidate(P.view.candidate); });
$("btnPlay").addEventListener("click", () => {
  const c = pickCandidate(P); if (!c) return;
  playback.open({ title: (c.is_current ? "Huidige indeling" : "Voorstel " + c.rank) + " · " + P.name,
    roomPx: P.result.room_px, config: P.result.config, tablesPx: c.tables_px });
});
$("btnExportJson").addEventListener("click", () => {
  const { photo, ...rest } = P;
  download(`${slug(P.name)}.json`, "application/json", JSON.stringify(rest, null, 2));
});
$("btnExportPng").addEventListener("click", () => exportPng());

function slug(s) { return (s || "zaal").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "zaal"; }
function download(name, type, data) {
  const blob = data instanceof Blob ? data : new Blob([data], { type });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

function exportPng() {
  const bb = G.bbox(P.rects); if (!bb) { toast("Teken eerst een zaal."); return; }
  const c = pickCandidate(P);
  const W = 2400, margin = 140, titleH = 300;
  const scale = Math.min((W - 2 * margin) / bb.w, 1400 / bb.h);
  const H = Math.round(bb.h * scale + 2 * margin + titleH);
  const cv = document.createElement("canvas"); cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d");
  // Papier
  ctx.fillStyle = COLORS.paper; ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = "rgba(136,149,139,0.07)";
  for (let i = 0; i < 9000; i++) { ctx.fillRect(Math.random() * W, Math.random() * H, 2, 2); }
  ctx.strokeStyle = COLORS.lineSoft; ctx.lineWidth = 2; ctx.strokeRect(40, 40, W - 80, H - 80);
  // Gevouwen hoek
  ctx.fillStyle = COLORS.paper; ctx.beginPath(); ctx.moveTo(W - 40, 40); ctx.lineTo(W - 40, 120); ctx.lineTo(W - 120, 40); ctx.closePath(); ctx.fill();
  ctx.fillStyle = "#E6EBE9"; ctx.beginPath(); ctx.moveTo(W - 40, 120); ctx.lineTo(W - 120, 40); ctx.lineTo(W - 120, 120); ctx.closePath(); ctx.fill();
  ctx.strokeStyle = COLORS.lineSoft; ctx.stroke();
  const view = { scale, ox: (W - bb.w * scale) / 2 - bb.x * scale, oy: margin - bb.y * scale };
  drawPlan(ctx, P, view, { photo: null, showGrid: false, candidateTables: c && !c.is_current ? c.tables : null,
    showCurrent: !c || c.is_current || $("showCurrent").checked, labels: true });
  // Titelblok
  const y0 = H - titleH - 40;
  ctx.strokeStyle = COLORS.line; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(90, y0); ctx.lineTo(W - 90, y0); ctx.stroke();
  ctx.fillStyle = COLORS.ink; ctx.textBaseline = "top";
  ctx.font = "700 44px Inter, system-ui, sans-serif"; ctx.fillText(P.name || "Zaal", 90, y0 + 30);
  ctx.font = "500 22px Inter, system-ui, sans-serif"; ctx.fillStyle = "#5E6B66";
  const date = new Date().toLocaleDateString("nl-NL", { day: "numeric", month: "long", year: "numeric" });
  ctx.fillText(`Zaalplanner · ${date}`, 90, y0 + 90);
  const seats = G.totalSeats(P.tables, P.tableTypes);
  const lines = [
    `Buitenmaten ${nl(bb.w, 1)} × ${nl(bb.h, 1)} m · ${seats} zitplaatsen · ${P.party.guests} gasten · ${M.PARTY_PRESETS[P.party.type].label} ${nl(P.party.durationH, 1)} uur`,
  ];
  if (c) {
    lines.push(c.is_current ? `Huidige indeling: looplengte bediening ${fmtKm(c.actual_m)} (± ${fmtKm(1.96 * c.sem_m)})`
      : `Voorstel ${c.rank}: looplengte bediening ${fmtKm(c.actual_m)} (± ${fmtKm(1.96 * c.sem_m)})` +
        (P.result.current && P.result.current.layout_valid ? ` · ${nl((c.actual_m - P.result.current.actual_m) / P.result.current.actual_m * 100, 1)}% t.o.v. huidig` : ""));
  }
  ctx.font = "400 22px Inter, system-ui, sans-serif";
  lines.forEach((l, i) => ctx.fillText(l, 90, y0 + 135 + i * 36));
  // Legenda
  const lx = W - 620, ly = y0 + 40;
  const leg = [[COLORS.candidate, "Tafel (voorstel)", COLORS.accent], ["rgba(0,0,0,0)", "Huidige indeling", COLORS.current], [COLORS.bar, "Bar", COLORS.barEdge], [COLORS.buffet, "Buffet", COLORS.buffetEdge], [COLORS.block, "Obstakel", COLORS.ink]];
  ctx.font = "400 20px Inter, system-ui, sans-serif";
  leg.forEach(([fill, label, edge], i) => {
    ctx.fillStyle = fill; ctx.fillRect(lx, ly + i * 40, 44, 26); ctx.strokeStyle = edge; ctx.lineWidth = 2; ctx.strokeRect(lx, ly + i * 40, 44, 26);
    ctx.fillStyle = COLORS.ink; ctx.fillText(label, lx + 60, ly + i * 40 + 2);
  });
  cv.toBlob(b => download(`${slug(P.name)}${c && !c.is_current ? "-voorstel-" + c.rank : ""}.png`, "image/png", b), "image/png");
}

// ── Eigenschappen ──────────────────────────────────────────────────────────
function syncProps() {
  const box = $("props"), s = editor.selected;
  if (!s) { box.innerHTML = `<p class="help">Selecteer iets op de plattegrond, of kies een gereedschap boven de tekening.</p>`; return; }
  const num = (label, val, step, min, cb, unit = "m") => {
    const d = document.createElement("div"); d.className = "field";
    d.innerHTML = `<label>${label}</label><div class="unit"><input type="number" step="${step}" min="${min}" value="${Math.round(val * 100) / 100}" /><span>${unit}</span></div>`;
    d.querySelector("input").addEventListener("change", e => { editor.pushUndo(); cb(parseFloat(e.target.value)); editor.changed(); });
    return d;
  };
  box.innerHTML = "";
  const title = document.createElement("div"); title.className = "props-title"; box.appendChild(title);
  if (s.kind === "rect") {
    const r = P.rects[s.index]; title.textContent = `Vloer ${s.index + 1}`;
    box.appendChild(num("Breedte", r.w, 0.1, 0.5, v => { r.w = v; }));
    box.appendChild(num("Diepte", r.h, 0.1, 0.5, v => { r.h = v; }));
    box.appendChild(num("Positie x", r.x, 0.1, -100, v => { r.x = v; }));
    box.appendChild(num("Positie y", r.y, 0.1, -100, v => { r.y = v; }));
    if (P.photo) {
      const cal = document.createElement("div"); cal.className = "stack";
      cal.innerHTML = `<p class="help small">Foto kalibreren: vul de echte breedte van deze rechthoek in zoals hij op de plattegrond staat. De foto en alles wat je tekende schalen mee.</p>
        <div class="field"><label>Echte breedte</label><div class="unit"><input id="calW" type="number" step="0.1" min="1" value="${Math.round(r.w * 10) / 10}" /><span>m</span></div></div>
        <div class="field"><label>Echte diepte</label><div class="unit"><input id="calH" type="number" step="0.1" min="1" value="${Math.round(r.h * 10) / 10}" /><span>m</span></div></div>
        <button class="btn primary" id="btnCal">Schaal foto op deze maten</button>`;
      box.appendChild(cal);
      cal.querySelector("#btnCal").addEventListener("click", () => {
        const W = parseFloat(cal.querySelector("#calW").value), Hh = parseFloat(cal.querySelector("#calH").value);
        if (!(W > 0)) return;
        calibrate(s.index, W, Hh > 0 ? Hh : null);
      });
    }
  } else if (s.kind === "block") {
    const b = P.blocks[s.index]; title.textContent = "Obstakel";
    box.appendChild(num("Breedte", b.w, 0.05, 0.1, v => { b.w = v; }));
    box.appendChild(num("Diepte", b.h, 0.05, 0.1, v => { b.h = v; }));
    box.appendChild(num("Positie x", b.x, 0.05, -100, v => { b.x = v; }));
    box.appendChild(num("Positie y", b.y, 0.05, -100, v => { b.y = v; }));
  } else if (s.kind === "bar" || s.kind === "buffet") {
    const it = P[s.kind]; title.textContent = s.kind === "bar" ? "Bar" : "Buffetlijn";
    const len = Math.max(it.w, it.h);
    box.appendChild(num("Lengte", len, 0.1, 1, v => editor.resizeWallItem(s.kind, Math.max(1, v))));
    if (s.kind === "buffet") box.appendChild(num("Opschepplekken", it.slots || 3, 1, 1, v => { it.slots = Math.max(1, Math.round(v)); }, ""));
    const p = document.createElement("p"); p.className = "help small";
    p.textContent = s.kind === "bar"
      ? `Tegen de wand ${it.wall}. De oranje stip is het punt waar de bediening dienbladen haalt. Sleep om de bar langs de wand te schuiven.`
      : `Tegen de wand ${it.wall}. De stippen zijn de plekken waar gasten opscheppen. Sleep om de lijn te verschuiven.`;
    box.appendChild(p);
  } else if (s.kind === "entrance") {
    title.textContent = "Ingang";
    const p = document.createElement("p"); p.className = "help small";
    p.textContent = `Op de wand ${P.entrance.wall || ""}. Gasten komen hier binnen en gaan hier weg. Sleep om te verplaatsen.`;
    box.appendChild(p);
  } else if (s.kind === "table") {
    const t = P.currentLayout[s.index]; title.textContent = "Tafel (huidige indeling)";
    const sel = document.createElement("div"); sel.className = "field";
    sel.innerHTML = `<label>Type</label><select class="select"></select>`;
    const se = sel.querySelector("select");
    for (const type of activeTypes()) { const o = document.createElement("option"); o.value = type; o.textContent = G.tableDef(type, P.tableTypes).name; o.selected = type === t.type; se.appendChild(o); }
    se.addEventListener("change", e => { editor.pushUndo(); t.type = e.target.value; editor.changed(); });
    box.appendChild(sel);
    box.appendChild(num("Draaiing", t.rotation || 0, 90, 0, v => { t.rotation = ((Math.round(v / 90) * 90) % 180 + 180) % 180; }, "°"));
    box.appendChild(num("Positie x", t.x, 0.05, -100, v => { t.x = v; }));
    box.appendChild(num("Positie y", t.y, 0.05, -100, v => { t.y = v; }));
    const p = document.createElement("p"); p.className = "help small";
    p.textContent = `Je huidige indeling wordt gesimuleerd als vergelijking. ${P.currentLayout.length} tafels geplaatst.`;
    box.appendChild(p);
  }
  if (s.kind !== "photo") {
    const del = document.createElement("button"); del.className = "btn ghost danger"; del.textContent = "Verwijder";
    del.addEventListener("click", () => editor.deleteSelected()); box.appendChild(del);
  }
}

// Schaal foto en geometrie zo dat rechthoek `i` de opgegeven maten krijgt.
function calibrate(i, W, Hh) {
  const r = P.rects[i];
  const f = W / r.w;
  const ox = r.x, oy = r.y;
  editor.pushUndo();
  const sc = (o) => { o.x = ox + (o.x - ox) * f; o.y = oy + (o.y - oy) * f; if ("w" in o) { o.w *= f; o.h *= f; } };
  P.rects.forEach(sc); P.blocks.forEach(sc); P.currentLayout.forEach(t => { t.x = ox + (t.x - ox) * f; t.y = oy + (t.y - oy) * f; });
  if (P.photo) sc(P.photo);
  if (P.bar) sc(P.bar); if (P.buffet) sc(P.buffet); if (P.entrance) { P.entrance.x = ox + (P.entrance.x - ox) * f; P.entrance.y = oy + (P.entrance.y - oy) * f; }
  if (Hh) r.h = Hh;
  // Wandmeubels hebben een vaste diepte: opnieuw tegen hun wand zetten.
  const segs = G.wallSegments(P.rects);
  for (const kind of ["bar", "buffet"]) {
    const it = P[kind]; if (!it) continue;
    const hit = G.segmentFor(it, segs) || G.nearestWall({ x: it.x + it.w / 2, y: it.y + it.h / 2 }, segs, 5);
    if (hit) Object.assign(it, G.placeOnWall(hit.seg, { x: it.x + it.w / 2, y: it.y + it.h / 2 }, Math.max(it.w, it.h), kind === "bar" ? G.BAR_DEPTH_M : G.BUFFET_DEPTH_M));
  }
  if (P.entrance) {
    const hit = G.nearestWall(P.entrance, segs, 5);
    if (hit) Object.assign(P.entrance, G.entranceFromWall(hit.seg, hit.point));
  }
  // Netjes op de decimeter
  const rd = (o) => { o.x = G.round2(o.x); o.y = G.round2(o.y); if ("w" in o) { o.w = G.round2(o.w); o.h = G.round2(o.h); } };
  P.rects.forEach(rd); P.blocks.forEach(rd);
  editor.changed(); syncPhoto(); editor.fit();
  toast(`Foto geschaald: deze vloer meet nu ${nl(W, 1)} × ${nl(P.rects[i].h, 1)} m.`);
}

// ── Gereedschap ────────────────────────────────────────────────────────────
document.querySelectorAll(".toolbar .tool[data-tool]").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll(".toolbar .tool[data-tool]").forEach(x => x.classList.toggle("active", x === b));
  editor.setTool(b.dataset.tool);
  if (b.dataset.tool === "table") showTableTypePicker();
  else if (b.dataset.tool === "bar" || b.dataset.tool === "buffet") showWallItemPicker(b.dataset.tool);
}));
function showTableTypePicker() {
  const box = $("props"); box.innerHTML = `<div class="props-title">Tafel plaatsen</div>
    <p class="help small">Tafels van je huidige indeling, ter vergelijking. Klik op de vloer om te plaatsen. R draait.</p>
    <div class="field"><label>Type</label><select class="select" id="pickType"></select></div>`;
  const se = box.querySelector("#pickType");
  for (const type of activeTypes()) { const o = document.createElement("option"); o.value = type; o.textContent = G.tableDef(type, P.tableTypes).name; o.selected = type === editor.params.tableType; se.appendChild(o); }
  se.addEventListener("change", e => { editor.params.tableType = e.target.value; });
}
function showWallItemPicker(kind) {
  const box = $("props");
  const key = kind === "bar" ? "barLength" : "buffetLength";
  box.innerHTML = `<div class="props-title">${kind === "bar" ? "Bar" : "Buffetlijn"} plaatsen</div>
    <p class="help small">Beweeg naar een wand; de ${kind === "bar" ? "bar" : "lijn"} klikt eraan vast. Klik om te plaatsen.</p>
    <div class="field"><label>Lengte</label><div class="unit"><input id="pickLen" type="number" step="0.1" min="1" value="${editor.params[key]}" /><span>m</span></div></div>`;
  box.querySelector("#pickLen").addEventListener("input", e => { editor.params[key] = Math.max(1, parseFloat(e.target.value) || 1); });
}
$("btnRotate").addEventListener("click", () => editor.rotate());
$("btnDelete").addEventListener("click", () => editor.deleteSelected());
$("btnUndo").addEventListener("click", () => editor.undo());
$("btnFitView").addEventListener("click", () => editor.fit());

// ── Tactiel: schaduwen die meebewegen bij scrollen, gescheurde plattegrond ──
for (const panel of document.querySelectorAll(".panel")) {
  panel.addEventListener("scroll", () => {
    const t = panel.scrollTop;
    panel.style.setProperty("--sx", (Math.sin(t / 160) * 4).toFixed(2));
    panel.style.setProperty("--sy", (Math.cos(t / 210) * 2).toFixed(2));
  }, { passive: true });
}
// Gescheurde rand voor het grote vel: een polygoon met veel kleine hoekjes.
(function tearPlan() {
  const el = $("planSheet"); if (!el) return;
  const pts = [];
  const rnd = (a) => a * (Math.random() - 0.5);
  const N = 42;
  for (let i = 0; i <= N; i++) pts.push(`${(i / N * 100).toFixed(2)}% ${(0.2 + Math.abs(rnd(0.7))).toFixed(2)}%`);
  for (let i = 1; i <= N; i++) pts.push(`${(100 - Math.abs(rnd(0.6))).toFixed(2)}% ${(i / N * 100).toFixed(2)}%`);
  for (let i = N - 1; i >= 0; i--) pts.push(`${(i / N * 100).toFixed(2)}% ${(100 - 0.2 - Math.abs(rnd(0.8))).toFixed(2)}%`);
  for (let i = N - 1; i >= 1; i--) pts.push(`${(Math.abs(rnd(0.5))).toFixed(2)}% ${(i / N * 100).toFixed(2)}%`);
  el.style.clipPath = `polygon(${pts.join(", ")})`;
})();

// ── Gezondheid server ──────────────────────────────────────────────────────
async function checkHealth() {
  try { const h = await api.health(); serverOk = !!h.ok; } catch (e) { serverOk = false; }
  $("healthDot").className = "dot " + (serverOk ? "ok" : "bad");
  $("healthDot").title = serverOk ? "Rekenserver bereikbaar" : "Rekenserver niet bereikbaar";
  syncChecklist();
}

// ── Alles verversen ───────────────────────────────────────────────────────
function syncAll() {
  syncProjectSelect(); syncPhoto(); syncRoomInfo(); syncTables(); syncParty(); syncChecklist(); syncJob(); syncProps();
  $("showCurrent").checked = P.view.showCurrent !== false;
  renderResults(P, showCandidate);
  if (P.result) showCandidate(P.view.candidate ?? 1); else { editor.overlay.candidateTables = null; editor.draw(); }
}

syncAll();
editor.status();
requestAnimationFrame(() => editor.fit());
checkHealth(); setInterval(checkHealth, 15000);
if (P.job && P.job.status === "bezig") startPolling();
