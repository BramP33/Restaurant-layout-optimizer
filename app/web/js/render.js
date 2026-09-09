// Tekenlaag voor de plattegrond. Gebruikt door de editor (interactief) en
// door de PNG-export (statisch). `view` = { scale (schermpixels per meter),
// ox, oy (schermpositie van wereldpunt 0,0) }.

import { wallSegments, tableDef, tableChairs, bbox, complementBlocks } from "./geometry.js";

export const COLORS = {
  paper:   "#F7F9F9",
  ink:     "#3F4A46",
  line:    "#88958B",
  lineSoft:"#CBD3CF",
  floor:   "#FFFFFF",
  table:   "#E7BB8F",          // zand
  tableEdge: "#C9976A",
  bar:     "#F1A57E",          // perzik
  barEdge: "#D07E55",
  buffet:  "#F7C9AE",
  buffetEdge: "#D07E55",
  accent:  "#EE5E28",
  block:   "#DDE2E0",
  candidate: "#FBE3D7",        // licht accent, rand in accent
  current: "#88958B",
  grid:    "rgba(136, 149, 139, 0.28)",
  gridMinor: "rgba(136, 149, 139, 0.12)",
};

export function worldToScreen(view, p) { return { x: view.ox + p.x * view.scale, y: view.oy + p.y * view.scale }; }
export function screenToWorld(view, p) { return { x: (p.x - view.ox) / view.scale, y: (p.y - view.oy) / view.scale }; }

function rrect(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

function paperShadow(ctx, x, y, w, h) {
  ctx.save();
  ctx.fillStyle = "rgba(63,74,70,0.16)";
  ctx.fillRect(x + 2, y + 3, w, h);
  ctx.restore();
}

function hatch(ctx, x, y, w, h, s, color) {
  ctx.save();
  ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
  ctx.strokeStyle = color; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let d = -h; d < w + h; d += s) { ctx.moveTo(x + d, y); ctx.lineTo(x + d + h, y + h); }
  ctx.stroke();
  ctx.restore();
}

export function fmtM(v) { return (Math.round(v * 100) / 100).toFixed(2).replace(".", ",") + " m"; }

// Tekent het hele plan. opts:
//   photo (Image) | selected {kind,index} | hover (preview-item) | showGrid
//   candidateTables [{type,x,y,rotation,w,h,seats}] | showCurrent | labels
export function drawPlan(ctx, p, view, opts = {}) {
  const S = view.scale;
  const W = (v) => v * S;
  const X = (v) => view.ox + v * S;
  const Y = (v) => view.oy + v * S;
  const rects = p.rects || [];
  const segs = wallSegments(rects);
  const bb = bbox(rects);

  // Foto
  if (p.photo && opts.photo && opts.photo.complete) {
    ctx.save();
    ctx.globalAlpha = p.photo.opacity ?? 0.55;
    ctx.drawImage(opts.photo, X(p.photo.x), Y(p.photo.y), W(p.photo.w), W(p.photo.h));
    ctx.restore();
  }

  // Ruitjespapier: 1 m hoofdlijnen, 0,5 m hulplijnen, over het hele vel
  if (opts.showGrid !== false && S >= 8) {
    const cw = ctx.canvas.width, ch = ctx.canvas.height;
    ctx.save();
    ctx.lineWidth = 1;
    const x0 = Math.floor(-view.ox / S) - 1, y0 = Math.floor(-view.oy / S) - 1;
    if (S >= 24) {
      ctx.strokeStyle = COLORS.gridMinor; ctx.beginPath();
      for (let gx = x0; X(gx) < cw; gx += 0.5) { ctx.moveTo(Math.round(X(gx)) + .5, 0); ctx.lineTo(Math.round(X(gx)) + .5, ch); }
      for (let gy = y0; Y(gy) < ch; gy += 0.5) { ctx.moveTo(0, Math.round(Y(gy)) + .5); ctx.lineTo(cw, Math.round(Y(gy)) + .5); }
      ctx.stroke();
    }
    const step = S < 14 ? 2 : 1;
    ctx.strokeStyle = COLORS.grid; ctx.beginPath();
    for (let gx = Math.floor(x0 / step) * step; X(gx) < cw; gx += step) { ctx.moveTo(Math.round(X(gx)) + .5, 0); ctx.lineTo(Math.round(X(gx)) + .5, ch); }
    for (let gy = Math.floor(y0 / step) * step; Y(gy) < ch; gy += step) { ctx.moveTo(0, Math.round(Y(gy)) + .5); ctx.lineTo(cw, Math.round(Y(gy)) + .5); }
    ctx.stroke();
    ctx.restore();
  }

  // Vloer: een opgeplakt vel, met zachte slagschaduw
  ctx.save();
  if (rects.length && !p.photo) {
    ctx.shadowColor = "rgba(63,74,70,0.22)"; ctx.shadowBlur = 10; ctx.shadowOffsetX = 2; ctx.shadowOffsetY = 4;
    ctx.fillStyle = COLORS.floor;
    for (const r of rects) ctx.fillRect(X(r.x), Y(r.y), W(r.w), W(r.h));
    ctx.shadowColor = "transparent";
  }
  ctx.fillStyle = p.photo ? "rgba(255,255,255,0.35)" : COLORS.floor;
  for (const r of rects) ctx.fillRect(X(r.x), Y(r.y), W(r.w), W(r.h));
  ctx.restore();

  // Weggesneden delen (visueel, buiten de vloer maar binnen de omhullende)
  if (bb) {
    for (const b of complementBlocks(rects)) hatch(ctx, X(b.x), Y(b.y), W(b.w), W(b.h), 9, "rgba(136,149,139,0.22)");
  }

  // Wanden
  ctx.save();
  ctx.strokeStyle = COLORS.ink; ctx.lineWidth = Math.max(2, S * 0.18); ctx.lineCap = "square";
  ctx.beginPath();
  for (const s of segs) { ctx.moveTo(X(s.x1), Y(s.y1)); ctx.lineTo(X(s.x2), Y(s.y2)); }
  ctx.stroke();
  ctx.restore();

  // Blokken (kolommen, podia)
  for (const [i, b] of (p.blocks || []).entries()) {
    paperShadow(ctx, X(b.x), Y(b.y), W(b.w), W(b.h));
    ctx.fillStyle = COLORS.block;
    ctx.fillRect(X(b.x), Y(b.y), W(b.w), W(b.h));
    hatch(ctx, X(b.x), Y(b.y), W(b.w), W(b.h), 7, "rgba(63,74,70,0.35)");
    ctx.strokeStyle = COLORS.ink; ctx.lineWidth = 1.5;
    ctx.strokeRect(X(b.x), Y(b.y), W(b.w), W(b.h));
    if (opts.selected?.kind === "block" && opts.selected.index === i) selection(ctx, X(b.x), Y(b.y), W(b.w), W(b.h));
  }

  // Bar
  if (p.bar) drawBar(ctx, p.bar, view, "Bar", opts.selected?.kind === "bar");
  // Buffet
  if (p.buffet) drawBuffet(ctx, p.buffet, view, opts.selected?.kind === "buffet", p.party?.hasBuffet === false);
  // Ingang
  if (p.entrance) drawEntrance(ctx, p.entrance, view, opts.selected?.kind === "entrance");

  // Eigen (huidige) indeling
  const showCurrent = opts.showCurrent !== false;
  if (showCurrent) {
    for (const [i, t] of (p.currentLayout || []).entries()) {
      const def = tableDef(t.type, p.tableTypes);
      const sel = opts.selected?.kind === "table" && opts.selected.index === i;
      drawTable(ctx, t, def, view, { fill: opts.candidateTables ? "rgba(255,255,255,0.0)" : COLORS.table,
        edge: opts.candidateTables ? COLORS.current : COLORS.tableEdge, dashed: !!opts.candidateTables,
        label: opts.labels !== false, selected: sel, custom: !def.fixed });
    }
  }

  // Voorgestelde indeling
  if (opts.candidateTables) {
    for (const t of opts.candidateTables) {
      const def = { w: t.w, h: t.h, seats: t.seats, fixed: true };
      drawTable(ctx, t, def, view, { fill: COLORS.candidate, edge: COLORS.accent, label: opts.labels !== false, strong: true });
    }
  }

  // Voorbeeld tijdens plaatsen
  if (opts.hover) {
    const h = opts.hover;
    ctx.save();
    ctx.globalAlpha = 0.6;
    if (h.kind === "bar") drawBar(ctx, h.item, view, "Bar", false);
    else if (h.kind === "buffet") drawBuffet(ctx, h.item, view, false, false);
    else if (h.kind === "entrance") drawEntrance(ctx, h.item, view, false);
    else if (h.kind === "table") drawTable(ctx, h.item, h.def, view, { fill: COLORS.table, edge: COLORS.tableEdge, custom: !h.def.fixed });
    else if (h.kind === "rect" || h.kind === "block") {
      ctx.fillStyle = h.kind === "rect" ? "rgba(231,187,143,0.25)" : "rgba(136,149,139,0.25)";
      ctx.fillRect(X(h.item.x), Y(h.item.y), W(h.item.w), W(h.item.h));
      ctx.setLineDash([6, 4]); ctx.strokeStyle = COLORS.accent; ctx.lineWidth = 1.5;
      ctx.strokeRect(X(h.item.x), Y(h.item.y), W(h.item.w), W(h.item.h));
      dimLabel(ctx, X(h.item.x) + W(h.item.w) / 2, Y(h.item.y) - 8, fmtM(h.item.w));
      dimLabel(ctx, X(h.item.x) - 8, Y(h.item.y) + W(h.item.h) / 2, fmtM(h.item.h), true);
    }
    ctx.restore();
  }

  // Geselecteerde vloerrechthoek
  if (opts.selected?.kind === "rect") {
    const r = rects[opts.selected.index];
    if (r) {
      selection(ctx, X(r.x), Y(r.y), W(r.w), W(r.h));
      dimLabel(ctx, X(r.x) + W(r.w) / 2, Y(r.y) - 10, fmtM(r.w));
      dimLabel(ctx, X(r.x) - 10, Y(r.y) + W(r.h) / 2, fmtM(r.h), true);
    }
  }

  // Buitenmaten
  if (bb && opts.labels !== false && S >= 8) {
    dimLine(ctx, X(bb.x), Y(bb.y + bb.h) + 18 + S * 0.15, X(bb.x + bb.w), Y(bb.y + bb.h) + 18 + S * 0.15, fmtM(bb.w), false);
    dimLine(ctx, X(bb.x + bb.w) + 18 + S * 0.15, Y(bb.y), X(bb.x + bb.w) + 18 + S * 0.15, Y(bb.y + bb.h), fmtM(bb.h), true);
  }
}

function selection(ctx, x, y, w, h) {
  ctx.save();
  ctx.setLineDash([5, 4]); ctx.strokeStyle = COLORS.accent; ctx.lineWidth = 1.5;
  ctx.strokeRect(x - 3, y - 3, w + 6, h + 6);
  ctx.restore();
}

function dimLabel(ctx, x, y, text, vertical = false) {
  ctx.save();
  ctx.font = "500 11px Inter, system-ui, sans-serif";
  ctx.fillStyle = COLORS.ink; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  const w = ctx.measureText(text).width + 10;
  ctx.translate(x, y);
  if (vertical) ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = "rgba(247,249,249,0.92)"; rrect(ctx, -w / 2, -9, w, 18, 2); ctx.fill();
  ctx.fillStyle = COLORS.ink; ctx.fillText(text, 0, 0.5);
  ctx.restore();
}

function dimLine(ctx, x1, y1, x2, y2, text, vertical) {
  ctx.save();
  ctx.strokeStyle = COLORS.line; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  const tick = 5;
  ctx.beginPath();
  if (vertical) { ctx.moveTo(x1 - tick, y1); ctx.lineTo(x1 + tick, y1); ctx.moveTo(x2 - tick, y2); ctx.lineTo(x2 + tick, y2); }
  else          { ctx.moveTo(x1, y1 - tick); ctx.lineTo(x1, y1 + tick); ctx.moveTo(x2, y2 - tick); ctx.lineTo(x2, y2 + tick); }
  ctx.stroke();
  ctx.restore();
  dimLabel(ctx, (x1 + x2) / 2, (y1 + y2) / 2, text, vertical);
}

function drawBar(ctx, b, view, label, selected) {
  const S = view.scale, x = view.ox + b.x * S, y = view.oy + b.y * S, w = b.w * S, h = b.h * S;
  ctx.save();
  paperShadow(ctx, x, y, w, h);
  ctx.fillStyle = COLORS.bar; rrect(ctx, x, y, w, h, 2); ctx.fill();
  ctx.strokeStyle = COLORS.barEdge; ctx.lineWidth = 1.5; ctx.stroke();
  
  // Dock: het punt waar de obers hun dienblad halen, 0,3 m voor de bar.
  const d = 20 / 67 * S;
  let dx, dy;
  if (b.wall === "links")       { dx = x + w + d; dy = y + 30 / 67 * S; }
  else if (b.wall === "rechts") { dx = x - d;     dy = y + 30 / 67 * S; }
  else if (b.wall === "boven")  { dx = x + 30 / 67 * S; dy = y + h + d; }
  else                          { dx = x + 30 / 67 * S; dy = y - d; }
  ctx.fillStyle = COLORS.accent; ctx.beginPath(); ctx.arc(dx, dy, Math.max(3, S * 0.12), 0, Math.PI * 2); ctx.fill();
  labelIn(ctx, x, y, w, h, label, S);
  if (selected) selection(ctx, x, y, w, h);
  ctx.restore();
}

function drawBuffet(ctx, b, view, selected, inactive) {
  const S = view.scale, x = view.ox + b.x * S, y = view.oy + b.y * S, w = b.w * S, h = b.h * S;
  ctx.save();
  if (inactive) ctx.globalAlpha = 0.4;
  paperShadow(ctx, x, y, w, h);
  ctx.fillStyle = COLORS.buffet; rrect(ctx, x, y, w, h, 2); ctx.fill();
  ctx.strokeStyle = COLORS.buffetEdge; ctx.lineWidth = 1.5; ctx.stroke();
  hatch(ctx, x, y, w, h, 8, "rgba(208,126,85,0.35)");
  // Opschepplekken aan de zaalkant
  const n = b.slots || 3, gap = 26 / 67 * S;
  ctx.fillStyle = COLORS.ink;
  for (let i = 0; i < n; i++) {
    const f = (i + 0.5) / n; let px, py;
    if (b.wall === "links")       { px = x + w + gap; py = y + f * h; }
    else if (b.wall === "rechts") { px = x - gap;     py = y + f * h; }
    else if (b.wall === "boven")  { px = x + f * w;   py = y + h + gap; }
    else                          { px = x + f * w;   py = y - gap; }
    ctx.beginPath(); ctx.arc(px, py, Math.max(2, S * 0.08), 0, Math.PI * 2); ctx.fill();
  }
  labelIn(ctx, x, y, w, h, "Buffet", S);
  if (selected) selection(ctx, x, y, w, h);
  ctx.restore();
}

function drawEntrance(ctx, e, view, selected) {
  const S = view.scale, x = view.ox + e.x * S, y = view.oy + e.y * S;
  const r = 0.9 * S / 2;  // deurvleugel ~0,9 m
  ctx.save();
  ctx.strokeStyle = COLORS.ink; ctx.lineWidth = 1.5;
  // Deuropening: gat in de wand + zwaaiboog
  const wall = e.wall || "onder";
  const d = 48 / 67 * S;
  let wx = x, wy = y, a0 = 0;
  if (wall === "boven") { wy = y - d; a0 = 0; }
  else if (wall === "onder") { wy = y + d; a0 = Math.PI; }
  else if (wall === "links") { wx = x - d; a0 = -Math.PI / 2; }
  else { wx = x + d; a0 = Math.PI / 2; }
  ctx.strokeStyle = COLORS.paper; ctx.lineWidth = Math.max(4, S * 0.24);
  ctx.beginPath();
  if (wall === "boven" || wall === "onder") { ctx.moveTo(wx - r, wy); ctx.lineTo(wx + r, wy); }
  else { ctx.moveTo(wx, wy - r); ctx.lineTo(wx, wy + r); }
  ctx.stroke();
  ctx.strokeStyle = COLORS.ink; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(wx - (wall === "boven" || wall === "onder" ? r : 0), wy - (wall === "links" || wall === "rechts" ? r : 0), 2 * r, a0, a0 + Math.PI / 2); ctx.stroke();
  ctx.fillStyle = COLORS.accent; ctx.beginPath(); ctx.arc(x, y, Math.max(3, S * 0.1), 0, Math.PI * 2); ctx.fill();
  if (S >= 14) { ctx.font = "500 11px Inter, system-ui, sans-serif"; ctx.fillStyle = COLORS.ink; ctx.textAlign = "center"; ctx.fillText("Ingang", x, y + (wall === "boven" ? 18 : -10)); }
  if (selected) selection(ctx, x - r, y - r, 2 * r, 2 * r);
  ctx.restore();
}

function labelIn(ctx, x, y, w, h, text, S) {
  if (S < 10) return;
  ctx.save();
  ctx.fillStyle = COLORS.ink; ctx.font = `600 ${Math.max(10, Math.min(13, S * 0.5))}px Inter, system-ui, sans-serif`;
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.translate(x + w / 2, y + h / 2);
  if (h > w * 1.4) ctx.rotate(-Math.PI / 2);
  ctx.fillText(text, 0, 0);
  ctx.restore();
}

export function drawTable(ctx, t, def, view, o = {}) {
  const S = view.scale;
  const cx = view.ox + (t.x + def.w / 2) * S, cy = view.oy + (t.y + def.h / 2) * S;
  ctx.save();
  // Stoelen
  const chairR = Math.max(2, 9 / 67 * S);
  for (const c of tableChairs(t, def)) {
    ctx.beginPath(); ctx.arc(view.ox + c.x * S, view.oy + c.y * S, chairR, 0, Math.PI * 2);
    ctx.fillStyle = o.dashed ? "rgba(136,149,139,0.35)" : "rgba(63,74,70,0.55)"; ctx.fill();
  }
  ctx.translate(cx, cy); ctx.rotate((t.rotation || 0) * Math.PI / 180);
  const w = def.w * S, h = def.h * S;
  if (!o.dashed) { ctx.shadowColor = "rgba(63,74,70,0.25)"; ctx.shadowBlur = 3; ctx.shadowOffsetX = 1; ctx.shadowOffsetY = 2; }
  ctx.fillStyle = o.fill || COLORS.table;
  rrect(ctx, -w / 2, -h / 2, w, h, Math.min(2, S * 0.04)); ctx.fill();
  ctx.shadowColor = "transparent";
  if (o.dashed) ctx.setLineDash([4, 3]);
  ctx.strokeStyle = o.edge || COLORS.tableEdge; ctx.lineWidth = o.strong ? 2 : 1.5; ctx.stroke();
  if (o.custom) { ctx.setLineDash([2, 2]); ctx.strokeStyle = COLORS.ink; ctx.lineWidth = 1; rrect(ctx, -w / 2 + 3, -h / 2 + 3, w - 6, h - 6, 2); ctx.stroke(); }
  if (o.label && S >= 16) {
    ctx.setLineDash([]);
    ctx.fillStyle = COLORS.ink; ctx.font = `500 ${Math.max(9, Math.min(12, S * 0.4))}px Inter, system-ui, sans-serif`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    if (h > w * 1.2) ctx.rotate(-Math.PI / 2);
    ctx.fillText(String(def.seats), 0, 0.5);
  }
  ctx.restore();
  if (o.selected) {
    const rad = (t.rotation || 0) * Math.PI / 180, cos = Math.abs(Math.cos(rad)), sin = Math.abs(Math.sin(rad));
    const bw = (def.w * cos + def.h * sin) * S, bh = (def.w * sin + def.h * cos) * S;
    selection(ctx, cx - bw / 2 - 4, cy - bh / 2 - 4, bw + 8, bh + 8);
  }
}
