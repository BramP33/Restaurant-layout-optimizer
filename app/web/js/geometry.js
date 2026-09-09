// Meetkunde in meters. Geen DOM, geen state: pure functies die het model
// (rechthoeken, wandsegmenten, meubels) beschrijven zodat editor, tekenlaag
// en export dezelfde antwoorden krijgen.

export const PX_PER_M = 67;
export const EPS = 1e-6;

export const TABLE_TYPES_PX = {
  small:  { w: 60,  h: 60, seats: 2, name: "Klein" },
  medium: { w: 80,  h: 60, seats: 4, name: "Middel" },
  large:  { w: 110, h: 70, seats: 6, name: "Groot" },
};
export const BAR_DEPTH_M    = 70 / PX_PER_M;
export const BUFFET_DEPTH_M = 60 / PX_PER_M;
export const ENTRANCE_INSET_M = 48 / PX_PER_M;   // afstand van het ingangspunt tot de wand
export const WALL_SIDES = ["boven", "onder", "links", "rechts"];

export const snap = (v, step = 0.1) => Math.round(v / step) * step;
export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
export const round2 = (v) => Math.round(v * 100) / 100;

export function normRect(r) {
  const x = Math.min(r.x, r.x + r.w), y = Math.min(r.y, r.y + r.h);
  return { x, y, w: Math.abs(r.w), h: Math.abs(r.h) };
}

export function bbox(rects) {
  if (!rects.length) return null;
  const x0 = Math.min(...rects.map(r => r.x)), y0 = Math.min(...rects.map(r => r.y));
  const x1 = Math.max(...rects.map(r => r.x + r.w)), y1 = Math.max(...rects.map(r => r.y + r.h));
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

export function pointInRect(p, r, pad = 0) {
  return p.x >= r.x - pad - EPS && p.x <= r.x + r.w + pad + EPS &&
         p.y >= r.y - pad - EPS && p.y <= r.y + r.h + pad + EPS;
}

export function pointInFloor(p, rects) {
  return rects.some(r => pointInRect(p, r));
}

export function rectsOverlap(a, b, pad = 0) {
  return a.x - pad < b.x + b.w && a.x + a.w + pad > b.x &&
         a.y - pad < b.y + b.h && a.y + a.h + pad > b.y;
}

// Rasterdecompositie op alle randcoordinaten; dezelfde aanpak als
// convert.complement_blocks in de server, zodat beide kanten dezelfde
// blokken zien.
export function decompose(rects) {
  const xs = [...new Set(rects.flatMap(r => [r.x, r.x + r.w]))].sort((a, b) => a - b);
  const ys = [...new Set(rects.flatMap(r => [r.y, r.y + r.h]))].sort((a, b) => a - b);
  const covered = (cx, cy) => rects.some(r => pointInRect({ x: cx, y: cy }, r));
  const cells = [];
  for (let j = 0; j < ys.length - 1; j++) {
    const row = [];
    for (let i = 0; i < xs.length - 1; i++) {
      row.push(covered((xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2));
    }
    cells.push(row);
  }
  return { xs, ys, cells };
}

export function complementBlocks(rects) {
  if (!rects.length) return [];
  const { xs, ys, cells } = decompose(rects);
  const out = [];
  for (let j = 0; j < cells.length; j++) {
    for (let i = 0; i < cells[j].length; i++) {
      if (!cells[j][i]) out.push({ x: xs[i], y: ys[j], w: xs[i + 1] - xs[i], h: ys[j + 1] - ys[j] });
    }
  }
  return out;
}

// Is de rechthoek volledig vloer? (Alle rastercellen die hij raakt zijn bedekt.)
export function rectCovered(r, rects) {
  if (!rects.length) return false;
  const { xs, ys, cells } = decompose(rects);
  const b = bbox(rects);
  if (r.x < b.x - EPS || r.y < b.y - EPS || r.x + r.w > b.x + b.w + EPS || r.y + r.h > b.y + b.h + EPS) return false;
  for (let j = 0; j < cells.length; j++) {
    if (ys[j + 1] <= r.y + EPS || ys[j] >= r.y + r.h - EPS) continue;
    for (let i = 0; i < cells[j].length; i++) {
      if (xs[i + 1] <= r.x + EPS || xs[i] >= r.x + r.w - EPS) continue;
      if (!cells[j][i]) return false;
    }
  }
  return true;
}

// Wandsegmenten van de vereniging. `side` is de kant van de zaal waar de
// wand staat, gezien vanaf de vloer: een "boven"-wand heeft de vloer eronder.
export function wallSegments(rects) {
  if (!rects.length) return [];
  const { xs, ys, cells } = decompose(rects);
  const cov = (i, j) => j >= 0 && j < cells.length && i >= 0 && i < cells[j].length && cells[j][i];
  const segs = [];
  for (let j = 0; j < cells.length; j++) {
    for (let i = 0; i < cells[j].length; i++) {
      if (!cells[j][i]) continue;
      if (!cov(i, j - 1)) segs.push({ side: "boven",  x1: xs[i], y1: ys[j],     x2: xs[i + 1], y2: ys[j] });
      if (!cov(i, j + 1)) segs.push({ side: "onder",  x1: xs[i], y1: ys[j + 1], x2: xs[i + 1], y2: ys[j + 1] });
      if (!cov(i - 1, j)) segs.push({ side: "links",  x1: xs[i], y1: ys[j],     x2: xs[i],     y2: ys[j + 1] });
      if (!cov(i + 1, j)) segs.push({ side: "rechts", x1: xs[i + 1], y1: ys[j], x2: xs[i + 1], y2: ys[j + 1] });
    }
  }
  // Collineaire, aansluitende segmenten met dezelfde kant samenvoegen.
  const merged = [];
  const used = new Array(segs.length).fill(false);
  for (let a = 0; a < segs.length; a++) {
    if (used[a]) continue;
    let s = { ...segs[a] };
    used[a] = true;
    let grew = true;
    while (grew) {
      grew = false;
      for (let b = 0; b < segs.length; b++) {
        if (used[b]) continue;
        const t = segs[b];
        if (t.side !== s.side) continue;
        const horiz = s.side === "boven" || s.side === "onder";
        if (horiz && Math.abs(t.y1 - s.y1) < EPS) {
          if (Math.abs(t.x1 - s.x2) < EPS) { s.x2 = t.x2; used[b] = true; grew = true; }
          else if (Math.abs(t.x2 - s.x1) < EPS) { s.x1 = t.x1; used[b] = true; grew = true; }
        } else if (!horiz && Math.abs(t.x1 - s.x1) < EPS) {
          if (Math.abs(t.y1 - s.y2) < EPS) { s.y2 = t.y2; used[b] = true; grew = true; }
          else if (Math.abs(t.y2 - s.y1) < EPS) { s.y1 = t.y1; used[b] = true; grew = true; }
        }
      }
    }
    merged.push(s);
  }
  return merged.map(s => ({ ...s, horizontal: s.side === "boven" || s.side === "onder",
                             length: Math.hypot(s.x2 - s.x1, s.y2 - s.y1) }));
}

// Dichtstbijzijnde punt op een wandsegment.
export function nearestWall(p, segs, maxDist = Infinity) {
  let best = null;
  for (const s of segs) {
    let t, q;
    if (s.horizontal) { t = clamp((p.x - s.x1) / (s.x2 - s.x1), 0, 1); q = { x: s.x1 + t * (s.x2 - s.x1), y: s.y1 }; }
    else              { t = clamp((p.y - s.y1) / (s.y2 - s.y1), 0, 1); q = { x: s.x1, y: s.y1 + t * (s.y2 - s.y1) }; }
    const d = Math.hypot(q.x - p.x, q.y - p.y);
    if (d <= maxDist && (!best || d < best.dist)) best = { seg: s, t, point: q, dist: d };
  }
  return best;
}

// Een rechthoek van `length` langs de wand en `depth` de zaal in, gecentreerd
// op het punt `center` (langs de wand), binnen het segment gehouden.
export function placeOnWall(seg, center, length, depth) {
  const L = Math.min(length, seg.length);
  if (seg.horizontal) {
    const lo = Math.min(seg.x1, seg.x2), hi = Math.max(seg.x1, seg.x2);
    const cx = clamp(center.x, lo + L / 2, hi - L / 2);
    const x = cx - L / 2;
    const y = seg.side === "boven" ? seg.y1 : seg.y1 - depth;
    return { wall: seg.side, x, y, w: L, h: depth };
  }
  const lo = Math.min(seg.y1, seg.y2), hi = Math.max(seg.y1, seg.y2);
  const cy = clamp(center.y, lo + L / 2, hi - L / 2);
  const y = cy - L / 2;
  const x = seg.side === "links" ? seg.x1 : seg.x1 - depth;
  return { wall: seg.side, x, y, w: depth, h: L };
}

// Het segment waar een wandmeubel op staat (om het langs de wand te schuiven).
export function segmentFor(item, segs) {
  const c = itemWallPoint(item);
  return nearestWall(c, segs.filter(s => s.side === item.wall), 0.05) || nearestWall(c, segs, 0.3);
}

export function itemWallPoint(item) {
  switch (item.wall) {
    case "boven":  return { x: item.x + item.w / 2, y: item.y };
    case "onder":  return { x: item.x + item.w / 2, y: item.y + item.h };
    case "links":  return { x: item.x, y: item.y + item.h / 2 };
    default:       return { x: item.x + item.w, y: item.y + item.h / 2 };
  }
}

export function entranceFromWall(seg, point) {
  const d = ENTRANCE_INSET_M;
  switch (seg.side) {
    case "boven":  return { x: point.x, y: point.y + d, wall: "boven" };
    case "onder":  return { x: point.x, y: point.y - d, wall: "onder" };
    case "links":  return { x: point.x + d, y: point.y, wall: "links" };
    default:       return { x: point.x - d, y: point.y, wall: "rechts" };
  }
}

// ── Tafels ────────────────────────────────────────────────────────────────

export function tableDef(type, tableTypes) {
  if (TABLE_TYPES_PX[type]) {
    const d = TABLE_TYPES_PX[type];
    return { w: d.w / PX_PER_M, h: d.h / PX_PER_M, seats: d.seats, name: d.name, fixed: true };
  }
  const c = (tableTypes || {})[type] || {};
  return { w: c.w || 1.2, h: c.h || 0.8, seats: c.seats || 4, name: c.name || type, fixed: false };
}

// AABB van een gedraaide tafel (x,y = linksboven van de ongedraaide tafel).
export function tableAABB(t, def) {
  const cx = t.x + def.w / 2, cy = t.y + def.h / 2;
  const rad = (t.rotation || 0) * Math.PI / 180;
  const cos = Math.abs(Math.cos(rad)), sin = Math.abs(Math.sin(rad));
  const bw = def.w * cos + def.h * sin, bh = def.w * sin + def.h * cos;
  return { x: cx - bw / 2, y: cy - bh / 2, w: bw, h: bh };
}

// Stoelankers, zelfde formule als computeChairs() in de simulator.
export function tableChairs(t, def) {
  const cx = t.x + def.w / 2, cy = t.y + def.h / 2;
  const rad = (t.rotation || 0) * Math.PI / 180;
  const cos0 = Math.cos(rad), sin0 = Math.sin(rad);
  const n = def.seats, margin = 14 / PX_PER_M;
  const top = Math.ceil(n / 2), bottom = Math.floor(n / 2);
  const hy = def.h / 2 + margin, local = [];
  for (let i = 0; i < top; i++)    { const f = top > 1 ? i / (top - 1) : 0.5;       local.push([(f - 0.5) * def.w * 0.8, -hy]); }
  for (let i = 0; i < bottom; i++) { const f = bottom > 1 ? i / (bottom - 1) : 0.5; local.push([(f - 0.5) * def.w * 0.8,  hy]); }
  return local.map(([dx, dy]) => ({ x: cx + dx * cos0 - dy * sin0, y: cy + dx * sin0 + dy * cos0 }));
}

export function totalSeats(tables, tableTypes) {
  return Object.entries(tables || {}).reduce((s, [type, n]) => s + (n || 0) * tableDef(type, tableTypes).seats, 0);
}
