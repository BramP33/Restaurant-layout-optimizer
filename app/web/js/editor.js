// Canvas-editor: gereedschappen, hit-testing, slepen, zoomen. Houdt zelf
// geen projectdata vast; hij krijgt het project via getProject() en meldt
// wijzigingen via onChange(). Zo blijft er één bron van waarheid.

import * as G from "./geometry.js";
import { drawPlan, screenToWorld } from "./render.js";

const MIN_SCALE = 4, MAX_SCALE = 220;

// Coordinaten op de centimeter afronden, zodat opgeslagen projecten geen
// 4.800000000000001 bevatten.
function roundItem(o) {
  for (const k of ["x", "y", "w", "h"]) if (typeof o[k] === "number") o[k] = Math.round(o[k] * 1000) / 1000;
  return o;
}

export class Editor {
  constructor(canvas, getProject, cb) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.getProject = getProject;
    this.cb = cb;                              // { onChange, onSelect, onStatus }
    this.view = { scale: 40, ox: 60, oy: 60 };
    this.tool = "select";
    this.selected = null;                      // { kind, index }
    this.hover = null;                         // voorbeeld tijdens plaatsen
    this.drag = null;
    this.params = { barLength: 4, buffetLength: 3, tableType: "medium", rotation: 0 };
    this.overlay = { candidateTables: null, showCurrent: true };
    this.photoImg = null; this.photoSrc = null;
    this.undoStack = [];
    this.spaceDown = false;
    this.mouse = null;

    this._bind();
    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(canvas.parentElement);
    this.resize();
  }

  // ── Grootte en weergave ────────────────────────────────────────────────
  resize() {
    const r = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(10, Math.round(r.width * dpr));
    this.canvas.height = Math.max(10, Math.round(r.height * dpr));
    this.canvas.style.width = r.width + "px"; this.canvas.style.height = r.height + "px";
    this.dpr = dpr;
    this.draw();
  }

  fit() {
    const p = this.getProject();
    const bb = G.bbox(p.rects);
    const w = this.canvas.width / this.dpr, h = this.canvas.height / this.dpr;
    if (!bb) {
      if (p.photo) {
        const s = Math.min((w - 80) / p.photo.w, (h - 80) / p.photo.h);
        this.view = { scale: s, ox: (w - p.photo.w * s) / 2 - p.photo.x * s, oy: (h - p.photo.h * s) / 2 - p.photo.y * s };
      } else {
        this.view = { scale: 40, ox: 80, oy: 80 };
      }
    } else {
      const s = G.clamp(Math.min((w - 120) / bb.w, (h - 120) / bb.h), MIN_SCALE, MAX_SCALE);
      this.view = { scale: s, ox: (w - bb.w * s) / 2 - bb.x * s, oy: (h - bb.h * s) / 2 - bb.y * s };
    }
    this.draw();
  }

  setTool(t) {
    this.tool = t; this.hover = null;
    this.cb.onTool?.(t);
    if (t !== "select") this.select(null);
    this.canvas.style.cursor = t === "select" ? "default" : t === "pan" ? "grab" : "crosshair";
    this.status();
    this.draw();
  }

  select(sel) {
    this.selected = sel;
    this.cb.onSelect?.(sel);
    this.draw();
  }

  status(extra) {
    const hints = {
      select: "Klik om te selecteren, sleep om te verplaatsen. Sleep op lege ruimte om te pannen.",
      rect:   "Sleep een rechthoek voor de vloer. Meerdere rechthoeken mogen overlappen.",
      block:  "Sleep een rechthoek voor een kolom, podium of ander vast obstakel.",
      bar:    "Beweeg naar een wand en klik om de bar te plaatsen. Lengte staat rechts.",
      buffet: "Beweeg naar een wand en klik om de buffetlijn te plaatsen.",
      entrance: "Klik op een wand voor de ingang.",
      table:  "Klik om een tafel van je huidige indeling te plaatsen. R draait.",
      photo:  "Sleep om de foto te verschuiven. Schaal en doorzichtigheid staan links.",
      pan:    "Sleep om te pannen, scroll om te zoomen.",
    };
    this.cb.onStatus?.(extra || hints[this.tool] || "");
  }

  // ── Tekenen ─────────────────────────────────────────────────────────────
  draw() {
    const p = this.getProject();
    if (p.photo && p.photo.dataUrl !== this.photoSrc) {
      this.photoSrc = p.photo.dataUrl;
      this.photoImg = new Image();
      this.photoImg.onload = () => this.draw();
      this.photoImg.src = p.photo.dataUrl;
    } else if (!p.photo) { this.photoImg = null; this.photoSrc = null; }
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    drawPlan(ctx, p, this.view, {
      photo: this.photoImg, selected: this.selected, hover: this.hover,
      candidateTables: this.overlay.candidateTables, showCurrent: this.overlay.showCurrent,
    });
  }

  // ── Invoer ──────────────────────────────────────────────────────────────
  _bind() {
    const c = this.canvas;
    c.addEventListener("pointerdown", e => this._down(e));
    c.addEventListener("pointermove", e => this._move(e));
    c.addEventListener("pointerup", e => this._up(e));
    c.addEventListener("pointerleave", () => { this.hover = null; this.mouse = null; this.draw(); });
    c.addEventListener("wheel", e => this._wheel(e), { passive: false });
    c.addEventListener("contextmenu", e => e.preventDefault());
    window.addEventListener("keydown", e => this._key(e));
    window.addEventListener("keyup", e => { if (e.code === "Space") this.spaceDown = false; });
  }

  _pt(e) {
    const r = this.canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }
  _world(e, snapTo = 0.1) {
    const w = screenToWorld(this.view, this._pt(e));
    return snapTo ? { x: G.snap(w.x, snapTo), y: G.snap(w.y, snapTo) } : w;
  }

  _wheel(e) {
    e.preventDefault();
    const pt = this._pt(e);
    const f = Math.exp(-e.deltaY * 0.0015);
    const ns = G.clamp(this.view.scale * f, MIN_SCALE, MAX_SCALE);
    const k = ns / this.view.scale;
    this.view.ox = pt.x - (pt.x - this.view.ox) * k;
    this.view.oy = pt.y - (pt.y - this.view.oy) * k;
    this.view.scale = ns;
    this.draw();
  }

  _key(e) {
    if (e.target && ["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
    if (e.code === "Space") { this.spaceDown = true; e.preventDefault(); return; }
    if (e.key === "Escape") { this.setTool("select"); return; }
    if ((e.key === "Delete" || e.key === "Backspace") && this.selected) { this.deleteSelected(); e.preventDefault(); return; }
    if ((e.key === "r" || e.key === "R")) { this.rotate(); return; }
    if ((e.ctrlKey || e.metaKey) && e.key === "z") { this.undo(); e.preventDefault(); return; }
  }

  // ── Undo ───────────────────────────────────────────────────────────────
  pushUndo() {
    const p = this.getProject();
    const snap = JSON.stringify({ rects: p.rects, bar: p.bar, buffet: p.buffet, entrance: p.entrance,
      blocks: p.blocks, currentLayout: p.currentLayout, photo: p.photo && { ...p.photo, dataUrl: undefined } });
    this.undoStack.push(snap);
    if (this.undoStack.length > 60) this.undoStack.shift();
  }
  undo() {
    const s = this.undoStack.pop();
    if (!s) return;
    const p = this.getProject();
    const d = JSON.parse(s);
    Object.assign(p, { rects: d.rects, bar: d.bar, buffet: d.buffet, entrance: d.entrance, blocks: d.blocks, currentLayout: d.currentLayout });
    if (p.photo && d.photo) Object.assign(p.photo, { x: d.photo.x, y: d.photo.y, w: d.photo.w, h: d.photo.h });
    this.select(null);
    this.changed();
  }
  changed() { this.cb.onChange?.(); this.draw(); }

  // ── Acties ──────────────────────────────────────────────────────────────
  deleteSelected() {
    const p = this.getProject(), s = this.selected;
    if (!s) return;
    this.pushUndo();
    if (s.kind === "rect") p.rects.splice(s.index, 1);
    else if (s.kind === "block") p.blocks.splice(s.index, 1);
    else if (s.kind === "table") p.currentLayout.splice(s.index, 1);
    else if (s.kind === "bar") p.bar = null;
    else if (s.kind === "buffet") p.buffet = null;
    else if (s.kind === "entrance") p.entrance = null;
    this.select(null);
    this.changed();
  }

  rotate() {
    const p = this.getProject();
    if (this.selected?.kind === "table") {
      this.pushUndo();
      const t = p.currentLayout[this.selected.index];
      t.rotation = ((t.rotation || 0) + 90) % 180;
      this.changed();
    } else if (this.tool === "table") {
      this.params.rotation = (this.params.rotation + 90) % 180;
      if (this.mouse) this._updateHover(this.mouse);
      this.draw();
    }
  }

  // Wandmeubel opnieuw plaatsen na een lengtewijziging in het paneel.
  resizeWallItem(kind, length) {
    const p = this.getProject();
    const item = p[kind];
    if (!item) return;
    const segs = G.wallSegments(p.rects);
    const hit = G.segmentFor(item, segs);
    if (!hit) return;
    const depth = kind === "bar" ? G.BAR_DEPTH_M : G.BUFFET_DEPTH_M;
    const center = { x: item.x + item.w / 2, y: item.y + item.h / 2 };
    const placed = G.placeOnWall(hit.seg, center, length, depth);
    Object.assign(item, placed);
    this.changed();
  }

  // ── Hit-test ────────────────────────────────────────────────────────────
  hitTest(w) {
    const p = this.getProject();
    if (p.entrance && Math.hypot(p.entrance.x - w.x, p.entrance.y - w.y) < 0.5) return { kind: "entrance" };
    for (let i = p.currentLayout.length - 1; i >= 0; i--) {
      const t = p.currentLayout[i];
      if (G.pointInRect(w, G.tableAABB(t, G.tableDef(t.type, p.tableTypes)), 0.1)) return { kind: "table", index: i };
    }
    if (p.bar && G.pointInRect(w, p.bar)) return { kind: "bar" };
    if (p.buffet && G.pointInRect(w, p.buffet)) return { kind: "buffet" };
    for (let i = p.blocks.length - 1; i >= 0; i--) if (G.pointInRect(w, p.blocks[i], 0.05)) return { kind: "block", index: i };
    for (let i = p.rects.length - 1; i >= 0; i--) if (G.pointInRect(w, p.rects[i])) return { kind: "rect", index: i };
    if (p.photo && this.tool === "photo") return { kind: "photo" };
    return null;
  }

  // ── Pointer ─────────────────────────────────────────────────────────────
  _down(e) {
    this.canvas.setPointerCapture(e.pointerId);
    const pt = this._pt(e);
    const w = this._world(e), wRaw = this._world(e, 0);
    const p = this.getProject();
    const pan = e.button === 1 || this.spaceDown || this.tool === "pan" || e.button === 2;
    if (pan) { this.drag = { kind: "pan", start: pt, view: { ...this.view } }; this.canvas.style.cursor = "grabbing"; return; }
    if (e.button !== 0) return;

    switch (this.tool) {
      case "select": {
        const hit = this.hitTest(wRaw);
        this.select(hit);
        if (hit) {
          const item = this._itemOf(hit);
          this.drag = { kind: "move", hit, start: w, orig: JSON.parse(JSON.stringify(item)), moved: false };
        } else {
          this.drag = { kind: "pan", start: pt, view: { ...this.view } };
        }
        break;
      }
      case "photo": {
        if (p.photo) this.drag = { kind: "move", hit: { kind: "photo" }, start: wRaw, orig: { ...p.photo }, moved: false };
        break;
      }
      case "rect":
      case "block":
        this.drag = { kind: "draw", start: w };
        break;
      case "bar":
      case "buffet":
      case "entrance":
      case "table":
        this._place(w, wRaw);
        break;
    }
  }

  _itemOf(hit) {
    const p = this.getProject();
    switch (hit.kind) {
      case "rect": return p.rects[hit.index];
      case "block": return p.blocks[hit.index];
      case "table": return p.currentLayout[hit.index];
      case "bar": return p.bar;
      case "buffet": return p.buffet;
      case "entrance": return p.entrance;
      case "photo": return p.photo;
    }
    return null;
  }

  _move(e) {
    const pt = this._pt(e);
    const w = this._world(e), wRaw = this._world(e, 0);
    this.mouse = w;
    this.cb.onCursor?.(wRaw);
    const d = this.drag;
    if (!d) { this._updateHover(w, wRaw); return; }

    if (d.kind === "pan") {
      this.view.ox = d.view.ox + (pt.x - d.start.x);
      this.view.oy = d.view.oy + (pt.y - d.start.y);
      this.draw(); return;
    }
    if (d.kind === "draw") {
      this.hover = { kind: this.tool, item: G.normRect({ x: d.start.x, y: d.start.y, w: w.x - d.start.x, h: w.y - d.start.y }) };
      this.draw(); return;
    }
    if (d.kind === "move") {
      const p = this.getProject();
      if (!d.moved) { this.pushUndo(); d.moved = true; }
      const dx = w.x - d.start.x, dy = w.y - d.start.y;
      const hit = d.hit, item = this._itemOf(hit);
      if (!item) return;
      if (hit.kind === "bar" || hit.kind === "buffet") {
        const segs = G.wallSegments(p.rects);
        const near = G.nearestWall(wRaw, segs, 2.5);
        if (near) {
          const depth = hit.kind === "bar" ? G.BAR_DEPTH_M : G.BUFFET_DEPTH_M;
          const len = hit.kind === "bar" ? Math.max(item.w, item.h) : Math.max(item.w, item.h);
          const placed = G.placeOnWall(near.seg, { x: G.snap(wRaw.x), y: G.snap(wRaw.y) }, len, depth);
          Object.assign(item, placed);
        }
      } else if (hit.kind === "entrance") {
        const near = G.nearestWall(wRaw, G.wallSegments(p.rects), 2.5);
        if (near) Object.assign(item, G.entranceFromWall(near.seg, { x: G.snap(near.point.x), y: G.snap(near.point.y) }));
      } else if (hit.kind === "photo") {
        item.x = d.orig.x + (wRaw.x - d.start.x); item.y = d.orig.y + (wRaw.y - d.start.y);
      } else {
        item.x = G.snap(d.orig.x + dx); item.y = G.snap(d.orig.y + dy);
      }
      this.draw();
    }
  }

  _up(e) {
    const d = this.drag;
    this.drag = null;
    if (!d) return;
    if (d.kind === "pan") { this.canvas.style.cursor = this.tool === "select" ? "default" : this.tool === "pan" ? "grab" : "crosshair"; return; }
    const p = this.getProject();
    if (d.kind === "draw") {
      const w = this._world(e);
      const r = G.normRect({ x: d.start.x, y: d.start.y, w: w.x - d.start.x, h: w.y - d.start.y });
      this.hover = null;
      const min = this.tool === "rect" ? 0.5 : 0.2;
      if (r.w >= min && r.h >= min) {
        r.x = G.round2(r.x); r.y = G.round2(r.y); r.w = G.round2(r.w); r.h = G.round2(r.h);
        this.pushUndo();
        if (this.tool === "rect") { p.rects.push(r); this.cb.onRectAdded?.(p.rects.length - 1); }
        else p.blocks.push(r);
        this.changed();
      } else this.draw();
      return;
    }
    if (d.kind === "move") {
      if (d.moved) this.changed(); else this.draw();
    }
  }

  _updateHover(w, wRaw = w) {
    const p = this.getProject();
    let hv = null;
    if (this.tool === "bar" || this.tool === "buffet") {
      const near = G.nearestWall(wRaw, G.wallSegments(p.rects), 2.5);
      if (near) {
        const depth = this.tool === "bar" ? G.BAR_DEPTH_M : G.BUFFET_DEPTH_M;
        const len = this.tool === "bar" ? this.params.barLength : this.params.buffetLength;
        hv = { kind: this.tool, item: { ...G.placeOnWall(near.seg, w, len, depth), slots: 3 } };
      }
    } else if (this.tool === "entrance") {
      const near = G.nearestWall(wRaw, G.wallSegments(p.rects), 2.5);
      if (near) hv = { kind: "entrance", item: G.entranceFromWall(near.seg, { x: G.snap(near.point.x), y: G.snap(near.point.y) }) };
    } else if (this.tool === "table") {
      const def = G.tableDef(this.params.tableType, p.tableTypes);
      hv = { kind: "table", def, item: { type: this.params.tableType, x: G.snap(w.x - def.w / 2), y: G.snap(w.y - def.h / 2), rotation: this.params.rotation } };
    }
    this.hover = hv;
    this.draw();
  }

  _place(w, wRaw) {
    const p = this.getProject();
    this._updateHover(w, wRaw);
    const hv = this.hover;
    if (!hv) { this.status("Geen wand in de buurt. Teken eerst de vloer, of beweeg dichter naar een wand."); return; }
    if (hv.kind === "bar" || hv.kind === "buffet") {
      if (!G.rectCovered(hv.item, p.rects)) { this.status("Hier past het niet: de zaal is op deze plek te ondiep."); return; }
      this.pushUndo();
      p[hv.kind] = roundItem({ ...hv.item });
      this.setTool("select"); this.select({ kind: hv.kind });
    } else if (hv.kind === "entrance") {
      this.pushUndo();
      p.entrance = roundItem({ ...hv.item });
      this.setTool("select"); this.select({ kind: "entrance" });
    } else if (hv.kind === "table") {
      const aabb = G.tableAABB(hv.item, hv.def);
      if (!G.rectCovered(aabb, p.rects)) { this.status("Een tafel moet helemaal op de vloer staan."); return; }
      this.pushUndo();
      p.currentLayout.push({ ...hv.item, id: Math.random().toString(36).slice(2, 8) });
    }
    this.changed();
  }
}
