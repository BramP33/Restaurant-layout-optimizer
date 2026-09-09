// Projectmodel en opslag. Alles in meters; de server rekent om naar de
// simulator. Opslag in localStorage, meerdere zalen naast elkaar.

const KEY = "zaalplanner.projects.v1";
const uid = () => Math.random().toString(36).slice(2, 10);

export const CUSTOM_SLOTS = 20;

export function defaultTableTypes() {
  const t = {};
  for (let i = 1; i <= CUSTOM_SLOTS; i++) {
    t["custom" + i] = { name: "", w: 1.2, h: 0.8, seats: 4, enabled: false };
  }
  return t;
}

export function newProject(name = "Nieuwe zaal") {
  return {
    id: uid(),
    name,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    photo: null,                       // { dataUrl, x, y, w, h, opacity }
    rects: [],                         // vloerrechthoeken (m)
    bar: null,                         // { wall, x, y, w, h }
    buffet: null,                      // { wall, x, y, w, h, slots }
    entrance: null,                    // { x, y, wall }
    blocks: [],                        // kolommen, podia
    tableTypes: defaultTableTypes(),   // eigen tafeltypes (20 slots)
    tables: { small: 0, medium: 6, large: 2 },
    currentLayout: [],                 // [{ id, type, x, y, rotation }]
    party: { type: "buffet", durationH: 5, hasBuffet: true, guests: 49, waiters: 3 },
    population: { avgAge: 40, ageSpread: 15, thirst: 1.0, appetite: 1.0, waiterSkill: 0.5 },
    generation: { quality: "normaal", top: 3 },
    job: null,                         // { id, status, pct, msg, ... }
    result: null,                      // samenvatting van de server
    view: { candidate: 1, showCurrent: true },
  };
}

// Ontbrekende velden aanvullen, zodat oude opgeslagen projecten blijven werken.
export function upgrade(p) {
  const d = newProject(p.name);
  const out = { ...d, ...p };
  out.party = { ...d.party, ...(p.party || {}) };
  out.population = { ...d.population, ...(p.population || {}) };
  out.generation = { ...d.generation, ...(p.generation || {}) };
  out.view = { ...d.view, ...(p.view || {}) };
  out.tableTypes = { ...d.tableTypes, ...(p.tableTypes || {}) };
  out.tables = { ...d.tables, ...(p.tables || {}) };
  return out;
}

export function loadAll() {
  try {
    const arr = JSON.parse(localStorage.getItem(KEY) || "[]");
    return Array.isArray(arr) ? arr.map(upgrade) : [];
  } catch (e) {
    return [];
  }
}

export function saveAll(projects) {
  try {
    localStorage.setItem(KEY, JSON.stringify(projects));
    return true;
  } catch (e) {
    // Meestal een te grote foto. De aanroeper toont dit aan de gebruiker.
    return false;
  }
}

export const PARTY_PRESETS = {
  receptie:   { label: "Receptie",          durationH: 2.5, hasBuffet: false },
  buffet:     { label: "Warm-koud buffet",  durationH: 5,   hasBuffet: true },
  feestavond: { label: "Feestavond",        durationH: 6.5, hasBuffet: true },
};

export const QUALITIES = {
  snel:    { label: "Snel",    hint: "ongeveer 3 tot 5 minuten, 3 simulaties per indeling" },
  normaal: { label: "Normaal", hint: "ongeveer 10 minuten, 7 simulaties per indeling" },
  grondig: { label: "Grondig", hint: "20 tot 30 minuten, 15 simulaties per indeling" },
};

// Foto verkleinen tot een handzaam dataURL, anders past hij niet in localStorage.
export function loadPhotoFile(file, maxSide = 1600) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      const s = Math.min(1, maxSide / Math.max(img.width, img.height));
      const c = document.createElement("canvas");
      c.width = Math.round(img.width * s); c.height = Math.round(img.height * s);
      c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
      URL.revokeObjectURL(url);
      resolve({ dataUrl: c.toDataURL("image/jpeg", 0.82), aspect: c.height / c.width });
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("Kon de afbeelding niet lezen.")); };
    img.src = url;
  });
}
