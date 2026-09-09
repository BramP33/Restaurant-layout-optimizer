// Resultaatkaarten. Toont kandidaten (en de eigen indeling) met looplengte,
// betrouwbaarheidsinterval en verschil met de huidige indeling.

const $ = (id) => document.getElementById(id);
const nl = (v, d = 1) => v.toLocaleString("nl-NL", { minimumFractionDigits: d, maximumFractionDigits: d });

export function fmtKm(m) { return nl(m / 1000, 2) + " km"; }

export function renderResults(p, onSelect) {
  const r = p.result;
  $("resultsEmpty").hidden = !!r;
  $("resultsBody").hidden = !r;
  if (!r) return;

  const cur = r.current;
  // Vergelijken heeft alleen zin als de eigen indeling evenveel tafels telt
  // als de voorstellen; anders meet je het aantal tafels, niet de indeling.
  const nGen = r.candidates[0] ? r.candidates[0].tables.length : 0;
  const nCur = cur ? cur.tables.filter(t => t.type !== "custom").length : 0;
  const comparable = cur && cur.layout_valid && nCur === nGen;
  const base = comparable ? cur.actual_m : null;
  const seeds = r.seeds;
  $("resultsMeta").textContent =
    `${r.candidates.length} voorstellen, elk ${seeds} keer gesimuleerd (${r.quality}). ` +
    `Looplengte is de totale afstand van de bediening over het hele feest; lager is beter.` +
    (cur && !comparable && cur.layout_valid ? ` Je huidige indeling telt ${nCur} tafels en de voorstellen ${nGen}; het verschil is daardoor niet vergelijkbaar.` : "") +
    (r.warnings?.length ? " " + r.warnings.join(" ") : "");

  const cards = $("resultCards");
  cards.innerHTML = "";
  const all = [...r.candidates];
  if (cur) all.push(cur);
  for (const c of all) {
    const el = document.createElement("div");
    el.className = "card" + (c.is_current ? " current" : "") + (p.view.candidate === (c.is_current ? 0 : c.rank) ? " active" : "");
    const ci = c.ci95_m;
    const pm = c.sem_m ? "± " + fmtKm(1.96 * c.sem_m) : "";
    let delta = "";
    if (!c.is_current && base) {
      const d = (c.actual_m - base) / base * 100;
      delta = `<span class="delta ${d < 0 ? "good" : "bad"}">${d < 0 ? "" : "+"}${nl(d, 1)}% t.o.v. huidig</span>`;
    }
    const flags = [];
    if (!c.layout_valid) flags.push("onbereikbare tafels");
    if (c.is_current && c.placement_ok === false) flags.push("te krap volgens de plaatsingsregels");
    el.innerHTML = `
      <div class="rank">${c.is_current ? "nu" : c.rank}</div>
      <div class="title">${c.is_current ? "Huidige indeling" : "Voorstel " + c.rank}</div>
      <div class="value">${fmtKm(c.actual_m)}</div>
      <div class="sub">
        <span>${pm}</span>
        <span>wacht ${nl(c.avg_wait_s / 60, 0)} min</span>
        <span>bediend ${nl(c.served, 0)}</span>
        ${delta}
        ${flags.map(f => `<span class="flag">${f}</span>`).join("")}
      </div>`;
    el.addEventListener("click", () => onSelect(c.is_current ? 0 : c.rank));
    cards.appendChild(el);
  }
}

export function pickCandidate(p) {
  const r = p.result;
  if (!r) return null;
  if (p.view.candidate === 0) return r.current;
  return r.candidates.find(c => c.rank === p.view.candidate) || r.candidates[0] || null;
}
