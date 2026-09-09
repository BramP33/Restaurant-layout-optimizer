// Afspelen van een indeling in de echte simulator, in een iframe van dezelfde
// oorsprong. De simulator zet window.__engine klaar; wij starten hem met
// dezelfde config als de headless validator, zodat wat je ziet ook is wat er
// gemeten is.

const $ = (id) => document.getElementById(id);

export class Playback {
  constructor() {
    this.modal = $("playModal");
    this.frame = $("playFrame");
    this.timer = null;
    this.speed = 5;
    $("playClose").addEventListener("click", () => this.close());
    $("playPause").addEventListener("click", () => this.togglePause());
    this.modal.querySelectorAll(".speed").forEach(b => b.addEventListener("click", () => {
      this.modal.querySelectorAll(".speed").forEach(x => x.classList.toggle("active", x === b));
      this.speed = parseInt(b.dataset.speed, 10);
      const eng = this.engine(); if (eng) eng.speedMul = this.speed;
    }));
    this.modal.addEventListener("click", e => { if (e.target === this.modal) this.close(); });
  }

  engine() { try { return this.frame.contentWindow && this.frame.contentWindow.__engine; } catch (e) { return null; } }

  open({ title, roomPx, config, tablesPx }) {
    $("playTitle").textContent = title;
    this.modal.hidden = false;
    $("playMetrics").innerHTML = "";
    this.frame.src = "/sim/simulatie.html?t=" + Date.now();
    this.frame.onload = () => {
      const win = this.frame.contentWindow, doc = this.frame.contentDocument;
      const style = doc.createElement("style");
      style.textContent = `
        .sidebar, .metrics, .editor-bar { display: none !important; }
        html, body { margin: 0; height: 100%; background: #EEF2F1 !important; overflow: hidden; display: block !important; }
        main.stage { position: fixed !important; inset: 0 !important; display: flex !important; align-items: center; justify-content: center; padding: 12px !important; box-sizing: border-box; background: #EEF2F1 !important; }
        main.stage > *:not(canvas#stage) { display: none !important; }
        canvas#stage { max-width: 100%; max-height: 100%; width: auto; height: auto; border-radius: 4px; box-shadow: 0 6px 18px rgba(63,74,70,.15); border: none !important; }
      `;
      doc.head.appendChild(style);
      const eng = win.__engine;
      if (!eng) { $("playMetrics").innerHTML = "<dt>Fout</dt><dd>Simulator niet geladen</dd>"; return; }
      const cfg = {
        ...config,
        room: roomPx, roomW: roomPx.w, roomH: roomPx.h,
        tSmall: 0, tMedium: 0, tLarge: 0,
        forcedLayout: tablesPx.filter(t => t.size !== "custom"),
        seed: null,
      };
      eng.start(cfg);
      eng.speedMul = this.speed;
      this.paused = false; $("playPause").textContent = "Pauze";
      clearInterval(this.timer);
      this.timer = setInterval(() => this.tick(), 500);
    };
  }

  tick() {
    const eng = this.engine();
    if (!eng) return;
    const m = eng.metrics || {};
    const t = eng.simTime || 0;
    const hh = Math.floor(t / 3600), mm = String(Math.floor((t % 3600) / 60)).padStart(2, "0");
    const km = ((m.waiterDist || 0) / 67 / 1000).toFixed(2).replace(".", ",");
    const wait = m.waitSamples ? Math.round(m.waitSum / m.waitSamples / 60) : 0;
    $("playMetrics").innerHTML = `
      <dt>Tijd</dt><dd>${hh}:${mm}${eng.running ? "" : " (klaar)"}</dd>
      <dt>Gelopen</dt><dd>${km} km</dd>
      <dt>Wachttijd</dt><dd>${wait} min</dd>
      <dt>Bediend</dt><dd>${m.totalServed || 0}</dd>
      <dt>Buffetbezoeken</dt><dd>${m.buffetVisits || 0}</dd>
      <dt>Ongeduldig</dt><dd>${m.impatient || 0}</dd>`;
  }

  togglePause() {
    const eng = this.engine(); if (!eng) return;
    this.paused = !this.paused;
    eng.paused = this.paused;
    $("playPause").textContent = this.paused ? "Verder" : "Pauze";
  }

  close() {
    clearInterval(this.timer);
    this.modal.hidden = true;
    this.frame.src = "about:blank";
  }
}
