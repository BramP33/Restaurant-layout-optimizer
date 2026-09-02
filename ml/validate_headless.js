/**
 * Headless validator voor optimizer resultaten.
 *
 * Opent de simulator in een headless browser, laadt de top-N optimizer
 * layouts, simuleert elk met 5 seeds en schrijft de echte scores naar
 * validation-results.json (bruikbaar voor active learning).
 *
 * Gebruik:
 *   node validate_headless.js
 *   node validate_headless.js --top 5 --seeds 10
 */

const { chromium } = require('playwright');
const path   = require('path');
const fs     = require('fs');

// ── Args ─────────────────────────────────────────────────────────────────────
const args    = process.argv.slice(2);
const TOP     = parseInt(args[args.indexOf('--top')   + 1] || '10');
const SEEDS   = parseInt(args[args.indexOf('--seeds') + 1] || '5');
const _inputIdx = args.indexOf('--input');
const INPUT   = _inputIdx >= 0 ? args[_inputIdx + 1] : null;
const _outIdx = args.indexOf('--out');
const OUT     = _outIdx >= 0 ? args[_outIdx + 1] : null;
const HERE    = __dirname;
const ROOT    = path.join(HERE, '..');   // data-artefacten staan in de repo-root
const SIM_URL = `file://${path.join(HERE, '..', 'simulatie.html')}`;
const OPT_IN  = INPUT ? path.resolve(INPUT) : path.join(ROOT, 'optimizer-results.json');
const VAL_OUT = OUT ? path.resolve(OUT) : path.join(ROOT, 'validation-results.json');

// ── Helpers ──────────────────────────────────────────────────────────────────
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function runValidation(page, layout, nSeeds) {
  // Inject de forcedLayout config en run een batch via de BatchRunner API
  const result = await page.evaluate(async ({ layout, nSeeds }) => {
    return new Promise((resolve) => {
      // Gebruik de globale engine + BatchRunner die al in de pagina leven
      const engine = window.__engine;
      if (!engine) { resolve({ error: 'engine niet gevonden' }); return; }

      const varTables = layout.tables.filter(t => t.size !== 'custom');
      const cfg = {
        roomW: layout.config.roomW || 640,
        roomH: layout.config.roomH || 640,
        guests: layout.config.guests || 49,
        waiters: layout.config.waiters || 3,
        tSmall: 0, tMedium: 0, tLarge: 0,
        partyType: layout.config.partyType || 'buffet',
        gridSize: 24,
        forcedLayout: varTables,
      };

      const batch  = new BatchRunner(engine);
      const runs   = [];
      batch.start({
        total:          nSeeds,
        seedsPerLayout: 1,
        configFn:       () => ({ ...cfg }),
        onProgress:     () => {},
        onFinish:       (results) => {
          const scores   = results.map(r => r.metrics.score);
          const dists    = results.map(r => r.metrics.waiterDist);
          const waits    = results.map(r => r.metrics.avgWait);
          const served   = results.map(r => r.metrics.servedDrinks);
          const impatient= results.map(r => r.metrics.impatientGuests);
          const avg      = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
          // Bouw trainingsdata-entries per seed (zelfde formaat als batch export)
          const trainingRuns = results.map(r => ({
            seed:    r.seed,
            seeds:   results.map(r2 => r2.seed),
            config:  cfg,
            metrics: r.metrics,
            tables:  layout.tables,
          }));
          resolve({
            rank:           layout.rank,
            predicted_dist: layout.predicted_waiterDist,
            predicted_score: layout.predicted_score,
            actual_dist:    avg(dists),
            actual_score:   avg(scores),
            seeds:          results.map(r => r.seed),
            dist_per_seed:  dists,
            score_per_seed: scores,
            wait_per_seed:  waits,
            served_per_seed:served,
            impatient_per_seed: impatient,
            tables:         layout.tables,
            config:         cfg,
            trainingRuns,
          });
        },
      });
    });
  }, { layout, nSeeds });

  return result;
}

// ── Main ─────────────────────────────────────────────────────────────────────
(async () => {
  if (!fs.existsSync(OPT_IN)) {
    console.error(`Niet gevonden: ${OPT_IN}`);
    process.exit(1);
  }

  const optResults = JSON.parse(fs.readFileSync(OPT_IN, 'utf8'));
  const toValidate = optResults.slice(0, TOP);
  console.log(`Valideer top-${toValidate.length} layouts × ${SEEDS} seeds elk`);
  console.log(`Simulator: ${SIM_URL}\n`);

  const browser = await chromium.launch({ headless: true });
  const page    = await browser.newPage();

  // Stille console (geen spam)
  page.on('console', msg => {
    if (msg.type() === 'error') console.error('Browser error:', msg.text());
  });

  console.log('Simulator laden…');
  await page.goto(SIM_URL, { waitUntil: 'networkidle' });

  // Wacht tot de engine klaar is en stel hem bloot
  await page.evaluate(() => {
    // De engine zit in de IIFE — we maken hem globaal toegankelijk via een hook
    // door een custom event te sturen; de engine is al gecreëerd bij DOMContentLoaded
    if (!window.__engine) {
      // Zoek de engine via de BatchRunner (die een referentie heeft)
      window.__engine = window._sim_engine || null;
    }
  });

  // Expose engine via window (de engine mist een globale referentie — patch dat)
  await page.evaluate(() => {
    // Simulatie is al gestart; canvas is beschikbaar — zoek SimulationEngine instantie
    // via de BatchRunner die al in scope is na DOMContentLoaded
    if (!window.__engine) {
      // Fallback: stuur een engine-expose event
      const origBatch = BatchRunner;
      window.__engine = null;
      // Haal engine op via het canvas element (SimulationEngine slaat canvas op)
      const canvas = document.getElementById('stage');
      if (canvas && canvas._engine) window.__engine = canvas._engine;
    }
  });

  // Betrouwbaardere methode: patch SimulationEngine constructor vóór laden
  // Herlaad pagina met engine-expose script geïnjecteerd
  await page.addInitScript(() => {
    window.__engineReady = false;
    const origConsole = console.log;
    // Wacht op SimulationEngine creatie door te patchen via prototype
    Object.defineProperty(window, '__setEngine', {
      set(v) { window.__engine = v; window.__engineReady = true; }
    });
  });

  // Herlaad zodat het init script actief is
  await page.reload({ waitUntil: 'networkidle' });

  // Patch na laden: zoek engine in globale scope van de IIFE
  const engineFound = await page.evaluate(() => {
    // De IIFE heeft `const engine = new SimulationEngine(...)` — niet globaal.
    // We moeten via BatchRunner of via een workaround.
    // Simpelste oplossing: definieer een globale factory-hook in SimulationEngine.
    // Maar die is al gecreëerd... gebruik de batch knop's click handler als proxy.

    // Alternatief: zoek via onclick handlers of event listeners.
    // Meest betrouwbaar: BatchRunner is ook niet globaal.
    // Laten we de engine blootstellen via een workaround:
    // We roepen _batchStart direct aan op het engine object dat we vinden
    // door het tijdelijk uit te lezen via de batchStartBtn listener.

    // Werkende aanpak: overschrijf BatchRunner.prototype.start zodat we engine opvangen
    if (typeof BatchRunner !== 'undefined') {
      const orig = BatchRunner.prototype.start;
      BatchRunner.prototype.start = function(opts) {
        window.__engine = this.engine;
        return orig.call(this, opts);
      };
      // Trigger een dummy click om de engine te exposen
      const btn = document.getElementById('batchStartBtn');
      if (btn) {
        // Sla de echte handler op en simuleer een start+stop
        btn.click();
        setTimeout(() => document.getElementById('batchStopBtn')?.click(), 0);
      }
      return true;
    }
    return false;
  });

  // Kleine pauze voor engine-expose
  await sleep(500);

  const hasEngine = await page.evaluate(() => !!window.__engine);
  if (!hasEngine) {
    console.error('Engine niet gevonden in pagina-scope. Controleer of simulatie geladen is.');
    await browser.close();
    process.exit(1);
  }

  console.log('Engine gevonden. Start validatie…\n');

  const validationResults = [];
  for (let i = 0; i < toValidate.length; i++) {
    const layout = toValidate[i];
    process.stdout.write(`  Layout #${layout.rank} (${i+1}/${toValidate.length})… `);
    const t0 = Date.now();

    try {
      const result = await runValidation(page, layout, SEEDS);
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

      if (result.error) {
        console.log(`FOUT: ${result.error}`);
      } else {
        const scoreErr = (result.actual_score - result.predicted_score).toFixed(0);
        const distErr  = Math.round(result.actual_dist - result.predicted_dist).toLocaleString();
        console.log(`actueel score=${result.actual_score.toFixed(0)}  ` +
                    `dist=${Math.round(result.actual_dist).toLocaleString()} px  ` +
                    `(fout: score${scoreErr > 0 ? '+' : ''}${scoreErr}, dist${distErr})  ` +
                    `[${elapsed}s]`);
        validationResults.push(result);
      }
    } catch (err) {
      console.log(`UITZONDERING: ${err.message}`);
    }
  }

  await browser.close();

  if (validationResults.length === 0) {
    console.error('\nGeen resultaten verzameld.');
    process.exit(1);
  }

  // Sorteer op werkelijke score (hoog = beter)
  validationResults.sort((a, b) => b.actual_score - a.actual_score);

  fs.writeFileSync(VAL_OUT, JSON.stringify(validationResults, null, 2));
  console.log(`\n${validationResults.length} gevalideerde layouts → ${VAL_OUT}`);

  // Samenvatting
  console.log('\n── Validatieresultaten (gesorteerd op echte score) ────────────');
  const pad = (s, n) => String(s).padEnd(n);
  console.log(`  ${pad('#rank',6)} ${pad('voorspeld',12)} ${pad('actueel',12)} fout`);
  for (const r of validationResults) {
    const err = (r.actual_score - r.predicted_score).toFixed(0);
    console.log(`  ${pad('#'+r.rank,6)} ${pad(r.predicted_score.toFixed(0),12)} ${pad(r.actual_score.toFixed(0),12)} ${err}`);
  }

  // Best gevonden layout
  const best = validationResults[0];
  console.log(`\n★ Beste layout: rank #${best.rank}  echte score=${best.actual_score.toFixed(0)}  dist=${Math.round(best.actual_dist).toLocaleString()} px`);
})();
