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
          // De export heet totalServed/impatient; servedDrinks en
          // impatientGuests bestaan niet, waardoor deze kolommen altijd
          // null waren in validation-results.json.
          const served   = results.map(r => r.metrics.totalServed);
          const impatient= results.map(r => r.metrics.impatient);
          const valid    = results.map(r => r.metrics.layoutValid);
          const unreach  = results.map(r => r.metrics.unreachableTables);
          const trapped  = results.map(r => r.metrics.trappedWaiters);
          const failures = results.map(r => r.metrics.pathFailures);
          const wFail    = results.map(r => r.metrics.waiterPathFailures);
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
            // Een layout waarin de bar of een tafel onbereikbaar is levert
            // een bedrieglijk lage loopafstand op; deze vlaggen maken dat
            // zichtbaar zodat de pipeline zulke runs kan uitsluiten.
            // BatchRunner middelt metrics numeriek, dus false komt hier
            // aan als 0 -- vergelijken met === false zou altijd waar zijn.
            layout_valid:       valid.every(v => Boolean(v)),
            unreachable_tables: Math.max(0, ...unreach.map(u => u || 0)),
            trapped_waiters:    Math.max(0, ...trapped.map(t => t || 0)),
            path_failures:      Math.max(0, ...failures.map(f => f || 0)),
            // Alleen oberroutes bepalen waiterDist; een ober die geen route
            // vindt blijft staan en maakt de layout kunstmatig goedkoop.
            waiter_path_failures: Math.max(0, ...wFail.map(f => f || 0)),
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

  // simulatie.html zet window.__engine zelf klaar (zie de haak vlak na
  // `new SimulationEngine`). Voorheen viste dit script de engine uit een
  // klik-handler op batchStartBtn; dat werkte, maar brak stil zodra de UI
  // veranderde en startte ongevraagd een echte batch.
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
