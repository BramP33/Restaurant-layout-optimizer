/**
 * Regressietest voor de bereikbaarheidscheck in simulatie.html.
 *
 * De optimizer had geleerd de bar in te metselen: een ober zonder route legt
 * geen afstand af, en dat is de goedkoopste "layout" die er bestaat. De
 * geldigheidsvlag moet zulke indelingen afkeuren. Deze test controleert de
 * eigenschap waar het echt om draait:
 *
 *     layoutValid == true   =>   waiterPathFailures == 0
 *
 * Een geldige layout mag dus geen enkele oberroute laten mislukken. Gastroutes
 * mogen wel falen (een stoel achter een muur is vervelend maar telt niet mee in
 * waiterDist); die worden apart gerapporteerd.
 *
 * Gebruik:
 *   node test_reachability.js                  # 40 geldige layouts
 *   node test_reachability.js --sample 250     # grondiger
 *   node test_reachability.js --data ../restaurant-sim-merged.json
 */
const { chromium } = require('playwright');
const path = require('path');
const fs   = require('fs');

const args   = process.argv.slice(2);
const argOf  = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const SAMPLE = parseInt(argOf('--sample', '40'), 10);
const HERE   = __dirname;
const ROOT   = path.join(HERE, '..');
const DATA   = path.resolve(argOf('--data', path.join(ROOT, 'restaurant-sim-merged.json')));
const SIM    = 'file://' + path.join(ROOT, 'simulatie.html');

const layoutKey = tables => tables
  .filter(t => t.size !== 'custom')
  .map(t => [t.size, Math.round(t.x), Math.round(t.y), Math.round(t.rotation || 0)].join(','))
  .sort().join('|');

function uniqueLayouts(runs) {
  const uniq = new Map();
  for (const r of runs) {
    if (!r.tables || !r.metrics) continue;
    const k = layoutKey(r.tables);
    if (!uniq.has(k)) uniq.set(k, { tables: r.tables, config: r.config || {}, dists: [] });
    uniq.get(k).dists.push(r.metrics.waiterDist);
  }
  return [...uniq.values()]
    .map(u => ({ ...u, dist: u.dists.reduce((a, b) => a + b, 0) / u.dists.length }))
    .sort((a, b) => a.dist - b.dist);
}

const cfgFor = s => ({
  roomW: s.config.roomW || 640, roomH: s.config.roomH || 640,
  guests: s.config.guests || 49, waiters: s.config.waiters || 3,
  tSmall: 0, tMedium: 0, tLarge: 0,
  partyType: s.config.partyType || 'buffet', gridSize: 24,
  forcedLayout: s.tables.filter(t => t.size !== 'custom'),
});

(async () => {
  if (!fs.existsSync(DATA)) {
    console.error(`Dataset niet gevonden: ${DATA}`);
    process.exit(2);
  }
  const raw  = JSON.parse(fs.readFileSync(DATA, 'utf8'));
  const runs = Array.isArray(raw) ? raw : (raw.runs || raw.data || []);
  const all  = uniqueLayouts(runs);
  console.log(`${runs.length.toLocaleString()} runs -> ${all.length.toLocaleString()} unieke layouts\n`);

  const browser = await chromium.launch({ headless: true });
  const page    = await browser.newPage();
  await page.goto(SIM, { waitUntil: 'networkidle' });
  if (!await page.evaluate(() => !!window.__engine)) {
    console.error('window.__engine ontbreekt — is de haak in simulatie.html weggevallen?');
    await browser.close();
    process.exit(2);
  }

  // 1. Statisch oordeel over de hele dataset (geen simulatie, dus snel).
  const verdicts = await page.evaluate(list => list.map(s => {
    const e = window.__engine;
    e._batchStart(s.cfg);
    const r = e._layoutReach();
    return { valid: r.valid, trapped: r.trappedWaiters, unreach: r.unreachable, boxed: r.dockBoxedIn };
  }), all.map(s => ({ cfg: cfgFor(s) })));

  const invalid = verdicts.filter(v => !v.valid).length;
  const firstOk = verdicts.findIndex(v => v.valid);
  console.log(`ongeldig: ${invalid}/${verdicts.length} (${(100 * invalid / verdicts.length).toFixed(1)}%)`);
  console.log(`  opgesloten ober: ${verdicts.filter(v => v.trapped > 0).length}`
            + `, onbereikbare tafel: ${verdicts.filter(v => v.unreach > 0).length}`);
  console.log(`goedkoopste GELDIGE layout: rang ${firstOk + 1}, ${Math.round(all[firstOk].dist).toLocaleString()} px`);
  console.log(`top-50 ongeldig: ${verdicts.slice(0, 50).filter(v => !v.valid).length}/50\n`);

  // 2. De eigenlijke invariant, met echte simulaties.
  const validIdx = verdicts.map((v, i) => [v, i]).filter(([v]) => v.valid).map(([, i]) => i);
  const step     = Math.max(1, Math.floor(validIdx.length / SAMPLE));
  const sample   = validIdx.filter((_, j) => j % step === 0).slice(0, SAMPLE);

  let violations = 0, worstGuest = 0;
  for (const i of sample) {
    const m = await page.evaluate(async cfg => {
      const e = window.__engine;
      e._batchStart(cfg);
      const safety = e.partyEndSimTime + 3 * 3600;
      let guard = 0;
      while (e.running && e.simTime <= safety && guard++ < 200000) e._batchTick(1);
      return e._batchSnapshot().metrics;
    }, cfgFor(all[i]));
    if (m.waiterPathFailures > 0) {
      violations++;
      console.log(`  SCHENDING layout #${i}: ${m.waiterPathFailures} mislukte oberroutes, `
                + `waiterDist ${Math.round(m.waiterDist).toLocaleString()} px`);
    }
    worstGuest = Math.max(worstGuest, m.guestPathFailures || 0);
  }
  await browser.close();

  console.log(`\n${sample.length} geldige layouts gesimuleerd -> ${violations} schendingen`);
  console.log(`hoogste aantal mislukte gastroutes: ${worstGuest} (mag, telt niet mee in waiterDist)`);
  if (violations) {
    console.log('\nFAAL: een geldig genoemde layout laat oberroutes mislukken.');
    process.exit(1);
  }
  console.log('\nOK: geen enkele geldige layout laat een oberroute mislukken.');
})();
