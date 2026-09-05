/**
 * Werkt de seed door op het gezelschap, en is dat reproduceerbaar?
 *
 * Twee eisen die makkelijk stilletjes breken:
 *   1. dezelfde seed geeft exact hetzelfde gezelschap (leeftijden, dorst,
 *      vakmanschap) -- anders is geen enkele meting herhaalbaar;
 *   2. een andere seed geeft een ander gezelschap -- anders doet de seed
 *      alleen iets met de loop van de simulatie en niet met wie er binnenkomt.
 *
 * Gebruik: node test_population.js
 */
const path = require("path");
const { chromium } = require("playwright");

const SIM = `file://${path.join(__dirname, "..", "simulatie.html")}`;

function summarise(page, seed, population) {
  return page.evaluate(({ seed, population }) => {
    const engine = window.__engine;
    const cfg = {
      roomW: 640, roomH: 640, guests: 49, waiters: 3,
      tSmall: 0, tMedium: 6, tLarge: 2, partyType: "buffet",
      gridSize: 24, seed, population,
    };
    // Bouw de wereld op met deze seed zonder de hele avond te draaien: de
    // eigenschappen worden bij het aanmaken van de agents getrokken.
    setSeed(seed);
    const pop = { ...POPULATION_DEFAULTS, ...(population || {}) };
    const guests = [], waiters = [];
    for (let i = 0; i < 20; i++) guests.push(new Guest(i, { x: 0, y: 0 }, pop));
    for (let i = 0; i < 3; i++)  waiters.push(new Waiter(i, { x: 0, y: 0 }, "#fff", pop));
    return {
      ages:    guests.map(g => g.age),
      thirst:  guests.map(g => +g.thirst.toFixed(4)),
      speeds:  guests.map(g => +g.speed.toFixed(3)),
      skills:  waiters.map(w => +w.skill.toFixed(4)),
      drinkCap: waiters.map(w => w.capacity),
      plateCap: waiters.map(w => w.plateCapacity),
    };
  }, { seed, population });
}

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto(SIM);
  await page.waitForFunction(() => !!window.__engine, { timeout: 20000 });

  const a1 = await summarise(page, 12345, null);
  const a2 = await summarise(page, 12345, null);
  const b  = await summarise(page, 99999, null);
  const oud = await summarise(page, 12345, { avgAge: 70, ageSpread: 5 });
  const dorstig = await summarise(page, 12345, { thirst: 1.8 });
  const groen = await summarise(page, 12345, { waiterSkill: 0.0, skillSpread: 0.01 });
  const top = await summarise(page, 12345, { waiterSkill: 1.0, skillSpread: 0.01 });

  await browser.close();

  const eq = (x, y) => JSON.stringify(x) === JSON.stringify(y);
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  let fail = 0;
  const check = (naam, ok, extra = "") => {
    console.log(`  ${ok ? "OK  " : "FAAL"} ${naam}${extra ? "  — " + extra : ""}`);
    if (!ok) fail++;
  };

  console.log("Seed en gezelschap\n");
  check("zelfde seed geeft exact hetzelfde gezelschap", eq(a1, a2));
  check("andere seed geeft een ander gezelschap", !eq(a1, b));
  check("leeftijd stuurt de loopsnelheid",
        mean(oud.speeds) < mean(a1.speeds),
        `gem. leeftijd ${mean(oud.ages).toFixed(0)} loopt ${mean(oud.speeds).toFixed(1)} px/s tegen ${mean(a1.speeds).toFixed(1)} bij ${mean(a1.ages).toFixed(0)}`);
  check("dorst schaalt mee",
        mean(dorstig.thirst) > mean(a1.thirst),
        `${mean(dorstig.thirst).toFixed(2)} tegen ${mean(a1.thirst).toFixed(2)}`);
  check("vakmanschap stuurt het draagvermogen",
        Math.max(...top.plateCap) > Math.max(...groen.plateCap),
        `borden ${groen.plateCap[0]} (groen) tot ${top.plateCap[0]} (ervaren), drankjes ${groen.drinkCap[0]} tot ${top.drinkCap[0]}`);
  check("borden nooit boven de 7", Math.max(...top.plateCap) <= 7,
        `max ${Math.max(...top.plateCap)}`);

  // ── Bereikt het profiel ook de obers via het ECHTE batch-pad? ──────────
  //
  // Dit is het gat dat de review vond: `_batchStart` en
  // `_batchStartWithLayout` bouwden hun obers zonder profiel, dus
  // `waiterSkill` deed in de hele ML-pipeline niets terwijl de interactieve
  // modus wel werkte. Een test die zelf `new Waiter(...)` aanroept ziet dat
  // niet -- die moet door de weg die de batch echt neemt.
  const browser2 = await chromium.launch();
  const page2 = await browser2.newPage();
  await page2.goto(SIM);
  await page2.waitForFunction(() => !!window.__engine, { timeout: 20000 });

  const viaBatch = (skill) => page2.evaluate((skill) => {
    const engine = window.__engine;
    const tables = [
      { size: "large",  x: 213, y: 351, rotation: 90 },
      { size: "large",  x: 291, y: 388, rotation: 90 },
      { size: "medium", x: 222, y: 34,  rotation: 90 },
      { size: "medium", x: 277, y: 186, rotation: 90 },
      { size: "medium", x: 404, y: 417, rotation: 90 },
      { size: "medium", x: 432, y: 315, rotation: 90 },
      { size: "medium", x: 454, y: 21,  rotation: 90 },
      { size: "medium", x: 463, y: 152, rotation: 90 },
    ];
    const cfg = {
      roomW: 640, roomH: 640, guests: 49, waiters: 3,
      tSmall: 0, tMedium: 0, tLarge: 0, partyType: "buffet", gridSize: 24,
      forcedLayout: tables,
      population: { waiterSkill: skill, skillSpread: 0.01 },
    };
    // BEIDE batch-paden toetsen. BatchRunner gebruikt _batchStart voor de
    // eerste seed en _batchStartWithLayout voor de volgende, dus een test die
    // er maar een aanraakt kan het profiel op het andere pad stil verliezen --
    // precies hoe deze bug er de eerste keer in kwam.
    engine._batchStart({ ...cfg, seed: 4242 });
    const viaStart = {
      skills:   engine.waiters.map(w => +w.skill.toFixed(3)),
      drinkCap: engine.waiters.map(w => w.capacity),
      plateCap: engine.waiters.map(w => w.plateCapacity),
    };
    const built = engine.tables;
    engine._batchStartWithLayout({ ...cfg, seed: 4242 }, built, 4242);
    const viaLayout = {
      skills:   engine.waiters.map(w => +w.skill.toFixed(3)),
      drinkCap: engine.waiters.map(w => w.capacity),
      plateCap: engine.waiters.map(w => w.plateCapacity),
    };
    return { viaStart, viaLayout };
  }, skill);

  const groenB = await viaBatch(0.0);
  const topB   = await viaBatch(1.0);
  await browser2.close();

  check("profiel bereikt de obers via _batchStart",
        !eq(groenB.viaStart, topB.viaStart),
        `groen ${groenB.viaStart.plateCap[0]} borden / ${groenB.viaStart.drinkCap[0]} drankjes  |  ` +
        `ervaren ${topB.viaStart.plateCap[0]} / ${topB.viaStart.drinkCap[0]}`);
  check("profiel bereikt de obers via _batchStartWithLayout",
        !eq(groenB.viaLayout, topB.viaLayout),
        `groen ${groenB.viaLayout.plateCap[0]} borden / ${groenB.viaLayout.drinkCap[0]} drankjes  |  ` +
        `ervaren ${topB.viaLayout.plateCap[0]} / ${topB.viaLayout.drinkCap[0]}`);
  check("borden ook via beide batch-paden nooit boven 7",
        Math.max(...topB.viaStart.plateCap, ...topB.viaLayout.plateCap,
                 ...groenB.viaStart.plateCap, ...groenB.viaLayout.plateCap) <= 7);

  console.log(fail ? `\n${fail} controle(s) gefaald.` : "\nAlles in orde.");
  process.exit(fail ? 1 : 0);
})();
