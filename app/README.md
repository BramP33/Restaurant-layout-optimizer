# Zaalplanner

Browser-app waarmee een uitbater of floormanager zijn eigen zaal intekent
(over een foto of tekening van de plattegrond), tafels, feest en gezelschap
opgeeft, en dan de door het model gegenereerde tafelindelingen te zien
krijgt, gevalideerd met de echte simulator.

## Starten

```bash
app/start.sh
```

Open daarna <http://localhost:8765>. De server gebruikt de `.venv` van het
project (FastAPI en uvicorn staan erin), Node met Playwright uit
`node_modules`, en `surrogate_model.pkl` uit de repo-root.

## Opbouw

```
app/
  start.sh              start de server
  server/
    main.py             FastAPI: statische bestanden + job-API
    convert.py          app-model (meters) -> simulatiezaal (px, 67 per m)
    run_optimizer.py    los proces rond ml/optimize_layout.py met voortgang
    jobs/               per job: input.json, optimizer-results.json,
                        validation-results.json, result.json (gitignored)
  web/
    index.html, styles.css
    js/geometry.js      vorm, wandsegmenten, tafels; puur, geen DOM
    js/render.js        tekent het plan (editor en PNG-export)
    js/editor.js        canvas-gereedschappen, selectie, slepen, undo
    js/model.js         projectmodel, opslag in localStorage, presets
    js/results.js       resultaatkaarten
    js/playback.js      simulator afspelen in een iframe
    js/api.js, js/main.js
```

## Hoe een job loopt

1. De browser stuurt het project (zonder foto) naar `POST /api/jobs`.
2. `convert.project_to_job` maakt er een `room`-object van zoals
   `ml/rooms.py` dat beschrijft: omhullende rechthoek, de weggesneden delen
   als `blocks`, bar met dock, buffetlijn, ingang. Meters worden pixels.
3. `run_optimizer.py` draait de bestaande brede zoektocht en verfijning
   (tweetraps: basismodel, dan frontier-model) en schrijft de top-N plus de
   eigen indeling van de gebruiker weg.
4. `ml/validate_headless.js` simuleert elke indeling met N seeds in de
   echte simulator. Dat is de meting die de app toont; de modelvoorspelling
   is alleen de zoekrichting.
5. `result.json` gaat terug naar de browser: looplengte per indeling in
   meters, betrouwbaarheidsinterval, wachttijd, en de tafels terug in
   app-coordinaten.

## Vormgeving

Tactiel maar rustig: gelaagd papier met licht gescheurde randen, karton met
preeg, een strook tape, passe-partout-invoervelden. Palet F7F9F9, E7BB8F,
F1A57E, 88958B, EE5E28. Geen filters op tekst, geen cursief. De scheurranden komen
van SVG-filters (`#torn`, `#torn2`, `#torn3` in `index.html`) op de
achtergrondlaag van elk vel, zodat de tekst zelf scherp blijft. Texturen
zijn feTurbulence-SVG's als data-URI in `styles.css`; er worden geen
afbeeldingen geladen. Lettertypes komen van Google Fonts (Inter voor tekst, Courier Prime voor
getallen en invoer) met systeemterugval.

## Grenzen van deze versie

- De generatie kent alleen de drie vaste tafeltypes (klein, middel, groot).
  Eigen tafeltypes (20 slots) worden bewaard en getekend, maar niet
  gegenereerd, omdat de simulator ze nog niet kent.
- De bar staat altijd tegen een wand; een eilandbar zit niet in de
  trainingsdata.
- Een zaal is een vereniging van rechthoeken (L-vormen, uitsparingen).
  Schuine wanden en ronde vormen niet.
- Het model is getraind op zes zaalarchetypen. Een eigen zaal die daar ver
  van afligt geeft een minder betrouwbare zoekrichting; daarom wordt elk
  voorstel altijd met de simulator nagemeten.
