# Zaalvormen: waar moet de simulator op variëren?

**Aanleiding:** alle 38.280 verzamelde runs draaien op één zaal — 640×640 px, bar rechts, buffet
links, ingang linksonder. Het model is daarmee goed geworden in *deze* zaal en weten we niets over
generalisatie. Dit stuk verzamelt hoe echte zalen eruitzien, om te bepalen waarop we moeten
variëren.

---

## Schaalijking

Voordat praktijkcijfers bruikbaar zijn moet de schaal vastliggen. Een `medium` tafel is 80 px
breed en zit vier personen; een echte vierpersoonstafel is ongeveer 1,20 m. Dus:

> **≈ 67 px per meter.** De huidige zaal van 640×640 px is dan **9,6 × 9,6 m = 92 m² (992 sq ft)**.

Met 36 zitplaatsen komt dat op 28 sq ft per gast. De branchenorm ligt op 10–15 sq ft voor banket,
12 bij ronde tafels en tot 22 voor bruiloften. Onze zaal is dus ruim — logisch, want bar én
buffetlijn zitten in diezelfde oppervlakte, en die tellen in die normen niet mee.

Praktisch betekent dit dat we ruimte hebben om **kleinere** zalen te simuleren zonder onrealistisch
te worden: 36 gasten passen volgens de norm al in 33–50 m², oftewel ongeveer 480×480 px.

## Wat maakt een zaal in de praktijk lastig?

De bronnen zijn het opvallend eens over wat een indeling in de weg zit, en het is bijna nooit de
oppervlakte:

- **Kolommen** worden het vaakst genoemd. Ze staan midden in de zaal, dragen het dak en verplaatsen
  zich niet. Samen met balken, nooduitgangen, donkere hoeken en stopcontacten vormen ze de
  "architecturale verrassingen" die een uitgewerkte plattegrond onderuit halen.
- **Onregelmatige vormen** verlagen de capaciteit meetbaar: minder gasten passen comfortabel in
  dezelfde oppervlakte. De aanbevolen rekenwijze is de zaal opdelen in rechthoeken en die optellen —
  wat precies de manier is waarop wij een zaal kunnen samenstellen.
- **Ronde tafels** worden aangeraden juist bíj kolommen en alkoven, omdat ze zich makkelijker om een
  obstakel heen laten schikken dan rechthoekige.
- **Zonering** is standaardpraktijk: ontvangst, eten, lounge, dansvloer, entertainment. Elke zone is
  in feite een gebied waar geen tafels staan.
- Op de plattegrond horen verplicht: **toegangen, nooduitgangen, toiletten, keukentoegang en
  kolommen**.

## Wat dat betekent voor de simulator

De huidige zaal heeft precies één van deze eigenschappen: een vaste bar en een vaste buffetlijn.
Geen kolommen, geen onregelmatige vorm, één ingang, één afmeting. Zes archetypen dekken samen wat
de bronnen beschrijven:

| Archetype | Wat het toevoegt | Waarom het ertoe doet |
|---|---|---|
| **Rechthoek** | de huidige zaal | referentie; houdt vergelijkbaarheid met alles wat er ligt |
| **Langwerpig** | zijverhouding 1,6–2,2 | looproutes worden lang en lineair in plaats van radiaal |
| **Met kolommen** | 2–6 dragende kolommen | het meest genoemde praktijkprobleem; dwingt omwegen midden in de zaal |
| **L-vorm** | hoek eruit gesneden | twee deelruimtes met een knik ertussen; zichtlijn en looplijn lopen niet gelijk |
| **Met nis** | alkoof aan één zijde | een uithoek die ver van de bar ligt — precies waar bediening duur wordt |
| **Met podium** | geblokkeerde strook langs een wand | zonering; neemt vloer weg zonder de omtrek te veranderen |

Daarnaast moet variëren wat nu vastligt: **afmeting** van de zaal, aan **welke wand** de bar staat,
of er een buffet is en waar, en waar de **ingang** zit. Dat zijn geen vormen maar wel precies de
punten waar alle looproutes op eindigen.

## Wat we bewust niet doen

- **Ronde tafels.** De bronnen raden ze aan bij kolommen, maar de simulator kent alleen rechthoekige
  tafels met stoelankers. Ronde tafels toevoegen raakt de collisiedetectie, de stoelberekening en de
  featureset; dat is een eigen project.
- **Meerdere verdiepingen of niveaus.** Buiten bereik van een 2D-loopgrid.
- **Nooduitgangen als aparte deuren.** Wel genoemd in de bronnen, maar gasten gebruiken ze niet
  tijdens een feest; ze zouden alleen de vloer opdelen zonder dat er verkeer overheen gaat.

---

## Bronnen

- [Banquet Hall Floor Plan: Smart Layout Tips](https://www.coohom.com/article/banquet-hall-floor-plan-essential-guide-for-remarkable-events) — verplichte elementen op een plattegrond: toegangen, nooduitgangen, toiletten, keukentoegang, kolommen
- [Banquet Hall Dimension Standards](https://www.coohom.com/article/industry-standards-for-banquet-wedding-and-event-hall-dimensions) — 10–15 sq ft per gast, tot 22 bij bruiloften
- [Banquet Hall Size Standards](https://www.coohom.com/article/banquet-hall-size-standards-across-different-event-industries) — 12 sq ft per gast bij ronde tafels
- [Event Capacity Calculator](https://www.socialtables.com/blog/event-planning/capacity-party-space-calculator/) — onregelmatige zalen opdelen in rechthoeken; capaciteitsverlies door vorm
- [An Event Planner's Guide to Engaging Room Layouts](https://meetings.skift.com/2024/08/01/engaging-room-layouts/) — kolommen, balken, nooduitgangen en hoeken als "architecturale verrassingen"
- [How to Design a Floor Plan for an Event](https://tripleseat.com/blog/how-to-design-a-floor-plan-for-an-event-and-transform-your-venues-space-for-success/) — ronde tafels bij kolommen en alkoven
- [Event Layouts (RoomSketcher)](https://www.roomsketcher.com/floor-plan-gallery/supplier/event-floor-plan-examples/) — voorbeeldplattegronden
- [Event Space Calculator](https://www.reventals.com/blog/how-much-event-space-do-you-need/) — 6 sq ft per staande gast bij een receptie
