# Hoe vaak loopt een gast naar het buffet — en dus hoe vaak moet er een bord weg?

**Onderzoeksvraag:** hoe vaak haalt een gast eten bij het buffet? Dat getal is tegelijk het
aantal vieze borden dat de bediening moet ophalen.

---

## TL;DR

- **Een schoon bord per ronde is een hygiëne-eis, geen etiquette.** De gezondheidsdienst schrijft
  voor dat je bij elke gang door de buffetlijn een schoon bord pakt. Buffetbezoeken en vieze borden
  zijn daarmee **één op één**: wie drie keer gaat, laat drie borden achter.
- **1,3–1,5 buffetbezoeken per gast** over een dinerbuffet, waarvan de eerste ronde vrijwel 100% en
  een tweede ronde 30–40%. Caterers rekenen met dat getal om eten in te kopen.
- **Afruimen is een aparte stroom werk.** Bij 36 zittende gasten zijn dat ~49 borden per avond.
  Een ober draagt er tot **7** bij het afruimen (hand, onderarm, hand) — dus grofweg **8–16 extra
  ritten**, bovenop de ~31 ritten die het drankjes rondbrengen nu kost.
- **Norm voor snelheid:** een tafel afruimen en schoonmaken hoort in 30–45 seconden te gebeuren.
- Het onderscheid dat telt voor de simulatie: **gasten lopen zelf naar het buffet, maar de obers
  ruimen af.** Het buffet voegt dus pas looplengte voor de bediening toe zodra afruimen erin zit.

---

## 1. Hoe vaak gaat een gast naar het buffet?

Uit `gastgedrag-buffetten-en-feesten.md`, hier kort herhaald omdat het de basis van de rekensom is:

| | |
|---|---|
| Passes door de buffetlijn | **1,3–1,5 per gast** |
| Eerste ronde | vrijwel 100% |
| Tweede ronde (seconds) | **30–40%** |
| Piek | 30–45 min na aanvang, 60–70% wil dan tegelijk |

Eén bron noemt een veel lagere 5% voor seconds. Dat lijkt te gelden voor een korte lunch met één
gang; voor een dinerbuffet van meerdere uren houden de cateringbronnen elkaar op 30–40%. Bij een
apart dessert loopt het totaal verder op, omdat dessert een eigen gang en dus een eigen bord is.

## 2. Krijgt elke ronde een schoon bord?

Ja, en dat is het scharnierpunt van deze hele vraag. Het is geen beleefdheidsregel maar
regelgeving: bij elke gang door de lijn hoort een schoon bord, omdat bestek en bord na één keer
eten besmet zijn en je met dat bord de opscheplepels aanraakt.

Voor de simulatie betekent dat:

> **aantal vieze borden = aantal buffetbezoeken**

Dat maakt de rekensom eenvoudig en het maakt afruimen even frequent als buffetbezoek.

## 3. Hoeveel borden plant een cateraar?

- **3 borden per gast** is de vuistregel voor een buffetreceptie met meerdere gangen.
- **1,5 per gast** volstaat als de cateraar tijdens het feest kan afwassen.

Die 3 ligt hoger dan de 1,3–1,5 passes omdat er salade- en dessertgangen bij zitten. Voor een
simulatie die alleen het hoofdbuffet modelleert is 1,3–1,5 het juiste getal; wil je later dessert
als aparte gang toevoegen, dan gaat het richting 2–3.

## 4. Hoe snel en hoeveel tegelijk wordt er afgeruimd?

| Parameter | Waarde | Hardheid |
|---|---|---|
| Tafel afruimen en schoonmaken | **30–45 s** | branchenorm |
| Borden op een dienblad | **4** (max) | praktijk |
| Borden dragend afruimen | **tot 7** | veldwaarneming |

De gepubliceerde bronnen noemen 3 borden met de hand en 4 op een dienblad, maar dat gaat over
*uitserveren*, waar de borden vol zijn en netjes moeten blijven. Bij **afruimen** ligt het hoger:
één bord in de hand, een stapel op de onderarm en nog één in de andere hand. Bram's eigen
veldwaarneming zet het maximum op **7**, en dat is het getal dat de simulator gebruikt — geschaald
naar vakmanschap, want een beginner haalt die stapel niet.

Verder geldt: in een formelere setting wacht je tot iedereen aan tafel klaar is voordat je die gang
afruimt; casual mag per gast. Voor een buffet is per gast realistischer, want gasten gaan op eigen
moment.

---

## Wat dit betekent voor de simulator

De simulator draait op 49 gasten, maar de zaal heeft maar **36 stoelen** (6 medium × 4 + 2 large ×
6). Er zitten dus 36 gasten; de rest komt niet aan tafel. Gemeten over 38.520 runs: **48,5
buffetbezoeken per run**, oftewel **1,35 per zittende gast** — netjes binnen de onderzochte
bandbreedte.

Daaruit volgt het werk dat er nu nog niet is:

| | per run |
|---|---|
| Vieze borden | **~49** |
| Ritten om ze op te halen (4 per rit) | **~12–16** |
| Ter vergelijking: ritten voor drankjes (245 drankjes, 8 per rit) | ~31 |

Afruimen is dus geen detail maar ongeveer **40% extra ritten** voor de bediening. En anders dan
buffetbezoek, dat alleen gástenverkeer is, telt dit rechtstreeks mee in `waiterDist`. Dat is
precies de reden dat het toevoegen van de buffetlijn de looplengte van de obers nauwelijks
veranderde: het halve mechanisme zat erin.

### Voorstel voor de implementatie

1. Elk voltooid buffetbezoek legt een vies bord op de tafel van die gast.
2. Een ober die toch in de buurt is neemt ze mee — of er komt een aparte afruimtaak zodra er
   genoeg borden liggen.
3. Draagcapaciteit **7**, geschaald naar vakmanschap (3 bij een beginner). Dat maakt afruimen tot een ketenprobleem net als
   het rondbrengen van drankjes: meerdere tafels in één rit.
4. Afruimtijd **30–45 s** per tafel.
5. De borden moeten ergens heen. De bar is het enige bestaande afgiftepunt; een aparte spoelhoek
   zou realistischer zijn en verandert de geometrie van het probleem.

Punt 5 is een ontwerpkeuze die het optimalisatieprobleem echt verandert: met één afgiftepunt bij
de bar lopen alle stromen naar rechts, met een aparte spoelhoek ontstaat een tweede bestemming.

---

## Bronnen

- [Why do buffet restaurants make you take a clean plate each time?](https://www.quora.com/Why-do-buffet-restaurants-make-you-take-a-clean-plate-each-time-you-go-through-the-buffet) — hygiëne-eis, handhaving door de gezondheidsdienst
- [Buffet etiquette: new plate every trip?](http://boards.straightdope.com/sdmb/showthread.php?t=592231) — bevestiging dat cateraars dit moeten volgen
- [How many plates will I need?](https://forums.theknot.com/discussion/404619/how-many-plates-will-i-need) — 3× per gast, 1,5× bij afwassen tijdens het feest
- [How Many Plates Do You Need for 50 Guests Buffet](https://www.royalwarechina.com/how-many-plates-do-you-need-for-50-guests-buffet/) — bordenplanning per gastaantal
- [Bussing Tables: A Complete Guide](https://www.call-the-service.com/blog/bussing-tables/) — 30–45 s per tafel afruimen
- [Server Tip: Clearing the Table](https://www.statefoodsafety.com/Resources/Resources/server-tip-clearing-the-table) — wanneer per gast en wanneer per tafel afruimen
- [Plate Handling for Servers](https://www.statefoodsafety.com/Resources/Resources/server-tip-plate-balancing) — 3 borden met de hand, meer gestapeld
- [Ask George: procedures for busing tables](https://www.stlmag.com/dining/ask-george-are-there-established-procedures-for-busing-restaurant-tables/) — dienblad maximaal 4 borden
