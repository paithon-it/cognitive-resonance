# cognitive-resonance

> [English](README.md) · **Italiano**. Il progetto è scritto in inglese: questa è la
> traduzione del README; il protocollo è nato in italiano (`PROTOCOL.it.md`), codice e
> risultati sono in inglese.

**Quando due menti sbagliano in versi opposti, i loro errori si cancellano. Fin dove
arriva la coppia, e si comporta come un circuito risonante?**

È il seguito di [turing-meter](https://github.com/paithon-it/turing-meter) e del suo
primo esperimento, `angle-between-errors`, le misure dell'articolo di paithon.it
[*Intelligenza apparente*](https://paithon.it/intelligenza-apparente/)
(doi:[10.5281/zenodo.24107649](https://doi.org/10.5281/zenodo.24107649)). Lì l'angolo
`phi` fra gli errori di una persona e di una macchina stava fra 59 e 81 gradi, e la
regione oltre i 90 gradi era rimasta inesplorata. Qui ci si va.

> Stato: **versione 2, corse fatte** (19 novembre 2026). Da E0 a E5 le racconta l'articolo
> di paithon.it [*Risonanza cognitiva*](https://paithon.it/risonanza-cognitiva/)
> (doi:[10.5281/zenodo.24140355](https://doi.org/10.5281/zenodo.24140355)). Quattro
> esperimenti nuovi, da E6 a E9, sono stati **registrati prima delle corse**: il protocollo
> 2.1 è stato pubblicato in questo repository prima di ogni addestramento, su cinque semi
> che nessuna corsa precedente ha usato; la versione 2.2 ha aggiunto una lettura di E6
> prima che un suo file esistesse. Dei loro 29 criteri, **25 passano e 4 no** (tabella
> più sotto). Il protocollo sta in [`PROTOCOL.it.md`](PROTOCOL.it.md) (traduzione inglese
> [`PROTOCOL.md`](PROTOCOL.md)); ciò che è cambiato dopo una corsa sta nella sua sezione
> «Deviazioni».

## L'identità

Due giudici decidono fra due classi, ciascuno con un indizio gaussiano; i loro rumori
hanno correlazione `rho = cos(phi)`. La coppia somma gli indizi con un peso `w` sulla
macchina. Nell'unità della persona, con `k = dM / dH`:

    Tu(w)^2 = (1 + w k)^2 / (1 + w^2 + 2 rho w)

Per `k = 1`, con `Omega = sqrt(w)`:

    Tu(w)^2 - 1 = tan^2(phi/2) / (1 + Q^2 (Omega - 1/Omega)^2),      Q = 1 / (2 cos(phi/2))

È la risposta di un circuito RLC serie, con `Omega` al posto della frequenza
normalizzata. Al picco (`w = 1`) la coppia vale `2Q = 1/cos(phi/2)` turing: `sqrt(2)` a
90 gradi, 2 a 120, 3,9 a 150, senza limite quando `phi` va verso 180. La banda, dove
`Tu^2 - 1` resta sopra metà del picco, si stringe quando l'angolo si apre: una risonanza
più acuta. Con pesi di ogni segno `Tu(w)^2` diventa un profilo di Fano, con un nullo a
`w0 = -1/k` dove i due segnali si cancellano e resta solo il rumore. La derivazione, in
una pagina, è in [`docs/identity.md`](docs/identity.md).

## Le grandezze

- `d_a = sqrt(2) z(AUC)`: la sensibilità di un giudice, dal suo indizio intero, senza
  soglia. Come misura secondaria il `d'` sì/no a soglia zero, come in turing-meter.
- `k = dM / dH`, la macchina nell'unità della persona.
- `rho`, `phi = arccos(rho)`: correlazione di Pearson degli indizi standardizzati entro
  classe (primaria), e correlazione tetracorica delle risposte giuste (lo stimatore di
  turing-meter), una accanto all'altra.
- `Tu_app = sqrt(1 + k^2 - 2 k cos phi) / sin phi`: i turing apparenti, il picco della
  coppia; `sqrt(1 + k^2)` a 90 gradi.
- **Turing sintonizzato**: il peso si sceglie su metà delle immagini, il turing si legge
  sull'altra metà, e le due metà si scambiano. È la misura; il massimo sulla griglia e
  sugli stessi dati è ottimista e non si usa.
- `Q = 1/(2 cos(phi/2))`, il fattore di merito della coppia; la banda come
  `w_alto / w_basso`.
- Il nullo `w0`, la sua larghezza (i pesi in cui `|Tu| < 1/2`), e
  `q = (1 - k cos phi) / (k sin phi)`, l'asimmetria di Fano, zero all'angolo utile
  `arccos(1/k)`.

## Gli esperimenti

| | Che cosa | Dati |
|---|---|---|
| E0 | lo strumento su dati del modello: errore sistematico e copertura | simulati |
| E1 | il banco: piccole reti convoluzionali su cinque coppie di classi di CIFAR-10, una cornice la cui luminosità le reti «più» e «meno» imparano in versi opposti, una manopola `mu` che regola l'angolo | corse su GPU |
| E2 | il rifasamento: una macchina addestrata a scorrelarsi da una rete con un pregiudizio, senza sapere quale | corse su GPU |
| E2 bis | il compagno addestrato sull'errore della coppia, e uno che vede solo la cornice (esplorativo, v1.3, dopo E2 ed E3) | corse su GPU |
| E2 ter | il solista che impara la persona: la macchina migliore da sola, affinata anche sull'errore della somma, e la stessa macchina con un ramo che vede solo la cornice (esplorativo, v1.4) | corse su GPU |
| E3 | persone vere: i voti di CIFAR-10H contro una ResNet-18, e versioni affinate verso la controfase | corse su GPU, CIFAR-10H |
| E3 bis | la ricetta della somma con le persone vere: la macchina di base affinata sull'indizio delle persone più i suoi logit (esplorativo, v1.4) | corse su GPU, CIFAR-10H |
| E4 | le coppie già misurate, rilette su questa scala | angoli pubblicati |
| E5 | il nullo, con pesi di ogni segno (aggiunto nella v1.2, prima di ogni analisi) | i file di E1 |
| E6 | la fiducia su misura: quattro compagni, ciascuno costruito per un peso diverso della macchina nella somma. Il peso a cui la coppia rende segue quello per cui il compagno è costruito? (v2.0) | corse su GPU |
| E7 | viste diverse: giudice e macchina vedono metà diverse dell'immagine, con tutto, un quarto o niente in comune. Che cosa impara un compagno addestrato da zero? (v2.0) | corse su GPU |
| E8 | il banco della cartella: il giudice vede l'immagine intera, la macchina tutta, metà o un quarto. Un teorema prevede la coppia con il solista, con il compagno e con un correttore cieco a partire da misure fatte solo sul giudice e sul solista (v2.1) | corse su GPU |
| E9 | la forma del rumore: le reti di E1 provate con una cornice della stessa varianza e di altra forma (due valori, uniforme, code pesanti). Dove la formula smette di valere per le risposte giuste? (v2.1) | corse su GPU |

Che cosa è uscito dai quattro esperimenti nuovi (25 unità per E6, 75 per gli altri; mediane
su cinque compiti e cinque semi):

| | Criteri | In una riga |
|---|---|---|
| E6 | 7 su 7 | la coppia ha il massimo al peso per cui il compagno è costruito (0,54, 1,12 e 2,43 per 0,5, 1 e 2) e arriva alla stessa altezza che con il solista; a pari peso il compagno vince in 25 unità su 25, a peso 8 vince il solista |
| E6, lettura di progetto | 3 su 4 | il compagno batte il solista fino a 0,96, 1,83 e 3,03 per i rami costruiti per 0,5, 1 e 2 (previsti 1, 2 e 4). **Non confermato (G2)**: a 0,285 del peso dovuto il compagno vince in 17 unità su 25, una meno delle 18 richieste, di 0,01 turing |
| E7 | 5 su 6 | dove le viste non hanno niente in comune la coppia con il compagno vale 2,92 contro 2,70 con il solista, in 5 compiti su 5. **Non confermato (V4)**: un compagno addestrato da zero impara un terzo del compito che il giudice non copre, non i 0,7 richiesti |
| E8 | 7 su 7 | quando la macchina vede un quarto dell'immagine la coppia con il compagno vale 2,87 contro 2,49 con il solista, in 5 compiti su 5; il guadagno è 1,02 volte quello previsto dal teorema; il correttore cieco batte il solista in 25 unità su 25 |
| E9 | 3 su 5 | con la cornice gaussiana la formula regge di nuovo su semi nuovi; con una cornice a due valori promette 3,69 dove la coppia dà 2,29, e la coppia non dipende dalla forma. **Non confermati**: le code pesanti portano il rapporto a 0,95, non sotto 0,9 (S2), e la cornice uniforme va a 0,945, appena sotto la sua banda (S4) |

Nove riletture di dati già raccolti, nessuna delle quali è una prova (le ultime cinque sono state aggiunte dopo i risultati di E6-E9):

| | Che cosa | File dei risultati |
|---|---|---|
| l'identità | con la sensibilità come rapporto fra segnale e rumore, la formula della coppia è un'identità algebrica dei momenti campionari: E1 ed E5 misurano quanto sono gaussiani gli indizi, non la formula | `results/identity.json` |
| il picco si sposta | sui file di E2 ter, la coppia a pesi fissati: il compagno non alza il massimo, sposta il peso a cui la coppia lo raggiunge | `results/e6_trust_from_e2ter.json` |
| le tre macchine | il teorema di E8 sui file di E2 bis ed E2 ter, unità per unità | `results/chart_from_e2.json` |
| per chi conviene un compagno | la lettura di progetto di E6 (protocollo, § 12 bis) sull'unico ramo di E2 ter: batte il solista fino a circa due volte il peso per cui è costruito, e non per chi dà al solista un quarto del peso dovuto | `results/e6_design_from_e2ter.json` |
| la spinta del ramo | sui file di E6: il ramo cancella la spinta del giudice senza errore sistematico, e contro la stessa macchina senza il ramo il pareggio cade a 1,02, 2,05, 4,25 e 8,96 per i pesi 0,5, 1, 2 e 4 (previsti 1, 2, 4 e 8); l'anticipo contro il solista di controllo viene dal controllo | `results/e6_push.json` |
| il teorema sulle viste diverse | il teorema di E8 sui file di E7: dove le viste non hanno niente in comune prevede 2,91 per la coppia con il compagno, misurato 2,92, e il compagno batte sia il solista sia un correttore cieco ideale in 25 unità su 25 | `results/chart_from_e7.json` |
| quanti giudizi servono | sui file di E8: una correzione imparata da 20 giudizi con la risposta nota tiene circa il 94 per cento del guadagno, da 100 circa il 99; il pregiudizio che resta è 1 / n | `results/e8_judgments.json` |
| quando la traccia dice qualcosa di vero | il teorema esteso a una traccia informativa a cui il giudice dà troppo peso: solo l'eccesso è un pregiudizio. Sui file di E8 ricampionati, un ramo addestrato sull'errore della somma toglie il 68 per cento della spinta dove l'eccesso è il 68 per cento | `results/e8_informative.json` |
| il teorema sulle persone | sulle 74 coppie di persone e macchine di turing-meter: se metà del rumore di una persona fosse un pregiudizio leggibile, la coppia migliore crescerebbe di circa il 14 per cento (10 su ChatBench). Il pregiudizio delle persone non è misurato | `results/e4_private.json` |

I risultati, con l'esito di ogni criterio: [`results/RESULTS.md`](results/RESULTS.md) (in
inglese).

## Che cosa c'è

- `src/cognitive_resonance/`: la libreria. `model.py` (le formule: `tu`, `tu_app`,
  `w_star`, `q_factor`, i bordi della banda, la forma di Fano e il nullo),
  `estimators.py` (standardizzazione, `d_a` dai ranghi con pesi interi per il bootstrap,
  correlazioni, turing sintonizzato, banda e nullo misurati, bootstrap a grappoli per
  immagine), `raw.py` (i lettori dei file grezzi, che controllano il contratto del
  § 11), `simulate.py` (dati del modello, per E0), `synthetic.py` (file grezzi finti,
  solo per i test e le prove a vuoto).
- `analysis/`: uno script per esperimento; ciascuno legge `results/raw/` e scrive un
  JSON con tutte le misure e gli esiti; `figure_data.py` raccoglie i dati delle figure.
  Per la versione 2: `fixed.py` (la coppia a pesi fissati, con bootstrap appaiato),
  `trust.py`, `views.py`, `chart.py`, `shape.py` (da E6 a E9), `identity.py`, e
  `design.py`, la lettura di progetto di E6 (protocollo, § 12 bis): per chi conviene un compagno, e la
  regola con cui si sceglie una macchina.
- `src/cognitive_resonance/chart.py`: il teorema delle tre macchine (la coppia con il
  solista, con il compagno, e la banda in cui vince un correttore cieco).
- `gpu/`: gli script che addestrano le reti e scrivono gli indizi grezzi.
- `tests/`: le identità, gli stimatori su dati simulati dal modello, i lettori.
- `results/`: i risultati derivati. Gli indizi grezzi (`results/raw/**/*.npz`) non sono
  sotto controllo di versione.

## Come si rifà

```bash
uv sync --extra dev
uv run pytest -q
uv run python analysis/e0_simulation.py        # results/e0_simulation.json
uv run python analysis/e4_reread.py            # results/e4_reread.json
```

Con i file grezzi delle corse su GPU in `results/raw/`:

```bash
uv run python analysis/bench.py                # E1, results/e1_bench.json
uv run python analysis/null.py                 # E5, results/e5_null.json
uv run python analysis/rephase.py              # E2, results/e2_rephase.json
uv run python analysis/rephase_bis.py          # E2 bis, results/e2bis_rephase.json
uv run python analysis/rephase_ter.py          # E2 ter, results/e2ter_rephase.json
uv run python analysis/download_cifar10h.py    # i voti di CIFAR-10H in data/ (fuori da git)
uv run python analysis/people.py --turing-meter PERCORSO_DEL_CLONE_DI_TURING_METER   # E3
uv run python analysis/people_bis.py --turing-meter PERCORSO_DEL_CLONE_DI_TURING_METER   # E3 bis
uv run python analysis/figure_data.py          # results/figure_data.json
uv run python analysis/results_md.py           # results/RESULTS.md
```

Versione 2. Le riletture dei dati vecchi chiedono i file grezzi di E1, E2 bis ed E2 ter:

```bash
uv run python analysis/identity.py             # results/identity.json
uv run python analysis/trust.py --ter          # results/e6_trust_from_e2ter.json
uv run python analysis/chart.py --e2           # results/chart_from_e2.json
uv run python analysis/design.py --ter         # results/e6_design_from_e2ter.json
```

Gli esperimenti nuovi, con i loro file grezzi in `results/raw/{trust,views,chart,shape}`
(li scrive `gpu/run.sh trust|views|chart|shape`):

```bash
uv run python analysis/trust.py                # E6, results/e6_trust.json
uv run python analysis/views.py                # E7, results/e7_views.json
uv run python analysis/chart.py                # E8, results/e8_chart.json
uv run python analysis/shape.py                # E9, results/e9_shape.json
uv run python analysis/design.py               # E6, lettura di progetto (dopo trust.py), results/e6_design.json
uv run python analysis/push.py                 # E6, la spinta del ramo (dopo trust.py), results/e6_push.json
uv run python analysis/chart.py --views        # il teorema sui file di E7, results/chart_from_e7.json
uv run python analysis/judgments.py            # quanti giudizi servono a una correzione, results/e8_judgments.json
uv run python analysis/informative.py          # quando la traccia dice qualcosa di vero, results/e8_informative.json
uv run python analysis/private_people.py --turing-meter PATH_TO_TURING_METER_CLONE   # results/e4_private.json
```

E3 ed E3 bis importano `compare_votes` da un clone locale di turing-meter e ne scrivono il
commit in `results/e3_people.json` e `results/e3bis_people.json`. Le analisi usano al
massimo tre processi e girano in pochi minuti ciascuna su quattro core.

## Come citare

endjneer (2026). *Risonanza cognitiva*. Versione 1. Zenodo. https://doi.org/10.5281/zenodo.24140357

Tutte le versioni: https://doi.org/10.5281/zenodo.24140355

Codice e dati: https://github.com/paithon-it/cognitive-resonance

## Licenza

- Codice sotto licenza MIT, © 2026 paithon.it.
- Il protocollo (`PROTOCOL*.md`), `docs/` e i risultati derivati (`results/`) sotto
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.it); ciò che deriva da
  CIFAR-10H (`results/e3_people.json`, `results/e3bis_people.json`) mantiene la sua licenza,
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.it).
- I dati delle fonti (CIFAR-10, CIFAR-10H) non si ridistribuiscono: gli script li leggono
  dalle fonti, con le licenze dei loro autori.
