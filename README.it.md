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

> Stato: **privato, in lavorazione** (novembre 2026). Il protocollo è stato scritto prima
> di ogni corsa ([`PROTOCOL.it.md`](PROTOCOL.it.md), traduzione inglese
> [`PROTOCOL.md`](PROTOCOL.md)); ciò che cambia dopo va nella sua sezione «Deviazioni».

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
| E3 | persone vere: i voti di CIFAR-10H contro una ResNet-18, e versioni affinate verso la controfase | corse su GPU, CIFAR-10H |
| E4 | le coppie già misurate, rilette su questa scala | angoli pubblicati |
| E5 | il nullo, con pesi di ogni segno (aggiunto nella v1.2, prima di ogni analisi) | i file di E1 |

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
uv run python analysis/download_cifar10h.py    # i voti di CIFAR-10H in data/ (fuori da git)
uv run python analysis/people.py --turing-meter PERCORSO_DEL_CLONE_DI_TURING_METER   # E3
uv run python analysis/figure_data.py          # results/figure_data.json
```

E3 importa `compare_votes` da un clone locale di turing-meter e ne scrive il commit in
`results/e3_people.json`. Le analisi usano al massimo tre processi e girano in pochi
minuti ciascuna su quattro core.

## Licenza

- Codice sotto licenza MIT, © 2026 paithon.it.
- Il protocollo (`PROTOCOL*.md`), `docs/` e i risultati derivati (`results/`) sotto
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.it); ciò che deriva da
  CIFAR-10H (`results/e3_people.json`) mantiene la sua licenza,
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.it).
- I dati delle fonti (CIFAR-10, CIFAR-10H) non si ridistribuiscono: gli script li leggono
  dalle fonti, con le licenze dei loro autori.
