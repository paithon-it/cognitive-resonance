# Risonanza cognitiva: protocollo

Versione 1.1, 10 novembre 2026. Scritto e registrato nel repository **prima di ogni corsa**:
ipotesi, stimatori e criteri stanno qui sotto, e ciò che cambierà dopo va nella sezione
«Deviazioni» in fondo, con la data e il motivo. Lingua originale: italiano
(`PROTOCOL.md` è la traduzione).

## 1. Che cosa si vuole sapere

È il seguito dell'esperimento `angle-between-errors` di
[turing-meter](https://github.com/paithon-it/turing-meter). Lì l'angolo `phi` fra gli
errori di una persona e di una macchina, misurato su quattro raccolte pubbliche, stava
in mediana fra 59 e 81 gradi; l'ideale della coppia era fermato a pesi non negativi, e
la regione oltre i 90 gradi non è stata esplorata. La formula dei turing apparenti,

    Tu_app = sqrt(1 + k^2 - 2 k cos(phi)) / sin(phi),      k = dM / dH,

cresce senza limite quando `phi` va verso i 180 gradi: errori in controfase si
cancellano. Chiamiamo **risonanza cognitiva** questa condizione, e le domande sono cinque.

- **D1.** La promessa della formula oltre i 90 gradi si realizza su reti vere, con
  immagini vere, o resta nel modello gaussiano?
- **D2.** Al variare del peso dato alla macchina, il turing ha la forma e la larghezza
  di una curva di risonanza?
- **D3.** La controfase si può costruire addestrando una macchina sugli errori di una
  mente («rifasamento»)? E la macchina migliore per la coppia è la migliore da sola?
- **D4.** Con persone vere l'angolo si apre?
- **D5.** Dove stanno, su questa scala, le coppie già misurate?

## 2. Il modello

Due giudici decidono fra due classi. Ciascuno ha un indizio graduato: `x_H = s dH/2 + n_H`,
`x_M = s dM/2 + n_M`, con `s = +1` o `-1` secondo la classe, rumori a varianza uno e
correlazione `rho = cos(phi)`. La coppia somma gli indizi con un peso `w >= 0` sulla
macchina, `x_H + w x_M`. In unità della persona:

    Tu(w)^2 = (1 + w k)^2 / (1 + w^2 + 2 rho w)

Il massimo è `Tu_app`, al peso `w* = (k - rho) / (1 - rho k)`.

**Identità di risonanza.** Per `k = 1`, con `Omega = sqrt(w)`:

    Tu(w)^2 - 1 = tan^2(phi/2) / (1 + Q^2 (Omega - 1/Omega)^2),     Q = 1 / (2 cos(phi/2))

È la curva di risonanza di un circuito RLC serie, con `Omega` al posto della frequenza
normalizzata. Al picco `Tu = 2 Q = 1 / cos(phi/2)`. Chiamiamo `Q` **fattore di merito**
della coppia, e lo definiamo così per ogni `k`. La **banda** è l'insieme dei pesi in cui
`Tu^2 - 1` resta sopra metà del suo massimo; per `k = 1` i bordi sono
`Omega - 1/Omega = +-1/Q`, e la larghezza si misura come rapporto `w_alto / w_basso`.

Altre grandezze, con i nomi di turing-meter: `k = dM/dH`; resa sulla macchina `dS/dM`;
angolo utile `arccos(min(k, 1/k))`.

## 3. Stimatori, fissati prima

Valgono per giudici di cui si ha l'indizio intero (reti); per le persone vere si usa lo
strumento di turing-meter (§ 7).

- **Indizio.** `z` = log-odds verso la classe 1. Ogni giudice si standardizza dividendo
  per la deviazione standard entro classe, stimata sulla **validazione**.
- **Sensibilità.** Primaria: `d_a = sqrt(2) z(AUC)`, senza soglia. Secondaria, come in
  turing-meter: convenzione sì/no, `z(hit) + z(rifiuti corretti)`, a soglia zero.
- **Correlazione.** Primaria: correlazione di Pearson degli indizi standardizzati, entro
  classe vera, messa insieme sulle due classi (residui dalla media di classe).
  Ponte con turing-meter: correlazione tetracorica delle risposte giuste entro classe.
  Si riportano tutte e due, e la loro distanza è un risultato. `phi = arccos(rho)`.
- **Turing al peso w.** `Tu(w) = d_a(z_H + w z_M) / d_a(z_H)`, su una griglia di 49 pesi
  a passo costante in logaritmo fra 1/64 e 64.
- **Turing sintonizzato.** Il peso si sceglie su metà delle immagini e il turing si
  legge sull'altra metà, poi si scambiano (media dei due). Il massimo sulla stessa
  griglia e sugli stessi dati è ottimista, e non si usa come misura.
- **Previsione.** `Tu_pred(w)` del § 2 con `k` e `rho` misurati: nessun parametro libero.
- **Banda misurata.** Sui pesi della griglia, per interpolazione lineare in `ln w`.
- **Incertezza.** Bootstrap sulle immagini (tutte le presentazioni della stessa immagine
  restano insieme), 1.000 ricampionamenti, intervalli percentili al 95 per cento. In
  ogni ricampionamento il peso della sintonia si sceglie di nuovo, su una griglia più
  rada (13 pesi); se il tempo di calcolo non basta resta quello scelto sui dati interi,
  e lo si dichiara nei risultati.

## 4. E0. Lo strumento sul modello

Dati generati dal modello del § 2, con la numerosità del banco (2.000 immagini per 5
presentazioni), `k` in {0,7; 1; 1,5} e `rho` in {0,6; 0,3; 0; -0,3; -0,6; -0,8; -0,9;
-0,95}; 200 esperimenti per situazione per l'errore sistematico. La copertura costa un
ricampionamento intero per esperimento, e si verifica su un insieme ridotto: `k = 1`,
`rho` in {0; -0,6; -0,9}, 100 esperimenti, 300 ricampionamenti.

- **Criterio E0a.** Errore sistematico del turing sintonizzato entro il 5 per cento del
  vero per `rho >= -0,9`.
- **Criterio E0b.** Sull'insieme ridotto, l'intervallo al 95 per cento contiene il vero
  fra 90 e 98 volte su cento.
- Per `rho = -0,95` si riporta di quanto lo strumento sbaglia, senza criterio: lì la
  coppia sbaglia così poco che la sensibilità si stima male.

## 5. E1. Il banco: due pregiudizi opposti

Reti piccole, immagini vere, e una manopola che regola l'angolo.

- **Compiti.** Cinque coppie di classi di CIFAR-10 che si confondono: gatto/cane,
  cervo/cavallo, automobile/camion, aereo/nave, uccello/rana. Ogni coppia è un compito
  binario: 10.000 immagini di addestramento (1.000 tenute da parte per la validazione),
  2.000 di prova.
- **Rete.** La rete convoluzionale piccola di `gpu/silicon.py` di turing-meter, con
  uscita binaria. Epoche e passo si fissano su un pilota a `mu = 0`, sulla validazione,
  e poi non cambiano.
- **L'indizio spurio.** Ogni immagine porta una cornice di due pixel di luminosità
  proporzionale a un numero `a`. In addestramento `a` dipende dalla classe:
  `a ~ N(+mu, 1)` per una classe e `N(-mu, 1)` per l'altra. Le reti «più» lo vedono
  in un verso, le reti «meno» nel verso opposto. In prova e in validazione `a` non
  dipende dalla classe: `a ~ N(0, 1 + mu^2)`, la stessa varianza vista in addestramento.
  Il pilota verifica che la cornice si legga (una sonda lineare sull'immagine ritrova
  `a` con `R^2 > 0,9`).
- **La manopola.** `mu` in {0; 0,25; 0,5; 0,75; 1; 1,5; 2}.
- **Giudici.** Per ogni compito, `mu` e seme: una rete «più» (la persona), una «meno»
  (la macchina), una seconda «più» con un altro seme (un'altra persona con lo stesso
  pregiudizio). Tre semi. Coppie **opposte** (più, meno) e coppie **uguali** (più, più).
- **Prova.** Ogni immagine di prova si presenta 5 volte con valori diversi di `a`, gli
  stessi per i giudici di una coppia: 10.000 presentazioni per compito.
- **Riferimento.** A `mu = 0` le reti non hanno pregiudizio: la loro sensibilità da
  sole e in coppia è il riferimento assoluto.

Ipotesi e criteri:

- **H1a, la manopola funziona.** Nelle coppie opposte `rho` scende con `mu`
  (correlazione di rango fra `mu` e `rho` non sopra -0,9, in mediana su compiti e semi),
  e ai due valori più alti di `mu` l'intervallo di `phi` sta tutto sopra i 90 gradi in
  almeno 4 compiti su 5.
- **H1b, oltre Pitagora.** Dove l'intervallo di `phi` sta sopra i 90 gradi, il turing
  sintonizzato supera `sqrt(1 + k^2)`, con l'estremo inferiore dell'intervallo sopra, in
  almeno l'80 per cento delle coppie.
- **H1c, la legge.** Nelle stesse coppie il rapporto fra turing sintonizzato e `Tu_app`
  sta in mediana fra 0,9 e 1,1, e l'errore relativo assoluto medio è minore di quello
  della previsione a 90 gradi.
- **H1d, la sintonia.** Nelle stesse coppie il logaritmo della larghezza di banda
  misurata sta in mediana entro il 30 per cento di quello previsto, e la larghezza
  scende con `phi` (correlazione di rango non sopra -0,7).
- **H1e, lo stesso pregiudizio.** Nelle coppie uguali `rho` sale con `mu`, e al `mu` più
  alto il turing sintonizzato mediano non supera 1,1.
- **H1f, niente si crea.** A ogni `mu` la sensibilità della coppia opposta sintonizzata
  non supera di più del 5 per cento quella di due reti senza pregiudizio messe insieme:
  la risonanza restituisce ciò che il pregiudizio copriva, non aggiunge indizi.

## 6. E2. Il rifasamento

Data una mente con un pregiudizio, si addestra la macchina che lo compensa guardando
solo gli indizi di quella mente, senza dirle quale sia il pregiudizio.

- **La persona H.** Rete «più» con `mu = 1`, addestrata su metà A dell'addestramento.
- **Le macchine.** Addestrate sulla metà B, con la cornice che **non** dipende dalla
  classe. `M_0`: addestramento normale (la migliore da sola). `M_lambda`: alla perdita
  si aggiunge `lambda` per la correlazione di Pearson, nel lotto, fra l'indizio di M e
  quello di H verso la classe vera, tolte le medie di classe; H è congelata e guarda
  la stessa presentazione. `lambda` in {0; 0,3; 1; 3; 10}.
- **Controlli.** `H'`: un'altra rete «più» con `mu = 1`, addestrata sulla metà B (una
  persona simile). Due semi, cinque compiti.
- **Misure.** `phi(H, M_lambda)`, `k`, sensibilità di `M_lambda` da sola, turing
  sintonizzato e sensibilità assoluta della coppia, pendenza dell'indizio di M su `a`
  entro classe (il pregiudizio che M ha imparato; quella di H è positiva).

Ipotesi e criteri:

- **H2a.** `phi(H, M_lambda)` cresce con `lambda` (correlazione di rango almeno 0,9, in
  mediana) e per almeno un `lambda` il suo intervallo sta sopra i 90 gradi, in almeno 3
  compiti su 5.
- **H2b, il miglior compagno non è il miglior solista.** Esiste un `lambda > 0` per cui
  `M_lambda` da sola è meno sensibile di `M_0`, e la coppia `(H, M_lambda)` è più
  sensibile della coppia `(H, M_0)`, con intervalli delle due differenze che escludono
  lo zero, in almeno 3 compiti su 5.
- **H2c.** Dove vale H2a, la pendenza dell'indizio di `M_lambda` su `a` ha segno opposto
  a quella di H.

Se H2a non passa si riporta di quanto l'angolo si è mosso.

## 7. E3. Persone vere

- **Dati.** CIFAR-10H (Peterson e colleghi, 2019): circa 51 voti umani per ciascuna
  delle 10.000 immagini di prova di CIFAR-10, dieci classi. Licenza CC BY-NC-SA 4.0: si
  scarica con uno script, non si ridistribuisce.
- **Macchina di base.** Una ResNet-18 per CIFAR addestrata sulle 50.000 immagini di
  addestramento, almeno il 93 per cento sulla prova.
- **Angolo naturale.** `compare_votes` di turing-meter (versione fissata), voti delle
  persone contro verdetto della macchina, `m = 10`.
- **Macchina in controfase.** Le 10.000 immagini si dividono in due metà per classe.
  Su una metà la macchina di base si affina con la perdita normale più `lambda` per la
  correlazione fra il suo indizio verso la classe vera e quello delle persone,
  `logit((h + 0,5) / (n + 1))` con `h` voti giusti su `n`; `lambda` in {0; 1; 3; 10},
  dove 0 è il controllo. Si misura sull'altra metà, poi si scambiano. Due semi.
- **Misure.** Angolo, accuratezza della macchina, turing apparenti, e un combinatore
  (regressione logistica su quota dei voti e probabilità della macchina, in validazione
  incrociata) per il turing del gruppo.

Ipotesi:

- **H3a.** Per qualche `lambda > 0` l'angolo sulla metà non vista è più aperto che con
  `lambda = 0`, con un intervallo della differenza (bootstrap appaiato sulle immagini)
  che esclude lo zero. L'attesa dichiarata è incerta: con cinquemila immagini e una
  persona su venti che sbaglia, è possibile che non si apra.
- **H3b.** L'angolo resta sotto i 90 gradi per ogni `lambda`.

## 8. E4. Le coppie già misurate

Dagli angoli pubblicati in «Intelligenza apparente» si calcola `Q = 1/(2 cos(phi/2))`
e il turing apparente che la stessa coppia avrebbe a parità di sensibilità. È una
rilettura, senza criteri: ChatBench 81 gradi, αNLI 64, HAIID testo 65, HAIID immagini
78, oggetti 59; due persone fra loro 66 e 54; due reti fra loro fra 22 e 43.

## 9. Che cosa conterebbe contro l'idea

- Se H1c non passa, la formula non regge oltre i 90 gradi neppure su un banco costruito
  per darle ragione: la risonanza resta nel modello.
- Se H1d non passa, la curva di sintonia non è quella di un circuito: l'identità vale
  per indizi gaussiani e non per le reti.
- Se H2a non passa, la controfase non si impara dagli errori: si può solo piantare.
- Se H3a non passa, con persone vere non sappiamo costruirla.

## 10. Ordine, tempi, pulizia

Ordine di priorità: E0, E1, E2, E3. Una GPU L4 da 24 GB; tetto di due ore e mezza di
calcolo in tutto. Se il pilota dei tempi dice che E1 supera i 60 minuti si scende a due
semi, poi ai primi tre compiti, e lo si scrive nelle Deviazioni. Sul server si lavora in
una sola cartella, che alla fine si cancella.

## 11. Dati grezzi: il contratto

Gli script della GPU scrivono solo indizi caso per caso; tutte le misure le fa
l'analisi, dopo, a partire da questi file. Un file `.npz` per unità, in `results/raw/`:

- `bench/{compito}_mu{mu}_s{seme}.npz`: `y` (0/1), `image` (indice dell'immagine di
  prova), `a` (cornice mostrata), `z_plus`, `z_minus`, `z_plus2` (log-odds verso la
  classe 1); gli stessi con prefisso `val_` sulla validazione.
- `rephase/{compito}_s{seme}.npz`: `y`, `image`, `a`, `z_H`, `z_H2`, e `z_M` di forma
  (numero di lambda, N) con `lambdas`; più `val_`.
- `people/fold{f}_s{seme}.npz`: `image` (indice in CIFAR-10 test), `y` (0-9), `logits`
  di forma (numero di lambda, N, 10) con `lambdas`; e `people/base_s{seme}.npz` con i
  logit della macchina di base su tutte le 10.000.
- Ogni cartella ha un `manifest.json`: iperparametri, tempi, versione delle librerie,
  accuratezze di controllo.

## Deviazioni

Nessuna. La versione 1.1 è dello stesso giorno della 1 e viene prima di ogni corsa:
precisa i conti di E0 e del bootstrap, che nella 1 non sarebbero stati eseguibili nel
tempo disponibile.
