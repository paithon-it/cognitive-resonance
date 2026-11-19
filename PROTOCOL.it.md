# Risonanza cognitiva: protocollo

Versione 2.0, 19 novembre 2026. Scritto e registrato nel repository **prima di ogni corsa**:
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
cancellano. Chiamiamo **risonanza cognitiva** questa condizione, e le domande sono otto.

- **D1.** La promessa della formula oltre i 90 gradi si realizza su reti vere, con
  immagini vere, o resta nel modello gaussiano?
- **D2.** Al variare del peso dato alla macchina, il turing ha la forma e la larghezza
  di una curva di risonanza?
- **D3.** La controfase si può costruire addestrando una macchina sugli errori di una
  mente («rifasamento»)? E la macchina migliore per la coppia è la migliore da sola?
- **D4.** Con persone vere l'angolo si apre?
- **D5.** Dove stanno, su questa scala, le coppie già misurate?
- **D6.** Accanto al picco esiste un nullo, un peso a cui la coppia non vale niente? In
  quali condizioni compare? (Aggiunta della versione 1.2: vedi § 8 bis.)
- **D7.** A pari valore della coppia, una macchina costruita sulla persona sposta il peso
  a cui la coppia rende? E quel peso si può decidere addestrandola? (Aggiunta della
  versione 2.0: vedi § 12.)
- **D8.** Se la persona vede qualcosa che la macchina non vede, il compagno costruito
  batte il miglior solista? (Aggiunta della versione 2.0: vedi § 13.)

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

## 6 bis. E2 bis. Il compagno addestrato per la coppia (esplorativo)

Aggiunto con la versione 1.3, **dopo** aver visto i dati di E2 ed E3: non è una prova
registrata prima, e come tale va letta. Le attese qui sotto sono scritte prima della corsa.

**Perché.** In E2 la penalità sulla correlazione è stata soddisfatta nel modo più
economico: la macchina ha capovolto il proprio indizio (`k < 0`). È un'antifase finta,
perché la coppia `(k, rho)` e la coppia `(-k, -rho)` sono la stessa coppia: basta
credere alla macchina al contrario. Si prova allora la ricetta che la scorciatoia non
la ammette: la macchina impara dall'errore della **somma**.

- **H e `M_0`** come in E2: stessi compiti, stessi semi, stesse metà, stessa cornice.
- **`M_coppia`.** Stessa rete di `M_0`, metà B, cornice che non dipende dalla classe.
  La perdita è l'entropia incrociata dell'indizio somma `z_H + z_M`; H è congelata e
  guarda la stessa presentazione.
- **`M_cornice`.** Come `M_coppia`, ma vede solo la cornice: l'interno dell'immagine è
  azzerato (in unità normalizzate). Da sola non può sapere niente della classe: è il
  caso `k = 0`, per cui il modello prevede `Tu = 1 / sin(phi)`.
- **Misure** come in E2, con gli stessi stimatori e lo stesso bootstrap.

Attese:

- **B1.** `M_coppia` e `M_cornice` non scendono sotto il caso (per `M_coppia` l'intervallo
  di `d_a` sta sopra lo zero; per `M_cornice` lo contiene) e la pendenza del loro indizio
  su `a` ha segno opposto a quella di H, in ogni unità.
- **B2.** `phi(H, M_coppia) - phi(H, M_0) > 0`, con intervallo che esclude lo zero in
  ogni seme, in almeno 3 compiti su 5.
- **B3, il miglior compagno non è il miglior solista.** `M_coppia` da sola è meno
  sensibile di `M_0`, e la coppia `(H, M_coppia)` è più sensibile della coppia
  `(H, M_0)`, con intervalli delle due differenze che escludono lo zero in ogni seme,
  in almeno 3 compiti su 5.
- **B4, la macchina che non sa niente.** Il turing sintonizzato della coppia
  `(H, M_cornice)` ha l'intervallo sopra 1 in ogni seme in almeno 3 compiti su 5, e in
  mediana sta entro il 10% da `1 / sin(phi)`.
- **B5.** Per `M_coppia` il turing sintonizzato sta, in mediana, entro il 10% dal turing
  apparente calcolato da `k` e `phi`.

Dati grezzi: `rephase_bis/{compito}_s{seme}.npz` con `y, image, a, z_H, z_M` (una riga
per macchina, nell'ordine di `machines` = `M0, Mpair, Mframe`) e gli stessi con prefisso
`val_`; `manifest.json`.

## 6 ter. E2 ter. Il solista che impara la persona (esplorativo)

Aggiunto con la versione 1.4, l'11 novembre 2026, **dopo** aver visto tutti i risultati e
dopo una revisione della bozza dell'articolo: non è una prova registrata prima. Le attese
qui sotto sono scritte prima della corsa.

**Perché.** In E2 bis la macchina addestrata da zero sull'errore della somma ha imparato
quasi soltanto a togliere il pregiudizio di H, e del compito non sa quasi niente: il
miglior solista è rimasto il miglior compagno (B3 non passa). Resta da provare a partire
dal solista, perché non debba scegliere fra il compito e la correzione.

- **H e `M_0`** come in E2 ed E2 bis: stessi compiti, stessi semi, stesse metà, stessa
  cornice.
- **`M_0+`, il controllo.** `M_0` affinata sulla metà B con la sua perdita normale: 10
  epoche, passo un quinto di quello del pilota, discesa a coseno fino a zero. Serve a
  non scambiare per effetto della ricetta dieci epoche in più.
- **`M_due`.** `M_0` affinata allo stesso modo, con la perdita mista: entropia incrociata
  del suo indizio più entropia incrociata dell'indizio somma `z_H + z_M`; H è congelata
  e guarda la stessa presentazione.
- **`M_ramo`.** `M_0` congelata, più un ramo: una rete nuova che vede solo la cornice
  (come `M_cornice`), addestrata con gli iperparametri del pilota sull'entropia
  incrociata di `z_H + z_M0 + z_ramo`. L'indizio della macchina è `z_M0 + z_ramo`.
- **Misure** come in E2 bis, con gli stessi stimatori e lo stesso bootstrap; in più la
  sensibilità della coppia sintonizzata contro quella di `M_0+` da sola.

Attese:

- **T1, non dimentica.** `d_a(M_due)` è almeno 0,9 volte `d_a(M_0+)`, in mediana sulle
  unità, e la pendenza dell'indizio di `M_due` su `a` ha segno opposto a quella di H in
  ogni unità.
- **T2, l'angolo si apre.** `phi(H, M) - phi(H, M_0+) > 0`, con intervallo che esclude lo
  zero in ogni seme, in almeno 3 compiti su 5; si giudica per `M_due` e per `M_ramo`.
- **T3, il compagno batte il solista.** La coppia sintonizzata `(H, M)` è più sensibile
  della coppia `(H, M_0+)`, con intervallo della differenza che esclude lo zero in ogni
  seme, in almeno 3 compiti su 5; si giudica per `M_due` e per `M_ramo`. Attesa
  dichiarata: passa, di poco. Il tetto è la coppia senza pregiudizio di E1 (`mu = 0`).
- **T4.** Il turing sintonizzato sta, in mediana, entro il 10% dal turing apparente
  calcolato da `k` e `phi`, per tutte e due le macchine.
- **T5, la coppia contro la macchina.** La coppia sintonizzata `(H, M_due)` è più
  sensibile di `M_0+` da sola, con intervallo della differenza che esclude lo zero in
  ogni seme, in almeno 3 compiti su 5. Attesa dichiarata: incerta.

Dati grezzi: `rephase_ter/{compito}_s{seme}.npz`, come `rephase_bis/`, con `machines` =
`M0, M0plus, Mdue, Mramo`; `manifest.json`.

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

## 7 bis. E3 bis. La ricetta della somma con le persone vere (esplorativo)

Aggiunto con la versione 1.4, alle stesse condizioni di E2 ter: dopo aver visto tutti i
risultati, attese scritte prima della corsa.

**Perché.** E3 ha provato sulle persone la ricetta della punizione, quella che sul banco
non ha funzionato. La ricetta della somma è nata dopo, e con le persone non è stata
provata.

- **Dati, macchina di base, metà e affinamento** come in E3: stessi semi, stesse due
  metà, 8 epoche a passo 0,01 con discesa a coseno, a partire dalla macchina di base.
- **L'indizio delle persone**, su dieci classi: `u_c = log((v_c + 0,5) / (n + 5))`, con
  `v_c` i voti per la classe `c` su `n`.
- **Tre ricette.** `controllo`: entropia incrociata dei logit della macchina (è
  `lambda = 0` di E3, rifatto). `somma`: entropia incrociata di `u + z_M`. `mista`: la
  somma delle due.
- **Misure** come in E3 (angolo, accuratezza della macchina, turing apparenti, turing
  del gruppo con il combinatore), sulla metà non usata per l'affinamento. In più
  l'accuratezza di `argmax(u + z_M)`, accanto a quella della maggioranza dei voti.

Attese:

- **P1, non crolla.** Con `mista` l'accuratezza della macchina da sola non scende più di
  2 punti sotto quella del controllo, in ogni metà e in ogni seme. Per `somma` nessuna
  attesa: si riporta.
- **P2, l'angolo si apre.** Per almeno una delle due ricette l'angolo sulla metà non
  vista è più aperto che nel controllo, con intervallo della differenza (bootstrap
  appaiato sulle immagini) che esclude lo zero, la stessa ricetta in tutti i semi.
- **P3, i turing apparenti salgono.** Per la stessa ricetta di P2 i turing apparenti
  superano quelli del controllo, con intervallo della differenza che esclude lo zero in
  tutti i semi. È il punto in cui la punizione ha fallito. Attesa dichiarata: incerta,
  con cinquemila immagini e una persona su venti che sbaglia.
- **P4.** L'angolo resta sotto i 90 gradi per tutte e due le ricette (stima puntuale).
- **P5, il gruppo.** Il turing del gruppo con il combinatore, per `mista`, supera quello
  del controllo con intervallo della differenza che esclude lo zero in tutti i semi.
  Attesa dichiarata: incerta.

Dati grezzi: `people_bis/fold{f}_s{seme}.npz` con `image`, `y`, `logits` di forma
(3, N, 10) e `recipes` = `controllo, somma, mista`; `people_bis/base_s{seme}.npz`;
`manifest.json`.

## 8. E4. Le coppie già misurate

Dagli angoli pubblicati in «Intelligenza apparente» si calcola `Q = 1/(2 cos(phi/2))`
e il turing apparente che la stessa coppia avrebbe a parità di sensibilità. È una
rilettura, senza criteri: ChatBench 81 gradi, αNLI 64, HAIID testo 65, HAIID immagini
78, oggetti 59; due persone fra loro 66 e 54; due reti fra loro fra 22 e 43.

## 8 bis. E5. Il nullo

Aggiunto il 10 novembre 2026 con la versione 1.2, su richiesta dell'autore, a corse
della GPU già avviate e **prima che l'analisi dei dati grezzi fosse scritta o
guardata**. Non chiede corse nuove: usa i file di E1.

**Il modello.** Se il peso può avere ogni segno, il turing con il segno è
`Tu(w) = (1 + w k) / sqrt(1 + w^2 + 2 rho w)`, e il suo quadrato è un profilo di Fano:

    Tu(w)^2 = k^2 (q + eps)^2 / (1 + eps^2)
    eps = (w + cos(phi)) / sin(phi),      q = (1 - k cos(phi)) / (k sin(phi))

- **Il nullo** sta a `w0 = -1/k` (`eps = -q`), per ogni angolo: i due segnali si
  cancellano e resta il rumore. Oltre il nullo il turing è negativo: la coppia sbaglia
  in modo sistematico.
- **Il picco** sta a `eps = 1/q` e vale `Tu_app`.
- `q = 0` all'angolo utile, `cos(phi) = 1/k`: il picco sparisce e resta solo il nullo.
- **La larghezza del nullo** è l'intervallo di pesi in cui `|Tu| < 1/2`. Per `k = 1`
  cresce con l'angolo, al contrario della banda del picco.
- **Due condizioni.** Con `k > 0` (tutti e due sopra il caso) il nullo chiede un peso
  negativo: fidarsi al contrario. Con `k < 0` (uno dei due sotto il caso, cioè in
  errore sistematico) compare a un peso positivo, `w0 = 1/|k|`.

**Stimatori.** Sensibilità con il segno: `d_a = sqrt(2) z(AUC)`, negativa quando
l'AUC è sotto 0,5. Griglia estesa: i 49 pesi del § 3 anche con il segno meno. Nullo
misurato: il peso in cui il turing con il segno attraversa lo zero, per interpolazione
lineare fra i due punti di griglia vicini. Larghezza misurata: l'intervallo in cui
`|Tu| < 1/2`, per interpolazione. Le previsioni usano `k` e `rho` misurati.

Ipotesi e criteri:

- **H5a, il nullo c'è ed è dove deve.** In tutte le coppie del banco, opposte e
  uguali, il turing con il segno attraversa lo zero a un peso negativo, e il prodotto
  `w0 k` sta in mediana fra -1,1 e -0,9.
- **H5b, è profondo.** Al peso previsto `-1/k`, calcolato su metà delle immagini, il
  valore assoluto del turing misurato sull'altra metà sta in mediana sotto 0,1.
- **H5c, la larghezza segue l'angolo, al contrario del picco.** Il logaritmo della
  larghezza misurata sta in mediana entro il 30 per cento di quello previsto, e la
  larghezza cresce con `phi` (correlazione di rango almeno 0,7 su tutte le coppie del
  banco).
- **H5d, il nullo a fiducia positiva.** Nelle coppie opposte con `mu >= 1` si prendono
  le presentazioni in cui la cornice punta contro la verità per la macchina (la rete
  «meno») oltre una deviazione standard di `a`; il verso si ricava dal segno della
  pendenza dell'indizio della macchina su `a`, non si assume. Gli indizi si
  standardizzano sulla validazione con la stessa selezione. Dove la sensibilità della
  macchina è negativa, con un intervallo che esclude lo zero, il turing con il segno
  attraversa lo zero a un peso positivo, e il rapporto fra il peso misurato e quello
  previsto, `1/|k|`, sta in mediana fra 0,8 e 1,25.
- **H5e, tutta la curva.** Nelle coppie del banco l'errore assoluto medio fra il turing
  con il segno misurato e quello previsto, sulla griglia estesa, diviso per `Tu_app`,
  sta in mediana sotto il 10 per cento.

## 9. Che cosa conterebbe contro l'idea

- Se H1c non passa, la formula non regge oltre i 90 gradi neppure su un banco costruito
  per darle ragione: la risonanza resta nel modello.
- Se H1d non passa, la curva di sintonia non è quella di un circuito: l'identità vale
  per indizi gaussiani e non per le reti.
- Se H2a non passa, la controfase non si impara dagli errori: si può solo piantare.
- Se H3a non passa, con persone vere non sappiamo costruirla.
- Se H5a o H5b non passano, il nullo è una proprietà del modello e non delle reti.
- Se F1 non passa, il vantaggio del compagno a pari peso visto in E2 ter era un caso di
  quelle dieci unità.
- Se F5 non passa, il peso a cui la coppia rende non si decide addestrando la macchina:
  resta una scelta di chi la usa.
- Se V1 non passa, il compagno costruito non batte il solista nemmeno quando la persona
  sa qualcosa di suo: come obiettivo di progetto, a fiducia libera, la controfase non
  serve.
- Se V4 non passa, è sbagliata la lettura per cui la macchina addestrata sulla somma
  impara ciò che alla persona manca.

## 10. Ordine, tempi, pulizia

Ordine di priorità: E0, E1, E2, E3. Una GPU L4 da 24 GB; tetto di due ore e mezza di
calcolo in tutto. Se il pilota dei tempi dice che E1 supera i 60 minuti si scende a due
semi, poi ai primi tre compiti, e lo si scrive nelle Deviazioni. Sul server si lavora in
una sola cartella, che alla fine si cancella.

E6 ed E7 (versione 2.0): stima di 35-45 minuti sulla stessa GPU, in una sottocartella
propria che alla fine si cancella. Se il tempo non basta si scende a due semi (2 e 3) e
lo si scrive nelle Deviazioni.

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
- `trust/{compito}_s{seme}.npz` (E6): come `rephase_ter/`, con sette macchine e
  `built_for_weight`, i pesi per cui sono costruiti i quattro rami.
- `views/{compito}_o{sovrapposizione}_s{seme}.npz` (E7): come `rephase_bis/`, con
  `machines` = `M0, Mramo, Mpair` e `overlap` (100, 50 o 0).
- Ogni cartella ha un `manifest.json`: iperparametri, tempi, versione delle librerie,
  accuratezze di controllo.

## 12. E6. La fiducia su misura

Aggiunto con la versione 2.0, il 19 novembre 2026, dopo la pubblicazione dell'articolo e
una sua revisione.

**Da dove viene.** In E2 ter il solista, il solista ritoccato e il solista con il ramo
davano alla coppia sintonizzata lo stesso valore, e T3 non è passato. Il 19 novembre gli
stessi file sono stati riletti a un peso fissato invece che al peso migliore. Le tre
macchine differiscono nel peso a cui la coppia arriva al massimo: circa 7 con il
solista, circa 1,2 con il ramo. A peso 1 la coppia con il ramo vale di più in 10 unità
su 10 (in mediana 1,47 turing contro 1,38); a peso 8 vale di meno (1,28 contro 1,47). E
moltiplicando per un fattore `c` la parte dell'indizio che viene dal ramo, il massimo si
sposta a un peso vicino a `1 / c`. Quella rilettura sta in
`results/e6_trust_from_e2ter.json` (`analysis/trust.py --ter`). **È esplorativa**: le
ipotesi qui sotto nascono da lì, e per questo non può confermarle. La prova si fa su
addestramenti nuovi, con semi mai usati, e con rami *addestrati* per pesi diversi, non
riscalati dopo.

**L'idea.** Se la persona ha nell'indizio un pregiudizio `b a` e la macchina vi aggiunge
`-c a`, la somma `z_H + w z_M` lo cancella al peso `w = b / c`. Un ramo addestrato sulla
somma con peso `w0` dovrebbe quindi imparare `c = b / w0`, e la coppia dovrebbe avere il
massimo vicino a `w0`. Il peso a cui una coppia rende, che nel § 2 è una scelta di chi
usa la macchina, diventerebbe una proprietà della macchina.

- **H, `M_0`, `M_0+`, `M_due`** come in E2 ter, con i semi 2, 3 e 4. E1-E5 hanno usato 0 e 1.
- **`M_ramo(w)`**, per `w` = 0,5, 1, 2 e 4. `M_0` congelata più un ramo che vede solo la
  cornice, addestrato con gli iperparametri del pilota sull'entropia incrociata di
  `z_H + w (z_M0 + z_ramo)`, con i log-odds così come escono dalle reti. I quattro rami
  hanno lo stesso seme, lo stesso ordine dei lotti, gli stessi ritagli e le stesse
  cornici: cambia solo `w`. L'indizio della macchina è `z_M0 + z_ramo`. `M_ramo(1)` è
  `M_ramo` di E2 ter.
- **La coppia a un peso fissato** (`analysis/fixed.py`). Su due scale. Standardizzata,
  quella del § 3: `d_a(z_H + w z_M)` ai pesi 0,5, 1, 2 e 8. E «così com'è»: la somma dei
  log-odds grezzi, `z_H + z_M`, cioè le due opinioni prese per quello che dicono.
- **Il peso del picco.** Sulla curva `Tu(w) = d_a(z_H + w z_M) / d_a(z_H)` con i log-odds
  grezzi, sui 49 pesi del § 3: la media geometrica dei due pesi a cui `Tu^2 - 1`
  attraversa il 90% del suo massimo, per interpolazione lineare in `ln w`. Il massimo di
  una curva piatta è un punto rumoroso, il centro no. Dove un attraversamento cade fuori
  dalla griglia vale il peso del massimo.
- **Differenze** appaiate sugli stessi ricampionamenti delle immagini (§ 3), 1.000
  ricampionamenti. Il controllo è `M_0+`. `M_due` si riporta senza criterio.

Ipotesi. Le primarie sono F1 e F5.

- **F1, a pari peso vince il compagno.** A peso standardizzato 1 la coppia `(H, M_ramo(1))`
  è più sensibile della coppia `(H, M_0+)`, con intervallo della differenza sopra lo zero
  in ogni seme, in almeno 3 compiti su 5.
- **F2, anche così com'è.** Lo stesso con la somma dei log-odds grezzi.
- **F3, il prezzo.** A peso standardizzato 8 la coppia `(H, M_0+)` è più sensibile della
  coppia `(H, M_ramo(1))`, con intervallo sotto lo zero in ogni seme, in almeno 3 compiti
  su 5: chi si fida molto della macchina sta meglio con il solista.
- **F4, chi deve farsi da parte.** Così com'è, la coppia `(H, M_0+)` è meno sensibile di
  `M_0+` da sola (intervallo sotto lo zero in ogni seme, in almeno 3 compiti su 5), e la
  coppia `(H, M_ramo(1))` no: la mediana sulle unità della sua differenza da `M_0+` da
  sola è almeno -0,05.
- **F5, la manopola sta nella macchina.** Il peso del picco della coppia `(H, M_ramo(w))`
  sta, in mediana sulle unità, entro un fattore 1,5 da `w` per `w` = 0,5, 1 e 2; e la
  correlazione per ranghi fra `w` e il peso del picco, sui quattro rami, è in mediana
  sulle unità almeno 0,9. A `w = 4` il posto del picco non si giudica: lì la curva è
  quasi piatta, come con il solista.
- **F6, lo stesso picco.** Il turing sintonizzato di `(H, M_ramo(w))` diviso per quello di
  `(H, M_0+)` sta, in mediana sulle unità, fra 0,97 e 1,03 per ciascuno dei quattro pesi.
- **F7, quanto costa da sola.** Più basso è il peso per cui è costruita, meno la macchina
  vale da sola: la correlazione per ranghi fra `w` e `d_a(M_ramo(w))` è in mediana almeno
  0,9, e `M_ramo(0,5)` da sola è meno sensibile di `M_0+`, con intervallo sotto lo zero in
  ogni seme, in almeno 3 compiti su 5.

Dati grezzi: `trust/{compito}_s{seme}.npz`, come `rephase_ter/`, con `machines` =
`M0, M0plus, Mdue, Mramo_w0.5, Mramo_w1, Mramo_w2, Mramo_w4` e `built_for_weight`;
`manifest.json`.

## 13. E7. Viste diverse

Aggiunto con la versione 2.0, il 19 novembre 2026. Di questa prova non esiste nessun
dato: le attese vengono dal modello e da un banco giocattolo
(`synthetic.write_views`), su cui l'analisi è stata provata.

**Perché.** In E1-E6 la macchina vede tutto ciò che vede la persona. Allora ogni modo di
mettere insieme i due indizi è una funzione di ciò che la macchina ha davanti, e la
macchina migliore possibile, da sola, farebbe almeno altrettanto. Che su quel banco il
miglior solista sia rimasto il miglior compagno (B3 e T3 non passano) è quindi una
conseguenza del disegno, non una scoperta. Perché togliere un pregiudizio liberi
qualcosa, la persona deve sapere qualcosa che la macchina non vede.

- **Le viste.** Ogni giudice vede metà dell'interno dell'immagine e tutta la cornice; il
  resto è azzerato (unità normalizzate), dopo che la cornice è stata disegnata. H vede
  sempre la metà sinistra (colonne 2-15). La macchina vede la metà sinistra
  (sovrapposizione 100%), la metà alta (righe 2-15: ha in comune con H il quarto in alto
  a sinistra, 50%) o la metà destra (colonne 16-29, 0%). Ogni metà passa per il centro
  dell'immagine: una striscia centrale varrebbe più di una laterale, e la sovrapposizione
  si confonderebbe con la bravura.
- **H**: rete «più», `mu = 1`, addestrata sulla metà A delle immagini di addestramento.
  È la stessa rete alle tre sovrapposizioni.
- **`M_0`**, il solista: addestrata sulla metà B sulla sua vista, cornice indipendente
  dalla classe.
- **`M_ramo`**: `M_0` congelata più un ramo che vede solo la cornice, addestrato
  sull'entropia incrociata di `z_H + z_M0 + z_ramo`.
- **`M_coppia`**: una rete nuova sulla vista della macchina, addestrata da zero
  sull'entropia incrociata di `z_H + z_M` (la ricetta di E2 bis), con lo stesso seme di
  `M_0`.
- Semi 2, 3 e 4; `M_0`, il ramo e `M_coppia` hanno gli stessi semi alle tre
  sovrapposizioni. Cornici di prova uguali alle tre sovrapposizioni.
- **Misure** del § 3 e di E2 bis, più la coppia così com'è (§ 12). Le differenze sono
  rispetto a `M_0` della stessa sovrapposizione, appaiate.

Ipotesi. Le primarie sono V1 e V4.

- **V1, con qualcosa di suo il compagno vince.** A sovrapposizione 0 la coppia
  sintonizzata `(H, M_ramo)` è più sensibile della coppia sintonizzata `(H, M_0)`, con
  intervallo della differenza sopra lo zero in ogni seme, in almeno 3 compiti su 5.
- **V2, la dose.** La mediana sulle unità di quella differenza cresce al calare della
  sovrapposizione (0% > 50% > 100%), e a 100% è al più un terzo di quella a 0%.
- **V3, meno brava da sola, migliore in coppia.** A sovrapposizione 0 `M_ramo` da sola è
  meno sensibile di `M_0` da sola (intervallo sotto lo zero in ogni seme, in almeno 3
  compiti su 5), e V1 passa.
- **V4, la macchina addestrata sulla somma impara ciò che alla persona manca.**
  `d_a(M_coppia) / d_a(M_0)`, in mediana sulle unità, è almeno 0,7 a sovrapposizione 0, al
  più 0,3 a 100% (in E2 bis, a vista intera, era 0,03), e decresce con la sovrapposizione.
- **V5, la coppia batte la macchina.** A sovrapposizione 0 la coppia sintonizzata
  `(H, M_ramo)` è più sensibile di `M_0` da sola, con intervallo sopra lo zero in ogni
  seme, in almeno 3 compiti su 5.
- **V6.** A sovrapposizione 0, così com'è, la coppia `(H, M_ramo)` è più sensibile della
  coppia `(H, M_0)`: intervallo sopra lo zero in ogni seme, in almeno 3 compiti su 5.

Un limite scritto prima: a 50% la metà della macchina è orizzontale e quella di H
verticale, quindi fra 0%, 50% e 100% non cambia soltanto la sovrapposizione.

Dati grezzi: `views/{compito}_o{sovrapposizione}_s{seme}.npz`, come `rephase_bis/`, con
`machines` = `M0, Mramo, Mpair` e `overlap`; `manifest.json`.

## Deviazioni

Tutte del 10 novembre 2026. Le prime vengono dalle corse; le altre sono scelte
dell'analisi dove il protocollo lasciava un margine, fissate prima di leggere i dati veri.

**Le corse.**

- **E1 con due semi** (0 e 1) invece di tre. Con 30 epoche una rete costa circa 12,5
  secondi e E1 con tre semi era stimato in circa 70 minuti, oltre il tetto di 60: è la
  prima riduzione del § 10. Con due semi la stima era di circa 47 minuti.
- **Il pilota** ha fissato 30 epoche, passo 0,1, lotto 256 e `delta = 1`; la sonda lineare
  ritrova `a` con `R^2 = 1,00`.
- **La cornice** è *impostata* al valore `a · delta` sul bordo di 2 pixel dei tre canali,
  in unità normalizzate, dopo ritaglio e ribaltamento: non è sommata all'immagine.
- **I semi.** Ogni giudice ha il suo seme, lo stesso a ogni `mu`; i numeri `eps` della
  cornice di validazione e di prova sono gli stessi per i tre giudici di un'unità e per
  ogni `mu`.
- **E3** senza pilota: iperparametri fissati prima, 40 epoche a passo 0,2 per la macchina
  di base, 8 epoche a passo 0,01 con discesa a coseno fino a zero per l'affinamento.
- **I manifest** scritti dalle corse (anche quello di `rephase/`) citano la versione 1.1:
  è la versione scritta negli script della GPU. Le versioni 1.2 e 1.3 non cambiano niente
  di ciò che E1, E2 ed E3 fanno.

**L'analisi.**

- **E0.** Il protocollo non fissa `dH`: si è usato `dH = 1`. Metà della varianza del
  rumore appartiene all'immagine (comune alle sue 5 presentazioni), con lo stesso `rho` ai
  due livelli; validazione di 1.000 immagini per 5 presentazioni. Aggiunto, senza
  criterio, lo stesso conto a `dH = 2`.
- **AUC uguale a 0 o a 1**: spostata di mezza coppia dal bordo, perché `d_a` resti finito.
  Quante volte succede si riporta.
- **Bootstrap**: ricampionamento multinomiale delle immagini (turing-meter usa pesi di
  Poisson). Le due metà del turing sintonizzato si estraggono classe per classe, con un
  seme fisso.
- **H1a.** Un compito conta se l'intervallo di `phi` sta sopra i 90 gradi in tutti i semi,
  a tutti e due i valori più alti di `mu`.
- **H1b.** L'estremo inferiore del turing sintonizzato si confronta con `sqrt(1 + k^2)`
  calcolato con il `k` puntuale.
- **H1c.** L'errore relativo è misurato rispetto al turing sintonizzato.
- **H1e.** «`rho` sale con `mu`» non ha una soglia: si usa quella di H1a e H2a,
  correlazione di rango mediana almeno 0,9.
- **H1f.** Il riferimento è, per compito, la media sui semi della coppia opposta
  sintonizzata a `mu = 0`; a ogni `mu` decide la mediana del rapporto.
- **H2a, H2b, H3a.** Un compito (per H3a, l'esperimento) conta se la condizione vale allo
  stesso `lambda` in tutti i semi. **H3b**: decide la stima puntuale dell'angolo.
- **E3.** Il combinatore è un logit condizionato su `logit((c + 0,5) / (n + 1))` dei voti
  di ogni classe e sul logaritmo della probabilità della macchina, con convalida
  incrociata a 5 parti sulle immagini. Il gruppo da solo è il voto di maggioranza;
  sensibilità con `m = 10`; il turing del gruppo è la sensibilità del combinatore divisa
  per quella della maggioranza. La differenza appaiata degli angoli usa le formule e il
  seme di `compare_votes`.
- **E5.** La larghezza del nullo è il rapporto fra il bordo lontano e quello vicino, la
  convenzione della banda: con la differenza dei pesi il logaritmo cambia segno vicino a
  1 e il criterio di H5c non si legge (la differenza si riporta comunque). Nullo e bordi
  per interpolazione lineare in `w`. H5e divide per
  `sqrt(1 + k^2 - 2 k cos(phi)) / sin(phi)` senza il vincolo dei pesi non negativi (il
  picco del profilo di Fano). In H5d la deviazione standard di `a` si stima su prova e
  validazione separatamente, e il verso viene dal segno della pendenza della rete «meno»
  sulla prova.

**E2 ter ed E3 bis.** Scelte dell'analisi fissate l'11 novembre 2026, a codice scritto e
prima delle corse.

- **T1.** Mediana sulle unità del rapporto `d_a(M_due) / d_a(M_0+)`; i segni delle
  pendenze sono stime puntuali, come in B1; se `d_a(M_0+)` non è positivo in un'unità,
  T1 non passa.
- **T2, T3.** Quattro verdetti, uno per macchina. Un compito conta se l'estremo inferiore
  dell'intervallo sta sopra lo zero in ogni seme; le differenze sono prese rispetto a
  `M_0+` e appaiate sugli stessi ricampionamenti. Quelle rispetto a `M_0` si riportano
  senza criterio.
- **T4.** Un solo verdetto, che passa se passano tutte e due le macchine; turing
  apparente come in B5 (massimo sui pesi non negativi, per `k` di ogni segno).
- **T5.** Sensibilità della coppia sintonizzata `(H, M_due)` meno quella di `M_0+` da
  sola, appaiate; `M_ramo` si riporta senza criterio.
- **Il tetto di T3** (la coppia di E1 a `mu = 0`) si riporta come contesto, senza
  criterio: quelle reti sono addestrate su 9.000 immagini, non su una metà.
- **Affinamento di E2 ter.** Copia dei pesi di `M_0`, medie della normalizzazione
  comprese; passo 0,02; `M_0+` e `M_due` condividono seme, ordine dei lotti, ritagli e
  cornici.
- **E3 bis.** Misure sull'unione delle due metà non viste, per seme, come in E3; la
  maggioranza è il voto più frequente, con i pari merito divisi. P1: stime puntuali per
  metà e per seme, e una discesa di 2 punti esatti passa. P3: passa solo se una ricetta
  passa P2 e ha anche la differenza dei turing apparenti con l'estremo inferiore sopra lo
  zero in ogni seme; i turing apparenti sono quelli di `compare_votes` (pesi non
  negativi), e la differenza è appaiata sugli stessi ricampionamenti degli angoli. P4:
  se una macchina è al livello del caso l'angolo non è definito, e P4 non è valutabile.
  P5: solo `mista`.
- **I manifest** delle due corse citano la versione 1.4.

**Le versioni.** La versione 1.1 è dello stesso giorno della 1 e viene prima di ogni corsa:
precisa i conti di E0 e del bootstrap, che nella 1 non sarebbero stati eseguibili nel
tempo disponibile. La versione 1.2, dello stesso giorno, aggiunge la domanda D6 e
l'esperimento E5 sul nullo (§ 8 bis): è stata scritta a corse della GPU avviate, ma
prima di ogni analisi dei dati grezzi, e non cambia niente di ciò che c'era. La versione
1.3 aggiunge E2 bis (§ 6 bis), esplorativo: è scritta dopo aver visto i dati di E2 ed E3
e prima della sua corsa, e non tocca criteri né risultati delle altre prove. La versione
1.4, dell'11 novembre, aggiunge E2 ter (§ 6 ter) ed E3 bis (§ 7 bis), esplorativi:
nascono da una revisione della bozza dell'articolo, sono scritti dopo aver visto tutti i
risultati e prima delle loro corse, e non toccano criteri né risultati delle altre prove.
Sul server le due corse usano una sottocartella propria, che alla fine si cancella.

La versione 2.0, del 19 novembre 2026, viene dopo la pubblicazione dell'articolo e nasce
da una sua revisione. Aggiunge le domande D7 e D8 e le prove E6 (§ 12) ed E7 (§ 13). È
scritta prima delle due esecuzioni, insieme al codice dell'addestramento
(`gpu/trust.py`, `gpu/views.py`), a quello dell'analisi (`analysis/fixed.py`,
`analysis/trust.py`, `analysis/views.py`) e ai loro test. E6 mette alla prova, su
addestramenti nuovi, ciò che si è visto rileggendo i file di E2 ter: quella rilettura è
nel repository, dichiarata esplorativa. E7 non ha dati precedenti. Non tocca criteri né
risultati delle altre prove. Una nota sul banco di E1 ed E5, aggiunta con questa
versione senza cambiare i loro esiti: se la sensibilità si misura come differenza delle
medie divisa per la deviazione standard entro la classe, la formula del § 2 è
un'identità dei momenti del campione, vera per qualunque dato. Con `d_a`, che viene
dall'area sotto la curva ROC, i criteri H1c, H1d e H5a-H5e misurano quindi quanto gli
indizi delle reti si scostano dal modello gaussiano a varianze uguali, non la formula.
