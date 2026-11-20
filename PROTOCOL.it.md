# Risonanza cognitiva: protocollo

Versione 2.4, 20 novembre 2026. Scritto e registrato nel repository **prima di ogni corsa**:
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
cancellano. Chiamiamo **risonanza cognitiva** questa condizione, e le domande sono dieci.

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
- **D9.** Quanto vale il compagno costruito si può prevedere prima di costruirlo, da ciò
  che il giudice sa e la macchina no? E quando una macchina che conosce solo il giudice
  batte una macchina che conosce il compito? (Aggiunta della versione 2.1: vedi § 14.)
- **D10.** Quanto dell'accordo fra misura e formula visto in E1 dipende dalla forma del
  rumore che abbiamo iniettato? (Aggiunta della versione 2.1: vedi § 15.)
- **D11.** Per chi conviene una macchina costruita sulla persona, e per quale peso va
  costruita se non si sa chi la userà? (Aggiunta della versione 2.2: vedi § 12 bis.)

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
- Se G1 non passa, il conto della manopola (`w = b / c`) non basta a dire per chi conviene
  un compagno: il punto di pareggio va misurato caso per caso.
- Se G2 non passa, a chi dà alla macchina un quarto del dovuto, letto su questo banco, un
  compagno non conviene.
- Se V1 non passa, il compagno costruito non batte il solista nemmeno quando la persona
  sa qualcosa di suo: come obiettivo di progetto, a fiducia libera, la controfase non
  serve.
- Se V4 non passa, è sbagliata la lettura per cui la macchina addestrata sulla somma
  impara ciò che alla persona manca.
- Se K2 non passa, la spiegazione data «col senno di poi» nella prima versione
  dell'articolo (al giudice del banco non restava niente da dare) è sbagliata o non basta.
- Se K4 non passa, il modello additivo del pregiudizio non descrive le reti.
- Se S2 e S3 non passano, l'accordo di E1 non dipende dalla forma del rumore iniettato, e
  la critica per cui E1 misura la gaussianità cade.

## 10. Ordine, tempi, pulizia

Ordine di priorità: E0, E1, E2, E3. Una GPU L4 da 24 GB; tetto di due ore e mezza di
calcolo in tutto. Se il pilota dei tempi dice che E1 supera i 60 minuti si scende a due
semi, poi ai primi tre compiti, e lo si scrive nelle Deviazioni. Sul server si lavora in
una sola cartella, che alla fine si cancella.

E6-E9 (versione 2.1): cinque semi (2-6) e una stima di due ore sulla stessa GPU, in una
sottocartella propria che alla fine si cancella. Ordine: E6, E8, E7, E9. Se il tempo non
basta si scende a tre semi (2, 3 e 4), e lo si scrive nelle Deviazioni.

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
- `chart/{compito}_v{v}_s{seme}.npz` (E8): come `rephase_bis/`, con `z_H0`, `machines` =
  `M, Mramo, Mblind` e `v` (100, 50 o 25).
- `shape/{compito}_mu{mu}_s{seme}.npz` (E9): `a`, `z_plus`, `z_minus` di forma (4, N), una
  riga per forma della cornice di prova, con `shapes`.
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

- **H, `M_0`, `M_0+`, `M_due`** come in E2 ter, con i semi da 2 a 6. E1-E5 hanno usato 0 e 1.
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

## 12 bis. E6, lettura di progetto: per chi conviene un compagno

Aggiunto con la versione 2.2, il 19 novembre 2026. Le corse di E6 erano partite alle
09:49 UTC, mentre questa sezione veniva scritta; è stata registrata nel repository pochi
minuti dopo (l'ora la dà il registro di GitHub), prima che un file di E6 fosse letto o
analizzato. Non chiede corse nuove: usa i file di E6. Il codice dell'addestramento non
cambia, e i manifesti delle corse riportano la versione 2.1.

**Da dove viene.** E6 dice se il peso a cui una coppia rende si può decidere costruendo
la macchina. Chi la costruisce deve poi sceglierlo, quel peso, senza conoscere chi la
userà. Kong e colleghi (2026) stimano che le persone di ChatBench diano alla macchina
0,285 del peso che le spetta. Qui si fissa, prima di leggere i dati, come quella frazione si
legge sul banco e con quale regola si sceglie una macchina.

**Il dovuto e la persona.** Il dovuto, `ŵ_u`, è il peso del picco (§ 12) della coppia
`(H, M_0+)` nell'unità `u`, sui log-odds grezzi; `ŵ` è la sua mediana sulle unità. Chi dà
al solista una frazione `f` del dovuto usa il peso grezzo `f ŵ_u`, e lo usa con ogni
macchina, perché non vede come è costruita. È un'assunzione. La lettura opposta, in cui
la persona dà a ogni macchina la frazione `f` del peso che spetta a quella macchina, si
riporta accanto, senza criterio. Le frazioni sono 1/16, 1/8, 0,285, 1/2 e 1.

**Che cosa si misura** (`analysis/design.py`).

- *Il turing a una frazione*: `Tu_X(f ŵ_u)`, per interpolazione lineare in `ln w` della
  curva grezza dell'unità sui 49 pesi; fuori dalla griglia vale l'estremo.
- *Il compagno per `f`*: `M_ramo(w_b)` con `w_b` più vicino, in `ln w`, a `f ŵ`. Lo
  decide il peso per cui è costruito, non il risultato: scegliere il migliore dopo
  averli misurati tutti darebbe un vantaggio fatto di rumore.
- *Il pareggio* di `M_ramo(w_b)`: unità per unità, sulla differenza fra la curva grezza
  di `(H, M_ramo(w_b))` e quella di `(H, M_0+)`. Si parte dal peso della griglia più
  vicino a `w_b` e si sale: il pareggio è il primo peso a cui la differenza diventa
  negativa, per interpolazione lineare in `ln w`. Se a `w_b` la differenza non è
  positiva, o se non diventa mai negativa, il pareggio resta aperto.
- *Il pavimento*: `k_X = d_a(X) / d_a(H)`, la macchina da sola rispetto al giudice da solo.

**Che cosa dice il modello.** Se il pregiudizio del giudice è `b a` e il ramo vi aggiunge
`-(b / w_b) a`, nella somma a peso `w` resta `(1 - w / w_b) b a`. Con il solista resta
`b a`, e tutto il resto della somma è uguale. I due resti hanno la stessa ampiezza a
`w = 2 w_b`: sotto vince il compagno, sopra il solista. Nella rilettura di E2 ter, con
`w_b = 1`, le curve si incrociavano vicino a 2. E dalla formula del § 2,
`Tu(w)^2 - 1 = [2 w (k - ρ) + w^2 (k^2 - 1)] / (1 + w^2 + 2 ρ w)`: con `k ≥ 1` non è mai
negativo per `w > 0`, con `k < 1` lo diventa ai pesi grandi. Una macchina che da sola
vale almeno quanto il giudice non porta la coppia sotto il giudice a nessun peso
positivo, neppure in chi la copia.

Ipotesi. Le primarie sono G1 e G2.

- **G1, il pareggio sta a due volte il peso di costruzione.** Il pareggio di
  `M_ramo(w_b)` è chiuso in almeno 15 unità su 25 e sta, in mediana sulle unità in cui è
  chiuso, entro un fattore 1,5 da `2 w_b`, per ciascuno dei pesi `w_b` = 0,5, 1 e 2. Per
  `w_b = 1` la rilettura di E2 ter lo aveva già mostrato; per 0,5 e 2 è una previsione.
  A `w_b = 4` non si giudica: lì le due curve sono quasi piatte.
- **G2, a un quarto del dovuto conviene il compagno.** A `f = 0,285` la coppia con il
  compagno per `f` ha un turing più alto della coppia con `M_0+` in almeno 18 unità su
  25. Con una moneta, 18 su 25 o più esce con probabilità 0,022; le unità condividono i
  compiti e non sono indipendenti, quindi il conto è indicativo.
- **G3, chi dà il dovuto non ne ha bisogno.** A `f = 1` la differenza fra il turing della
  coppia con il compagno per `f` e quello della coppia con `M_0+` sta, in mediana sulle
  unità, fra -0,03 e 0,03.
- **G4, il compagno di chi si fida meno non regge il pavimento.** `k` è, in mediana sulle
  unità, sotto 1 per `M_ramo(0,5)` e almeno 1 per `M_ramo(1)`, `M_ramo(2)` e `M_ramo(4)`.

**La regola di scelta**, fissata qui. Per le persone fra `f_1` e `f_2`: fra le macchine
con `k` almeno 1 in mediana si prende quella con il più alto valore minimo della curva
mediana sulle unità, sui pesi della griglia fra `f_1 ŵ` e `f_2 ŵ` e ai due estremi; a
parità, quella con la media più alta sugli stessi punti. L'intervallo registrato va da
0,285 a 1. Il caso peggiore e non il valore atteso, perché di come le persone si
distribuiscono fra quei pesi non sappiamo niente, e chi copia la macchina è coperto dal
pavimento. **Attesa, senza criterio**: la regola sceglie `M_ramo(4)`, che per il modello
non sta sotto il solista fino al peso 8, con un vantaggio su `M_0+` nel caso peggiore
sotto 0,05 turing. Si riporta anche in quante unità la stessa regola, applicata alla
sola unità, fa la stessa scelta.

Nessuna di queste letture riguarda persone: il giudice è una rete, e la frazione di Kong
e colleghi viene da un altro compito.

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
- Semi da 2 a 6; `M_0`, il ramo e `M_coppia` hanno gli stessi semi alle tre
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

## 14. E8. Il banco della cartella

Aggiunto con la versione 2.1, il 19 novembre 2026, prima di ogni esecuzione di E6-E9. Il
disegno, il teorema e le attese riprendono quelli proposti da una revisione dell'articolo;
qui sono riscritti sulle reti e sugli stimatori di questo protocollo.

**Il teorema.** L'indizio del giudice è `h + beta a`: un indizio pulito `h`, con sensibilità
`d_h` e rumore di varianza uno, più un pregiudizio che non dipende dalla classe, di varianza
`B`. L'indizio della macchina `m` ha sensibilità `d_m`, rumore di varianza uno e correlazione
`r` con il rumore di `h`. Per la migliore combinazione lineare, con pesi di ogni segno:

    D1^2 = (d_h^2 - 2 r d_h d_m + (1 + B) d_m^2) / (1 + B - r^2)      il giudice con il solista m
    D2^2 = (d_h^2 - 2 r d_h d_m + d_m^2) / (1 - r^2)                  il giudice con m e con chi legge a
    D2^2 - D1^2 = B / (1 - r^2 + B) * P,     P = (d_h - r d_m)^2 / (1 - r^2)

`P = D2^2 - d_m^2` è ciò che il giudice pulito sa e la macchina no, e si annulla all'angolo
utile, `r = d_h / d_m`. Il giudice con un correttore cieco, che legge solo `a`, vale `d_h`, e
batte la coppia con il solista quando `d_m / d_h` sta fra
`(r - sqrt(B (1 + B - r^2))) / (1 + B)` e `(r + sqrt(B (1 + B - r^2))) / (1 + B)`. La soglia
che conta è la seconda; la prima supera lo zero solo se `r^2 > B`. Il codice è in
`src/cognitive_resonance/chart.py`, verificato contro l'inversione diretta delle matrici.

**Sui dati vecchi.** Applicato unità per unità ai file di E2 bis ed E2 ter
(`analysis/chart.py --e2`, `results/chart_from_e2.json`), il teorema ridà le coppie
misurate: con il solista 3,05 e 3,07 (misurate 3,04 e 3,06), con la macchina bendata 2,86
(misurata 2,85), con il solista più il ramo 3,13 (misurata 3,10). Dice anche perché T3 non è
passato: su quel banco il compagno costruito poteva dare l'1,4% in più. È una rilettura,
non una prova.

**Il banco.** In E1-E6 la macchina vede tutto ciò che vede il giudice, e `P` è quasi zero.
Qui il giudice vede la fotografia intera; la macchina vede la cornice e solo una parte
dell'interno: tutto (`v = 100`, il banco di E2 ter), la metà sinistra (`v = 50`, colonne
2-15) o il quarto sinistro (`v = 25`, colonne 2-8).

- **H**: rete «più», `mu = 1`, metà A, fotografia intera. **H0**: la stessa rete senza
  pregiudizio (stesso seme, cornice indipendente dalla classe).
- **`M_cieca`**: vede solo la cornice, addestrata sull'entropia incrociata di
  `z_H + z_cieca` (la `M_cornice` di E2 bis). È la stessa ai tre valori di `v`.
- Per ogni `v`: **`M`**, il solista sulla sua parte di fotografia; **`M_ramo`**, `M` congelata
  più un ramo che vede solo la cornice, addestrato sull'entropia incrociata di
  `z_H + z_M + z_ramo`.
- Semi 2-6, cinque compiti. `M` e il ramo hanno gli stessi semi ai tre valori di `v`.
- **Le grandezze del teorema, misurate unità per unità.** La pendenza dell'indizio di H
  sulla cornice, a parità di classe, dà il pregiudizio; ciò che resta dell'indizio tolto il
  pregiudizio è `h`. Da `h` e da `M`: `d_h`, `d_m` (con `d_a`), `r` e `B`.
- **Il miglior uso lineare dei tre indizi** `(z_H, z_M, a)`: i pesi si stimano su metà delle
  immagini e la sensibilità si legge sull'altra metà, poi si scambiano. È `D2` misurato senza
  passare da una rete.

Attese. Δ è la sensibilità della coppia sintonizzata `(H, M_ramo)` meno quella di `(H, M)`,
appaiate. Le primarie sono K2 e K4.

- **K1, il nullo di E2 ter.** A `v = 100`, `|Δ|` diviso per la sensibilità della coppia
  `(H, M)` è, in mediana sulle unità, al più 0,03.
- **K2, con qualcosa di suo il compagno vince.** A `v = 25`, Δ > 0 con intervallo che esclude
  lo zero in ogni seme, in almeno 3 compiti su 5.
- **K3, la dose.** La mediana di Δ cresce quando la macchina vede meno: `v = 25 > 50 > 100`.
- **K4, la legge.** A `v = 50` e `v = 25`, `D2^2 - D1^2` misurato con il miglior uso lineare
  dei tre indizi, diviso per il guadagno previsto dal teorema con le grandezze della stessa
  unità, sta in mediana fra 0,8 e 1,25.
- **K5, meno brava da sola, migliore in coppia.** A `v = 25`, `M_ramo` da sola è meno
  sensibile di `M` e la coppia con lei è più sensibile, con intervalli che escludono lo zero
  in ogni seme, in almeno 3 compiti su 5.
- **K6, chi conosce il giudice contro chi conosce il compito.** Quale fra `(H, M_cieca)` e
  `(H, M)` sia la coppia più sensibile è ciò che dice la soglia del teorema, calcolata con le
  grandezze dell'unità, in almeno l'80% delle unità (tutti i `v`).
- **K7, niente si crea.** La coppia `(H, M_ramo)` divisa per la coppia `(H0, M)` è, in
  mediana, al più 1,05 a ogni `v`.

Un limite scritto prima: `M_ramo` è costruita per il peso 1, e un peso solo deve cancellare
il pregiudizio e pesare la macchina. Con una macchina debole può rendere meno del miglior
uso dei tre indizi: per questo K4 usa quest'ultimo, e K2 la rete.

Dati grezzi: `chart/{compito}_v{v}_s{seme}.npz`, come `rephase_bis/`, con `z_H0`, `machines`
= `M, Mramo, Mblind` e `v`; `manifest.json`.

## 15. E9. La forma del rumore

Aggiunto con la versione 2.1, il 19 novembre 2026, prima di ogni esecuzione di E6-E9.

**Perché.** Con la sensibilità misurata come rapporto fra segnale e rumore la formula del § 2
è un'identità (`analysis/identity.py`): nessun dato può smentirla. Con `d_a`, che viene
dall'AUC, vale per indizi gaussiani. E in E1, oltre i 90 gradi, quasi tutto il rumore è la
cornice, che abbiamo estratto noi da una gaussiana. La domanda è quanto la misura si
allontana dalla formula quando quella parte del rumore ha un'altra forma.

**Che cosa si sapeva.** Una prova preliminare fatta durante una revisione, su reti con un
terzo dei canali, quattro compiti e un seme, ha dato per il rapporto fra coppia misurata e
prevista 0,29 con una cornice a due valori, 0,90 con una uniforme e 0,93 con una t di
Student, con attese sbagliate nel verso per due forme su tre. Le attese qui sotto vengono da
un modello: indizi gaussiani più una cornice della forma data, con i parametri mediani di E1
(§ 5). Il modello dà, a `mu = 1,5`, 1,09 (due valori), 1,04 (uniforme), 0,86 (t); a
`mu = 2`, 0,79, 1,05 e 0,80.

- **Le reti** sono quelle di E1 (`gpu/bench.py`): «più» e «meno», 9.000 immagini, cornice di
  addestramento gaussiana. `mu` = 1, 1,5 e 2; semi 2-6; cinque compiti.
- **Alla prova** la cornice è estratta da quattro distribuzioni con media zero e la stessa
  varianza `1 + mu^2`: `gauss` (è E1, su semi nuovi), `two` (due valori, `±sqrt(1 + mu^2)`),
  `uniform`, `t3` (t di Student con 3 gradi di libertà, riscalata). La standardizzazione usa
  la validazione con la stessa forma.
- **Misure** del § 3 per ogni forma, più il rapporto segnale/rumore accanto a `d_a`. Il
  rapporto è quello di H1c: turing sintonizzato diviso `Tu_app` da `k` e `rho` della stessa
  unità e forma.

Attese.

- **S1, replica.** Con la cornice gaussiana, a `mu` = 1,5 e 2: l'angolo sta sopra i 90 gradi
  in ogni unità e il rapporto sta in mediana fra 0,9 e 1,1.
- **S2, la formula promette troppo.** A `mu = 2` il rapporto mediano è sotto 0,9 sia con la
  cornice a due valori sia con la t.
- **S3, il verso dipende dalla forma.** A `mu = 1,5` il rapporto mediano è sopra 1 con la
  cornice a due valori e sotto 0,95 con la t.
- **S4.** Con la cornice uniforme il rapporto mediano resta fra 0,95 e 1,10 a `mu` = 1,5 e 2.
- **S5, la coppia è la stessa.** La sensibilità della coppia sintonizzata con una forma,
  divisa per quella con la cornice gaussiana, sta in mediana fra 0,95 e 1,05 per ogni forma
  a `mu` = 1,5 e 2: nella coppia la cornice si cancella, e a cambiare è ciò che promettono i
  singoli.
- Senza criterio: lo scarto dell'identità con il rapporto segnale/rumore, per ogni forma.

Dati grezzi: `shape/{compito}_mu{mu}_s{seme}.npz`: `a`, `z_plus` e `z_minus` di forma
(4, N), una riga per forma nell'ordine di `shapes`; `manifest.json`.

## 16. E10. La cartella che dice il vero

Aggiunto con la versione 2.3, il 20 novembre 2026, dopo una quarta revisione dell'articolo e
prima di ogni addestramento di E10, E11 ed E12. Le tre prove condividono un banco e una
corsa (`gpu/trace.py`), su semi mai usati (7-11).

**Perché.** In E8 la cornice, alla prova, non dice niente della classe. Una traccia vera
spesso dice qualcosa e riceve troppo peso. Il teorema del § 14 si estende a quel caso
(`cognitive_resonance.chart`): con una traccia di sensibilità `d_a` a cui il giudice dà
peso `beta`, il peso che merita è `beta_0 = d_a (1 - r^2) / (d_h - r d_m)` e il guadagno di
un compagno è `B / (1 - r^2 + B) * P * (1 - beta_0 / beta)^2`. Finora lo abbiamo guardato
solo ricampionando i file di E8, con un ramo lineare a due numeri, dopo aver visto i
risultati: una lettura, non una prova.

**Che cosa si sapeva** (dai file di E8 ricampionati: `analysis/informative.py`,
`analysis/informative_face.py`). Con la macchina che vede un quarto della fotografia il ramo
lineare toglie il 68 e il 31 per cento della spinta dove l'eccesso `1 - beta_0 / beta` del
teorema è 68 e 34 (cornice di sensibilità 0,5 e 1). Con la macchina che vede tutto toglie il
61 e il 13 dove il teorema dice 21 e -53: lì l'accordo non c'è, e non ne conosciamo la
ragione. Al peso migliore il compagno di E8, che toglie tutta la spinta, vale quanto il ramo
nuovo o poco di più (0,02 in 25 unità su 25 a sensibilità 0,5): la frase dell'articolo
«butterebbe via anche la parte buona» non ha lì un sostegno. A peso fissato sì: con i due
indizi sommati così come sono il ramo nuovo supera il compagno di E8 di 0,05, 0,17 e 0,38
(sensibilità 0,5, 1 e 1,5; 25 unità su 25), e da sensibilità 1 in su il compagno di E8 fa
peggio del solista senza ramo (-0,18 e -0,38, 0 unità su 25).

**Il banco.** La cornice a livello `d` è `a = mu_d (2y - 1) + sd_d eps`, con
`mu_d = d sqrt(2 / (4 + d^2))` e `sd_d^2 = 2 - mu_d^2`: varianza totale 2 come in E8,
sensibilità `d`. I livelli sono 0 (il banco di E8), 0,5, 1 e 1,5. In validazione e alla prova
`eps` è lo stesso ai quattro livelli.

- **H**: il giudice di E8 (rete «più», `mu = 1`, metà A, fotografia intera). È stato
  addestrato dove la cornice valeva 2: alla prova le dà più peso di quanto merita.
- **`C_d`**, uno per livello: stessa rete e stesso seme, addestrata con cornici di livello
  `d`. È il giudice che dà alla cornice il peso che ha alla prova (`C_0` è l'H0 di E8).
- Per ogni vista della macchina (`v = 25`, il quarto sinistro; `v = 100`, tutto l'interno):
  **`M`**, il solista, sulla sua parte di fotografia. Qui non vede la cornice: in E8 la
  vedeva senza che dicesse niente, e un solista che la leggesse quando dice il vero sarebbe
  un'altra macchina. **`R_t`**, uno per livello `t`: un ramo che vede solo la cornice,
  addestrato da zero sull'entropia incrociata di `z_H + z_M + z_ramo` con cornici di livello
  `t`. Il compagno costruito per il livello `t` è `M + R_t`; **`R_0` è il compagno di E8**,
  addestrato dove la cornice non diceva niente, che toglie tutta la spinta.
- I quattro rami di un'unità hanno lo stesso seme, lo stesso ordine e lo stesso rumore delle
  cornici: cambia solo il livello. Ogni ramo è letto a ogni livello.
- Semi 7-11, cinque compiti, 30 epoche e gli iperparametri della prova pilota.

**Misure**, per unità e per livello `d` della prova, con il compagno costruito per quel
livello. `s`, la quota della spinta che il ramo toglie: meno la pendenza dell'indizio grezzo
del ramo sulla cornice, divisa per la pendenza dell'indizio grezzo di H, a parità di classe.
`e = 1 - beta_0 / beta`, l'eccesso secondo il teorema, con le grandezze della stessa unità
misurate come in `analysis/informative.py`. Le coppie sintonizzate del § 3 e le coppie così
come sono (somma degli indizi grezzi), con il bootstrap appaiato per immagine.

Attese. Le primarie sono T2 (qui), X1 e X2 (§ 17), Y1 e Y2 (§ 18).

- **T1, la dose.** A `v = 25` la mediana di `s` sta fra 0,9 e 1,1 a livello 0 e cala al
  crescere del livello: `s(0) > s(0,5) > s(1) > s(1,5)`.
- **T2, solo l'eccesso (primaria).** A `v = 25`, ai livelli 0,5 e 1, la mediana sulle unità
  di `|s - e|` è al più 0,10.
- **T3, a vista intera il teorema non descrive il ramo.** A `v = 100` e livello 0,5 la
  mediana di `s - e` è sopra 0,20: è ciò che si è visto sui dati vecchi, scritto come attesa
  perché una prova nuova possa smentirlo.
- **T4, dove c'è un eccesso il compagno vince.** A `v = 25` e livello 0,5 la coppia
  sintonizzata `(H, M + R_0,5)` è più sensibile della coppia `(H, M)`, con intervallo della
  differenza sopra lo zero in ogni seme, in almeno 3 compiti su 5.
- **T5, dove il peso è quasi giusto non c'è niente da prendere.** A `v = 25` e livello 1,5,
  così come sono, la differenza fra la coppia con `M + R_1,5` e la coppia con `M`, divisa
  per la seconda, è in mediana fra -0,03 e 0,03.
- **T6, niente si crea.** A `v = 25` e a ogni livello la coppia sintonizzata `(H, M + R_d)`
  divisa per la coppia `(C_d, M)` è in mediana al più 1,05.
- Senza criterio: le quote di risposte giuste di ogni coppia, la quota del guadagno previsto
  che il compagno prende, e tutte le misure a `v = 100`.

Dati grezzi: `trace/{compito}_v{v}_s{seme}.npz`: `levels`, `y`, `image`, `eps`, `a` (4, N),
`z_H` (4, N), `z_C` (4, N: `C_d` al suo livello), `z_M` (N), `z_R` (4, 4, N: costruito per,
letto a), e gli stessi con `val_`; `manifest.json`.

## 17. E11. Togliere tutto o solo l'eccesso, a peso fissato

Aggiunto con la versione 2.3, sugli stessi file di E10.

**Perché.** Il teorema dice che chi toglie tutta la spinta, quando la traccia dice il vero,
butta via anche la parte buona. Al peso migliore i dati vecchi non lo mostrano: il peso
compensa. La domanda va fatta a peso fissato, dove nessuno aggiusta niente: i due indizi
sommati così come sono.

Misure: la sensibilità della somma grezza `z_H + z_M + z_R`, con `R_d` (il ramo costruito
per il livello della prova, «nuovo») e con `R_0` (il compagno di E8, «vecchio»), e della
somma `z_H + z_M` (il solista); differenze appaiate.

- **X1, il nuovo batte il vecchio (primaria).** A `v = 25` e livello 1 la coppia con
  `M + R_1` è più sensibile della coppia con `M + R_0`, con intervallo sopra lo zero in ogni
  seme, in almeno 3 compiti su 5.
- **X2, togliere tutto fa peggio che non togliere niente (primaria).** A `v = 25` e livello
  1,5 la coppia con `M + R_0` è meno sensibile della coppia con `M`, con intervallo sotto lo
  zero in ogni seme, in almeno 3 compiti su 5.
- **X3, la dose.** A `v = 25` la mediana della differenza fra nuovo e vecchio è positiva ai
  livelli 0,5, 1 e 1,5 e cresce con il livello.
- **X4, il peso compensa.** A `v = 25` e ai livelli 0,5 e 1 le coppie sintonizzate con
  `M + R_d` e con `M + R_0` differiscono in mediana, in valore assoluto, di al più 0,02
  volte la seconda.
- **X5, anche a vista intera.** A `v = 100` e livello 1 vale ciò che dice X1.
- Senza criterio: le quote di risposte giuste delle tre somme a ogni livello.

Se X1 non passa, la frase sull'articolo va tolta. Se X2 non passa, resta vero soltanto che
il compagno nuovo non perde.

## 18. E12. La spinta stimata da sì e no

Aggiunto con la versione 2.3, sugli stessi file (`v = 25`, livello 0: il caso del § 14).

**Perché.** `analysis/judgments.py` stima la spinta dall'indizio graduato del giudice, e il
pregiudizio che resta dopo `n` giudizi è `1 / n`. Una persona non dà un indizio graduato:
dà una risposta. Con un sì o un no l'informazione sulla pendenza è minore, e i giudizi che
servono sono di più.

**La previsione.** Nel modello la risposta è sì quando `h + beta a > 0`: una regressione
probit della risposta sulla classe (due intercette) e sulla cornice dà `beta` in unità del
rumore pulito. L'informazione di Fisher sulla pendenza, per giudizio, è
`E[lambda a^2] - sum_y p_y E_y[lambda a]^2 / E_y[lambda]`, con
`lambda = phi(eta)^2 / (Phi(eta) (1 - Phi(eta)))` e `eta = m_y + beta a`, dove `m_y` è la media
dell'indizio pulito nella classe `y`; con l'indizio graduato è la varianza di `a`. Il loro
rapporto `kappa` dice quanti sì e no valgono un giudizio graduato: con le grandezze mediane
di E8 (medie `±1,45`, `beta` 0,6, varianza della cornice 2) fa circa 5, e sta fra 4 e 6 per
giudici un po' meno o un po' più sensibili (`cognitive_resonance.chart.yes_no_factor`). La quota di guadagno che resta con `n` sì e no è quella prevista
per `n / kappa` giudizi graduati (`cognitive_resonance.chart.share_kept_after`).

**Come.** Per unità e per `n` (5, 10, 20, 50, 100, 200, 500, 1.000, 5.000): 40 estrazioni di
`n` presentazioni dalla validazione, le stesse per i due stimatori (una sola estrazione
quando `n` è tutta la validazione). Graduato: la pendenza dell'indizio standardizzato sulla
cornice, a parità di classe. Sì e no: la regressione probit della risposta del giudice
(il segno dell'indizio grezzo), con una penalità quadratica di 0,05 sui tre coefficienti
perché resti definita quando le risposte separano le classi; la pendenza si riporta alle
unità dell'indizio standardizzato dividendo per `sqrt(1 + beta^2 var(a))`. La correzione si
applica all'indizio graduato del giudice alla prova, e la quota che resta è
`(D_n^2 - D_1^2) / (D_tutti^2 - D_1^2)` come in `analysis/judgments.py`, mediana sulle
estrazioni. Restano fuori le unità con un guadagno al quadrato sotto 0,2.

- **Y1, con sì e no ne servono di più (primaria).** Con 50 giudizi la quota che resta con
  sì e no è più bassa di quella con l'indizio graduato in almeno 20 unità su 25.
- **Y2, quanti di più (primaria).** Il numero di giudizi a cui la quota che resta arriva a
  0,90 (interpolazione lineare in `ln n` sulla griglia) è, con sì e no, fra 3 e 9 volte
  quello con l'indizio graduato, in mediana sulle unità. Un'unità che con sì e no non
  arriva a 0,90 nemmeno con 5.000 giudizi conta come fuori.
- **Y3, la previsione.** Con 100 e con 200 sì e no la mediana sulle unità dello scarto, in
  valore assoluto, fra la quota misurata e quella prevista con `kappa` dell'unità è al più
  0,07.
- **Y4, alla lunga basta.** Con 5.000 sì e no la quota che resta è, in mediana, almeno 0,90.

Se Y2 passa, «poche decine di giudizi» nell'articolo diventa «fra cinquanta e cento» per chi
risponde sì o no. Se non passa perché ne servono più di nove volte tanti, o perché Y4 non passa, la
stima da risposte secche con questo modello non basta, e l'articolo lo dice.

## 19. E13. Il pregiudizio che si legge, su persone

Aggiunto con la versione 2.4, il 20 novembre 2026, dopo due revisioni indipendenti
dell'articolo. È la prima prova su persone misurate una per una. Non chiede addestramenti.

**Perché.** Sul banco il pregiudizio l'abbiamo piantato noi, e vale circa 0,7 del rumore
pulito del giudice. Sulle persone abbiamo solo un conto, fatto sui giudici d'asilo di Chen,
Moskowitz e Shue: fra 0,002 e 0,01. Nessuno dei due numeri è una misura di quanto, in una
persona, il rumore del giudizio si legga da una traccia, né di quanto renda toglierlo. I
dati per misurarlo sono pubblici: la raccolta di Rahnev e colleghi (2020, Confidence
Database, `osf.io/s46pr`, licenza CC0) ha, prova per prova, lo stimolo, la risposta e la
sicurezza dichiarata da migliaia di persone in compiti di scelta fra due risposte.

**Che cosa si misura.** Per ogni persona il giudizio graduato è la risposta con la sua
sicurezza. La traccia è la storia delle sue prove precedenti, che non dice niente dello
stimolo che sta per arrivare. Il correttore è la macchina bendata del § 14: non vede lo
stimolo, conosce solo la persona. `B` è la varianza della parte del giudizio che la traccia
prevede, a parità di classe, divisa per la varianza del resto.

**Dati e inclusione**, fissati prima di aprire un file di dati (alla scrittura di questo
paragrafo è stato letto solo l'elenco delle cartelle del progetto su OSF).

- I file: tutti i `data_*.csv` della cartella «Confidence Database».
- Colonne richieste: `Subj_idx`, `Stimulus`, `Response`, `Confidence`. Le righe con un valore
  mancante o non numerico in una delle quattro si tolgono per prime; la storia si calcola
  sulle righe rimaste.
- Compito a due risposte: dopo la pulizia `Stimulus` e `Response` hanno gli stessi due
  valori distinti. Gli altri file restano fuori.
- Sicurezza con almeno tre valori distinti nel file, e orientata: sull'intero file la
  sicurezza media delle risposte giuste supera quella delle sbagliate. Altrimenti il file
  resta fuori.
- Ordine: le righe di una persona, nell'ordine del file, sono le sue prove nell'ordine in
  cui le ha fatte. È un'ipotesi: se in un file non fosse vero, lì il pregiudizio misurato
  sarebbe più piccolo del vero.
- Persona inclusa: almeno 400 prove valide, fra il 55 e il 95 per cento di risposte giuste,
  ogni classe in almeno il 20 per cento delle prove, almeno tre valori distinti di sicurezza.
- File incluso: almeno 10 persone incluse. L'unità che si ricampiona è il file.

**Il giudizio graduato.** Per ogni persona le prove si ordinano dalla risposta più sicura
per la prima classe alla più sicura per la seconda (chiave: segno della risposta per uno più
il rango della sicurezza), e a ogni prova si dà il punteggio normale del suo rango medio,
`Phi^-1((rango - 0,5) / n)`, su tutte le prove della persona.

**La traccia**, tre numeri noti prima che lo stimolo compaia: il segno della risposta
precedente, il giudizio graduato precedente, la media dei giudizi graduati delle venti prove
precedenti (di quelle che ci sono, se sono meno). La prima prova di ogni persona resta fuori.

**Gli stimatori.**

- `B` di una persona: la regressione ai minimi quadrati del giudizio sulla classe (due
  medie) e sulle tre tracce, su tutte le sue prove; `R^2` parziale delle tracce, corretto
  per i gradi di libertà (`1 - (1 - R^2)(n - 2)/(n - 5)`), e `B = R^2 / (1 - R^2)`. Può
  venire negativo.
- Il correttore personale: le pendenze stimate sulla prima metà delle prove della persona
  si tolgono dal giudizio nella seconda metà. Il guadagno `G` è la sensibilità del giudizio
  corretto divisa per quella del giudizio com'è, tutte e due sulla seconda metà, ricavate
  dall'area sotto la curva ROC (`d' = sqrt(2) Phi^-1(AUC)`).
- Il correttore di tutti: le stesse pendenze stimate sulle prime metà delle altre persone
  dello stesso file (giudizi e tracce centrati entro persona e classe), applicate alla
  seconda metà della persona.
- Il controllo: le prove di ogni persona rimescolate una volta (seme ricavato dal nome del
  file e dalla persona), e tutto rifatto. Lì la storia non dice niente.
- Riassunto: la mediana sulle persone, con l'intervallo al 95 per cento da 2.000
  ricampionamenti dei file.

**Esplorazione e conferma.** I file si dividono con una regola scritta prima: la prima
cifra esadecimale dello SHA-1 del nome del file (in minuscolo) è 0, 1 o 2 per i file di
esplorazione, un'altra per quelli di conferma. Il codice si scrive e si prova su dati
simulati e sui soli file di esplorazione, poi si registra nel repository, e solo dopo gira
una volta sui file di conferma. I criteri si leggono sui file di conferma; quelli di
esplorazione si riportano a parte. Ogni modifica al codice dopo la prima corsa di conferma
va nelle «Deviazioni».

**Le attese.**

- **P1, c'è (primaria).** La mediana di `B` sulle persone è maggiore di zero, con un
  intervallo che esclude lo zero. Nel controllo rimescolato l'intervallo della mediana
  contiene lo zero, oppure la mediana è sotto 0,002 in valore assoluto.
- **P2, è piccolo (primaria).** La mediana del guadagno del correttore personale è sotto
  1,02. L'articolo sostiene che cancellare a una persona il pregiudizio che si legge rende
  poco: se la mediana è 1,02 o più, rende più di quanto abbiamo scritto, e l'articolo lo dice.
- **P3, segue la formula.** Per ogni persona il guadagno previsto è
  `G^2 = (1 + B_1) / (1 + 3/n_1)`, con `B_1` stimato come sopra sulla sola prima metà e
  `n_1` le prove della prima metà: è la (4) con il pregiudizio che resta dopo `n` giudizi,
  `p/n` con tre tracce. Sulle persone di conferma la mediana di `G^2` misurato meno `G^2`
  previsto sta fra -0,01 e 0,01, e la correlazione per ranghi fra i due è positiva, con un
  intervallo che esclude lo zero.
- **P4, su misura o per tutti (senza soglia).** In quante persone il correttore personale
  rende più di quello di tutti, e di quanto in mediana.
- Si riportano senza soglia: la quota di persone con `B` sopra 0,05 e sopra 0,2; il decimo e
  il novantesimo percentile di `B` e di `G`; l'angolo fra l'errore della persona e quello
  del correttore, `arccos(-sqrt(B/(1 + B)))`, alla mediana di `B`.

**Che cosa non dice.** La traccia è una sola famiglia, la storia delle prove, scelta prima
dalla letteratura (Fründ, Wichmann e Macke, 2014): pregiudizi che si leggono da altro non
entrano. La sicurezza dichiarata è una scala a pochi gradini, non un indizio continuo. I
compiti sono percettivi o di memoria, di laboratorio. E il piano lo ha scritto chi esegue,
lo stesso giorno: la divisione fra esplorazione e conferma impedisce di scegliere stimatori
e soglie guardando i file di conferma, non sostituisce una replica fatta da altri.

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

**E6, E7, E8 ed E9.** Le corse sono del 19 novembre 2026, su una scheda condivisa con
altri utenti. Nessuna riduzione: cinque semi e cinque compiti in tutte e quattro le prove.

- **Tre sessioni.** Il programma che lancia le corse cede la scheda quando un altro utente
  la usa, e riparte quando è libera. È successo due volte: le sessioni vanno dalle 09:49
  alle 10:15, dalle 10:19 alle 11:27 e dalle 19:20 alle 20:13 (UTC). Un'unità si scrive
  quando finisce; quelle in corso a un'interruzione sono ripartite da capo, con gli
  stessi semi. E6: 18 unità scritte nella prima sessione, 7 nella seconda. E8: tutte
  nella seconda. E7: 51 nella seconda, 24 nella terza. E9: tutte nella terza.
- **Un processo di E7 fermo.** Nella seconda sessione uno dei tre processi di E7 si è
  fermato dopo la sua prima unità. Il suo registro è stato sovrascritto dalla terza
  sessione, e la causa non è documentata. Le altre sue 24 unità sono state eseguite
  nella terza sessione, con un solo processo sulla scheda. I semi sono dell'unità, non
  del processo.
- **I tempi nei manifest.** `stage_time` riporta solo l'ultima sessione; i secondi di ogni
  unità stanno in `unit_seconds`. Le tre note qui sopra sono scritte anche nei manifest.
- **L'ambiente** è stato ricostruito per queste corse: numpy 2.5.5 (2.5.4 in E1-E5);
  python, torch, torchvision, CUDA e cuDNN sono gli stessi.
- **Le ore del 19 novembre, UTC.** Il programma di lancio parte alle 09:49:19 e prepara
  l'ambiente; la versione 2.2 di questo protocollo arriva nel repository alle 09:50:31;
  il primo addestramento di E6 parte alle 09:55:39; il codice della lettura di progetto
  (`analysis/design.py`) arriva nel repository alle 09:55:45; il primo file di E6 è
  scritto alle 09:59:05. Il codice della GPU usato sul server coincide, file per file
  (md5), con quello del commit `ab216dc`.
- **I manifest** citano la versione 2.1: è quella del codice della GPU, che la 2.2 non
  tocca.
- **L'analisi** è quella scritta prima delle corse, senza modifiche. Dove il protocollo
  lasciava un margine, la scelta è scritta accanto al criterio nei file dei risultati
  (`rule`).

**Letture aggiunte dopo i risultati di E6-E9.** Del 20 novembre 2026, dopo una revisione
dell'articolo. Non sono prove: nascono da risultati già visti, non hanno criteri, e non
toccano criteri né esiti delle prove registrate.

- **Il teorema del § 14 sui file di E7** (`analysis/chart.py --views`,
  `results/chart_from_e7.json`). Lo stesso conto di E8 su un banco per cui non era stato
  registrato, dove anche la macchina vede cose che il giudice non vede. In E7 non è stata
  addestrata una macchina bendata: il correttore cieco è il miglior uso lineare
  dell'indizio del giudice e della cornice, stimato su metà delle immagini e letto
  sull'altra.
- **La spinta del ramo sui file di E6** (`analysis/push.py`, `results/e6_push.json`). La
  lettura del § 12 bis trova il pareggio contro `M_0+` prima del doppio del peso di
  costruzione. Si misurano, sui log-odds grezzi, la pendenza `b` dell'indizio del giudice
  sulla cornice e la pendenza `c` del ramo, e il pareggio di ogni compagno anche contro
  `M_0`, la stessa macchina senza il ramo. In mediana il ramo lascia meno dello 0,5 per
  cento della spinta, e contro `M_0` il pareggio cade a 1,02, 2,05, 4,25 e 8,96 per i pesi
  0,5, 1, 2 e 4: l'anticipo del confronto registrato viene dal controllo, che da solo è
  una macchina un po' migliore di `M_0`.

**Letture aggiunte dopo una terza revisione.** Del 20 novembre 2026. Non sono prove: non
hanno criteri, e non toccano criteri né esiti delle prove registrate.

- **Quanti giudizi servono** (`analysis/judgments.py`, `results/e8_judgments.json`). Sui
  file di E8, la pendenza dell'indizio del giudice sulla cornice si stima da `n` giudizi
  con la risposta nota; il pregiudizio che resta vale `1 / n` in unità del rumore pulito.
- **Quando la traccia dice qualcosa di vero** (`analysis/informative.py`,
  `results/e8_informative.json`). Il teorema del § 14 si estende a una traccia informativa
  a cui il giudice dà un peso `beta` diverso da quello che merita, `beta_0`: il guadagno è
  `B / (1 - r^2 + B) · P · (1 - beta_0 / beta)^2` (`cognitive_resonance.chart`, verificato
  contro l'inversione delle matrici). Nessuna corsa ha una cornice informativa alla prova:
  la si ottiene ricampionando le presentazioni di E8, e il ramo nuovo è lineare.
- **Il teorema sulle persone già misurate** (`analysis/private_people.py`,
  `results/e4_private.json`). Sulle 74 coppie di persone e macchine di turing-meter, con
  `d_H`, `d_m` e `rho` pubblicati: quanto crescerebbe la coppia migliore se una quota del
  rumore della persona fosse un pregiudizio leggibile. Il pregiudizio delle persone non è
  misurato.
- **Due conti senza dati**: il tetto `(1 + w k) / |1 - w|` per ogni `k`, e le radici di due
  sistemi che si adattano l'uno all'altro (`cognitive_resonance.model`).
- **Le risposte giuste di E8, coppia per coppia** (`analysis/results_v2.py`, dalle unità di
  `results/e8_chart.json`): le mediane dei due indizi sommati così come sono, la
  differenza fra compagno e solista con minimo e massimo sulle unità, in quante unità il
  compagno supera la macchina cieca, e in quante il giudice senza pregiudizio, da solo,
  supera il giudice corretto dal compagno.

**E10, E11 ed E12.** Del 20 novembre 2026; le ore sono UTC, quelle del repository vengono dal
registro delle attività di GitHub.

- La versione 2.3 di questo protocollo arriva nel repository alle 16:34:13 (`f6d9795`).
- Sul server ambiente e dati sono preparati fra le 16:33 e le 16:36, senza addestrare niente.
- Alle 16:38:18 un controllo tecnico: un'epoca, un compito, un seme fuori piano (99), in una
  cartella a parte cancellata subito. Serve a sapere che lo script arriva in fondo; i suoi
  file non sono stati letti da nessuna analisi.
- La corsa parte alle 16:38:53 e finisce alle 17:44:30: 3.936 secondi, tre processi, una
  sessione sola (nessun altro ha chiesto la scheda), 50 file.
- Le analisi `analysis/trace.py` e `analysis/yesno.py` arrivano nel repository alle 16:41:43
  (`84d27f8`); il primo file della corsa è delle 16:43:56.
- Ambiente proprio: torch 2.15.0+cu132, numpy 2.5.5. La cartella sul server è stata
  cancellata dopo aver copiato i file e confrontato le impronte.

**Letture aggiunte dopo i risultati di E10-E12.** Del 20 novembre 2026. Non sono prove, e
non toccano criteri né esiti.

- **Che cosa lascia il ramo** (`analysis/trace_left.py`, `results/e10_trace_left.json`). T3 è
  passato: a vista intera l'eccesso del teorema non descrive ciò che il ramo toglie. Il ramo
  toglie le stesse quote alle due viste, e lascia alla cornice un coefficiente vicino alla
  pendenza del suo rapporto di verosimiglianza (0,38, 0,89 e 1,51 contro 0,36, 0,79 e 1,33).
  L'eccesso del teorema è quello del miglior uso dei tre indizi e dipende dalla macchina; il
  ramo, addestrato sulla somma così com'è, no. Coincidono dove la macchina vede un quarto.
- **Il turing in tre fattori** (`analysis/factors.py`, `results/pair_factors.json`). Il teorema del § 14
  diviso per ciò che il giudice vale da solo: con il compagno `Tu^2 = (1 + B)(q^2 + p)`, con
  `q = d_m / d_h` e `p = (1 - r q)^2 / (1 - r^2)`; con il solista `p` si riduce di
  `(1 - r^2) / (1 - r^2 + B)`, con la macchina cieca resta `1 + B`. È la stessa algebra
  riordinata (`cognitive_resonance.chart.pair_factors`), letta sui file di E1, E8 ed E10.
- **Perché la previsione di E12 era sbagliata** (`analysis/yesno.py`, campi `kappa_corrected`
  e `predicted_yes_no_corrected`, aggiunti dopo). La pendenza probit è in unità del rumore
  pulito; riportarla alle unità dell'indizio intero con la stessa stima ne riduce l'errore di
  `(1 + B)^(3/2)`. Il pregiudizio che resta con `n` sì e no è `kappa / (n (1 + B)^2)`, non
  `kappa / n`: con le grandezze della prova fa 1,8 volte i giudizi graduati, dove il
  misurato è 1,6 e la fascia scritta prima era da 3 a 9. Con un pregiudizio piccolo il
  fattore torna vicino a `kappa`.

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

La versione 2.1, dello stesso giorno, viene prima di ogni esecuzione di E6 ed E7 e nasce da
due altre revisioni dell'articolo. Porta i semi di E6 ed E7 da tre a cinque (2-6), senza
toccarne ipotesi e criteri, e aggiunge le domande D9 e D10 e le prove E8 (§ 14, il banco
della cartella con il suo teorema) ed E9 (§ 15, la forma del rumore), con il codice
dell'addestramento (`gpu/chart.py`, `gpu/shape.py`), dell'analisi (`analysis/chart.py`,
`analysis/shape.py`) e i test. «In ogni seme» vuol dire ora in tutti e cinque. Il teorema
applicato ai file di E2 bis ed E2 ter è una rilettura di dati vecchi, dichiarata come
tale. In `analysis/fixed.py` sono state aggiunte, senza criterio, le quote di risposte
giuste.

La versione 2.2, dello stesso giorno, è stata registrata a corse di E6 appena partite
(09:49 UTC) e prima che un loro file fosse letto. Aggiunge la
domanda D11 e il § 12 bis, una lettura dei file di E6 che non chiede corse nuove: come si
legge sul banco la frazione di peso stimata da Kong e colleghi, una previsione senza
parametri (il compagno costruito per un peso batte il solista fino a due volte quel
peso), e la regola con cui si sceglie una macchina, fissata prima di leggere i dati. Il codice
dell'addestramento non cambia.

La versione 2.3, del 20 novembre 2026, viene da una quarta revisione dell'articolo e prima di
ogni addestramento delle prove che aggiunge: E10, E11 ed E12 (§§ 16-18), sullo stesso banco
e nella stessa corsa, con semi mai usati. Le tre letture fatte dopo i risultati (traccia
informativa, quanti giudizi, ramo nuovo contro vecchio) diventano tre prove con le attese
scritte prima; le soglie vengono dai dati vecchi, e lo dichiarano. Non tocca criteri né
esiti delle prove registrate.

La versione 2.4, dello stesso giorno, viene da due revisioni indipendenti dell'articolo e
prima di aprire un file di dati della prova che aggiunge: E13 (§ 19), la misura del
pregiudizio che si legge su persone, sui dati pubblici del Confidence Database. Non tocca
criteri né esiti delle prove registrate.
