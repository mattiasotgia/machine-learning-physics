# Introduzione

(19 febbraio 2024)

Obiettivo di questo corso non è tanto trovare e mostrare nel dettaglio il funzionamento di algoritmi di ML, quanto invece osservare come questi algoritmi possono esssere utilizzati per affrontare esempi di fisica, o più in geenrale applicati all'ambito della ricerca fisica (o scientifica). 

Esistono molti testi per quanto riguarda gli argomenti che sono trattati in questo corso, i principali sono in bibliografia {cite}`aggarwal2018, cowan1998a, goodfellow2016, hastie2009, lista2017a, plehn2022, rosasco2017`.

Esistono quattro aspetti essenziali su cui vi è forte divergenza tra il _general purpouse machine learning (GPML) e invece il _scientific machine learning_ (SML). Questi sono 
 1. Lo scopo (scope)
 2. Il modo in cui vengono rappresentati i dati (representation)
 3. La quantità e la qualità dei dati (amount and quality of data)
 4. Quanto ci aspettiamo che un certo algoritmo performi rispetto ad un algoritmo GPML (performance)

| GPML | SML | | |
| - | - | - | - |
| Fare cose che sono facili per l'uomo ma difficili per il computer (recognizing objects, listening, reasoning, driving, playing video games), ovvero ottenere un AGI (artificial general intelligence) | Risolvere problemi troppo difficili per l'uomo e anche per il computer (integration, doing math, symbolic math, simulate syntethic data), ovvero un ASI (artificial scientific intelligence?) | 1. scope | La differenza principale tra i due casi (GPML vs SML) è che sotto al cappello della SML si ipotizza che ci sia un modello, che quindi i dati possono utilizzare per _imparare_. | 
| Cercando di replicare l'uomo, la rappresentazione dei dati è allineata con i cinque sensi (sight $\to$ image, writing; earing $\to$ audio, video, eg.: transcribe a spoken text; other application to touch, mainly in robotics; other application in smell, industry use). Ci riferiamo a queste rappresentazioni come UNCODED (anche se spesso in letteratura queste sono riferite come CODED)  | Queste rappresentazioni sono dette CODED, essendo dati rappresentati in modi informatici. Questi sono rappresentati come dati tabulari, plot | 2. representation | Queste differenze sono estremamente importanti. Il grosso della evoluzione è infatti avvenuto nella GPML, dove gli algoritmi sono specialmente funzionali con questi modi di rappresentazioni. Quindi quello che si è spesso fatto è stato di convertire dalla rappresentazione per la SML alle rappresentazioni di GPML. Questo però ha chiaramente un costo che ha reso molto più difficile lo sviluppo della SML. Si sta cercando ora di capire come generalizzare gli algoritmi a tutte le rappresentazioni.  |
| Grande quantità di dati, ha raggiunto il livello della fisica. | Da sempre grandi quantità di dati (LHC). | 3. amount and quality of data | Non ci sono grandissime differenze da queste, eccetto casi estremamente specifici (GW, eventi rari). In generale di background abbiamo sempre tantissimi dati, mentre per gli eventi un po meno. Sulla **qualità** dei dati invece ci sono dei discorsi da fare. Nel caso sperimentale della SML, è invece un punto centrale, essendo l'esperimento controllato fin dall'inizio, e quindi noto in molti dei suoi dettagli. Un discorso importante che deve essere fatto è quello della possibilità di simulare molti dei processi. Questo permette quindi di possedere con estrema facilità molti dati puliti, generati con Monte Carlo, su cui poi 'allenare' con fedeltà gli algoritmi. Questo **non è analogamente vero per GPML**, dove non sempre è possibile generare in modo casuale processi (foto di cani o gatti) su cui allenare la rete, soffrendo di un bias iniziale (se gnenero solo su un certo tipo di cani, perché il computer conosce solo quello, allora non scoprirò mai che il mio algoritmo è allenato con un bias) |
| Non essendo estremeamente definite le FEATURES e le caratteristiche che si cercano sui dati che si considerano, e quindi le performances non possono essere estremamente alte. | Le features sono espresse dal modello fisico, quindi si ha che le performances sono assolutamente più solide. | 4. performance |  |

## Rappresentazioni: qualche dettaglio in più

:::{margin}
È importante osservare che per quanto non esplicitiamo, né espliciteremo mai, la forma analitica, queste funzioni sono in realtà funzioni in cui teoricamente il modello è noto. Non sto considerando perciò funzioni senza modello!
:::

Considerato un esperimento supervisionato, per cui dato una variabile $x$ (che può essere un vettore, o anche in un problema semplificato uno scalare) vogliamo ottenere 

$$ 
    x \to y = f(x). 
$$

```{figure} ../../images/reparametrization.png
---
name: reparametrization
figclass: margin
---
Riparametrizzando da coordinate $(x,y)$ a coordinate $(r, \theta)$ possiamo semplificare molto la classificazione. 
```

Chiamiamo FEATURES il vettore delle $x$, e TARGET il vettore delle $y$. La dimensionalità del problema è detta $n$. La rappresentazione consiste nello scegliere la base $\{x_1,x_2,\dots,x_n\}$ su cui intendiamo invece scomporre le $\underline x$. La scelta di una buona rappresentazione è fondamentale perché potrebbe permettere di semplificare la forma della $f(\underline x)$ e quindi avere più chiarezza su come esprimerla. Un esempio è dato da considerare un insieme di dati che hanno distribuzione che sul piano è scomodo, ma appena passiamo in coordinate radiali è semplificato (come si può vedere in {numref}`reparametrization`).

La caratteristica che distingue SML da GPML è che in SML le features sono in genere suggerite dalla fisica che si sta studiando, mentre nel caso di GPML sono più complicate. 

:::{admonition} Features in GPML.
:class: tip
Per esempio pensando solo al dover dare la differenza cane vs gatto: si parla allora di _representation_ o _feature_ learning ({numref}`featurelearning`). 
:::

```{figure} ../../images/Feature_Learning_Diagram.png
---
name: featurelearning
---
Diagram of the feature learning paradigm in machine learning for application to downstream tasks, which can be applied to either raw data such as images or text, or to an initial set of features for the data. (wikipedia)
```

:::{admonition} Confounding features 
:class: warning
Esistono, in entrambi i casi, quelle che sono invece definite le _confounding features_, ovvero FEATURES che non vorremmo in realtà tenere in considerazione, ma che, un po' come un background sperimentale, rimangono invece e possono essere problematiche per l'algoritmo. Queste sono spesso associate però a proprietà di simmetria di un sistema, come per esempio una invarianza per traslazione: considerata una immagine, posso aspettarmi che una certa feature sia invece spostata dalla posizione in cui mi aspetti che sia. In questi casi non è il sistema che è invariante per simmetria, ma è la domanda che mi sto ponendo che deve invece esserlo. 
:::

## Learning strategies and task

(22 febbraio 2024)

Vogliamo portci la domanda di capire *quali* sono i problemi che vogliamo risolvere, ovvero i _task_, in funzione dei dati che sono coinvolti in questa particolare task. 

:::{admonition} Strategie, inizio
:class: tip
In ambito di ML ci sono due macro-categorie di strategie. 
 1. __supervised__. In questo caso i dati sono $(\textbf x, \textbf y)$, e obiettivo sarà individuare la fuinzione $H(\textbf x) = \textbf y$, ovvero la funzione che le lega. 
 2. __unsupervised__. I dati qua sono sono invece dati solo da $\textbf x$.
:::

Ma quali sono le _task_ che possiamo avere con il ML?

:::{margin}
_Classification_ e _regression_ sono di fatto praticamente uguali, ma differiscono solo all'ultimo layer, dove divergono nell'ultimo layer, associando i label. 
:::

 - __Regression__. Questi rappresentano i tipi di problemi principali che possiamo avere in ML. Questo corrisponde a determinare

   $$
     f(\cdot, \theta): \mathbb X \to \mathbb Y
   $$

   Il vettore $x$ viene chiamato _features vector_, che noi chiameremo invece INPUT. Il vettore Y invece lo definiremo OUTPUT. La funzione potrebbe avere diverse rappresentazioni, in particolare alcune sono mappe rappresentabili come

   $$
     f: \mathbb R^n \to \mathbb R, \quad f: [[a,b],[c,d],\dots] \to [y_0,y]
   $$ 
 - __Classification__. Questo problema può essere formulata di nuovo come

   $$
       f(\cdot, \theta): \mathbb X \to \mathbb L
   $$

   dove indichiamo il set di arrivo $\mathbb L$ come Label. Il set di arrivo può essere visto come per la regressione, ma discreto.

   Esempio molto facile di classificatori è il _binary classificator_. Questo infatti è dato da una funzione $f:\{0,1\}\to\{0,1\}$, e rappresenta gli operatori AND, OR, XOR.

   La seconda classe di classificatori è data dai _multi-class classificators_. Questi sono per esempio centrali in ambito della fisica delle particelle.

   Chi decide queste __label__ è in genere l'uomo, per le applicazioni di GPML, ma per le applicazioni invece SML si ha che spesso è nota la label per costruzione (come nel caso di dati sintetici che possono essere simulati fedelmente).

:::{admonition} Se mancano i dati...
:class: tip
Ci sono casi in cui non tutti i dati sono noti, o mancanti, si parla di _classification with missing data (supervised)_ e _data imputation (unsupervised)_.
:::

### Time series forecasting

Poter predirre l'evoluzione si serie temporali è stato un problema centrale, sia in fisica che anche in molti altri ambiti. Si può affrontare come un problema di regressione: date le condizioni $x$ ora, voglio determinare $f(x) = y$, ovvero le condizioni in un futuro (immediato). Un altro modo di vedere il problema è quello invece di studiare $f(y_{t'\ll t_0}) = y_{t_0}$, ovvero andare a vedere il passato (anche molto lontano) per vedere se esistono delle periodicità o dei modelli che prevedono quello che è ora ($t_0$). 

Spesso quello che si vuole ottenere è una funzione, un modello che tenga conto di entrambe queste visioni, e quindi avere una funzione 

$$
    f(x, y_{t'\ll t}) = y_{t}.
$$

### Structured output (supervised) and generation and sampling (unsupervised)

__Structured output__. Dato un insieme di features, l'outpunt non è semplicemente un singolo dato, ma un insieme, un grafo, una frase, un simbolo, ... Lo spazio $\mathbb Y$ allora diventa di dimensionalità (cardinalità) molto maggiore. 

Lo structured output è spesso utilizzato per mappare diverse rappresentazioni tra di loro
 - Speec-to-text translation
 - Text comprehension

Le LLM (Large Language Models) sono sia modelli generativi (_generation and sampling_, che può essere sia _supervised_ che _unsupervised_) che structured output. Non sono uno o l'altro, ma a seconda del task sono una buona mistura delle due. I modelli DEEPFAKE sono spesso _generatives_ che generano a partire da _noise_ una immagine. 

I modelli generativi, per quanto siano invece molto utilizzati in ambito industriale e in GPML, nell'ambito della scienza possono risultare molto pericolosi. Una scoperta a $5\sigma$ con un modello generativo è denotata da una incertezza che non possiamo permetterci. 

### Unsupervised tasks

__Density estimation__. Supponiamo di conoscere dei dati $\textbf x$, e vogliamo ottenerne la distribuzione in probabilità $p(\textbf x | \boldsymbol \theta)$. Il vettore delle $\theta$ è l'insieme dei parametri che caratterizzano la PDF (per la gaussiana sono per esempio $\{\boldsymbol \mu, \xi\}$, con $\boldsymbol \mu$ il vettore dei valori medi, e $\xi$ la matrice di covarianza).  

__Anomaly detection__. Spesso confusa con la classificazione a due categorie (sig/bkg). Non è così, ma invece è un modello unsupervised, e consiste invece nel voler trovare un __outlier__ rispetto ad una certa popolazione. Dal punto di vista scientifico è utile
 - permette di trovare velocemente degli outlier in una teoria.
 - data quality monitor: se osservo in una raccolta dati dei valori _strani_ posso ipotizzare che questi siano _sbagliati_.
 - possono essere utilizzati in algoritmi di trigger online, che permetta di scartare eventi che non sono utili alla fisica che si vuole fare.

__Denoising__. Considerato un segnale, che può essere per esempio segnale astrofisico, o segnale di altro genere, si vuole rimuovere quello che è ritenuto __fondo__ o __rumore__.

__Clustering__. È quello che abbiamo fatto nell'esempio della {numref}`reparametrization`. Si tratta di voler dividere dei dati iniziali rispetto ad alcune caratteristiche, ovvero dividere i dati rispetto ad alcuni _clusters_. 


## Learning or statistical inference?

Formulare un modello, prendere dei dati, stimare i parametri, e poi utilizzare il modello per vedere come si comporta su altri dati sono caratteristiche condivise tra i due casi, _ML_ e _statistical inference_. In particolare osserviamo che condividono alcune caratteristiche comuni 

$$
    \text{fit (or parameter estimation)} \leftrightarrow \text{training}
$$
$$
    \text{goodness of fit test} \leftrightarrow \text{model validation}
$$

L'inferenza statistica tradizionale fornisce dei dati (che chiamiamo train data), che, considerato un modello $f(\bf x, \boldsymbol \theta) = \bf y$, mi permette di stimare i parametri $\boldsymbol \theta$. Generalmente quindi un test GoF lo vado a fare su due modelli. 

Lo scopo del ML non è quello di capire se i miei dati mi danno dei parametri buoni, quano invece di dare un modello che mi permette di essere riutilizzato. Si vede quindi la difficoltà dell'overtraining. Quindi diventa fondamentale anche il passaggio di model validation, che non è tra modelli diversi, ma tra modelli uguali, con differenti livelli di predizione del modello, e procedo fintanto che il livello di predizione del modello e dei dati di validation sono in buon accordo.

:::{admonition} Learning
:class: important
La procedura del learning consiste nel costruire un modello statistico, di cui conosco appena alcune proprietà; fittare il modello ai dati; calcolare la stessa funzione su un altro set di dati; se mi soddisfa ottimo, altrimenti modifico dei dati e ripeto il processo da capo. 

In un nuovo linguaggio
 - training, on some hyperparameters
 - model validation
 - calibration
:::
















