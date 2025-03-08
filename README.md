# Nome del Progetto

Breve descrizione del progetto, cosa fa e quali problemi risolve.

## Struttura del Progetto

Il progetto è organizzato in diverse directory e file, alcuni dei quali sono ignorati dal controllo di versione per mantenere il repository pulito e gestibile. Di seguito è riportata una descrizione delle principali sezioni del file `.gitignore`:

### File di byte-compilati Python
- `__pycache__/`: Directory che contiene file di cache generati da Python.
- `*.py[cod]`: File di bytecode Python.
- `*$py.class`: File di classe Python.

### Distribuzione / Packaging
- `dist/`, `build/`, `*.egg-info/`: Directory e file generati durante il processo di packaging e distribuzione.

### Ambiente Virtuale
- `venv/`, `env/`, `ENV/`, `phi4_env/`: Directory degli ambienti virtuali Python.
- `png/`: (Se applicabile, descrivere l'uso di questa directory).

### File di Cache
- `.cache/`, `.pytest_cache/`, `.coverage`, `htmlcov/`: File e directory di cache generati durante l'esecuzione dei test e la copertura del codice.

### File di Log
- `*.log`: File di log generati dall'applicazione.

### File di Ambiente Locale
- `.env`, `.env.local`, ecc.: File di configurazione dell'ambiente locale.

### File IDE e Editor
- `.idea/`, `.vscode/`, `*.swp`, `*.swo`, `.DS_Store`: File e directory specifici dell'IDE o dell'editor.

### File Specifici di Sistema Operativo
- `Thumbs.db`: File generati dal sistema operativo.

### File di Dati
- `*.csv`, `*.dat`, `*.db`, `*.sqlite`: File di dati utilizzati dall'applicazione.

### Directory di Output
- `output/`, `downloads/`, `uploads/`: Directory per file di output e trasferimenti.

### Notebook Jupyter
- `.ipynb_checkpoints`: Directory di checkpoint per i notebook Jupyter.

### File Temporanei
- `tmp/`, `temp/`: Directory per file temporanei.

## Istruzioni per l'Installazione

1. Clonare il repository:
   ```bash
   git clone <url-del-repository>
   ```
2. Creare un ambiente virtuale:
   ```bash
   python -m venv venv
   ```
3. Attivare l'ambiente virtuale:
   - Su Windows:
     ```bash
     .\\venv\\Scripts\\activate
     ```
   - Su macOS e Linux:
     ```bash
     source venv/bin/activate
     ```
4. Installare le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

## Utilizzo


`back-end.py` è un'applicazione server che utilizza la libreria Transformers per eseguire inferenze su modelli di linguaggio. Ecco una panoramica delle sue funzionalità principali:

- **Impostazione delle variabili d'ambiente**: Configura variabili d'ambiente critiche per ottimizzare le prestazioni e gestire le risorse, come la disabilitazione di moduli problematici e l'uso di offload su disco.
- **Logging avanzato**: Utilizza il modulo `logging` per fornire informazioni dettagliate sullo stato del sistema e delle risorse.
- **Monitoraggio delle risorse**: Implementa funzioni per monitorare l'utilizzo della memoria RAM, GPU e spazio su disco, loggando periodicamente le statistiche.
- **Ottimizzazioni CUDA**: Applica ottimizzazioni specifiche per CUDA per migliorare le prestazioni durante l'inferenza.
- **Threading e parallelismo**: Utilizza thread per monitorare le risorse e gestire richieste concorrenti.

### scan_analyze_png.py

`scan_analyze_png.py` è un'applicazione che analizza immagini in una directory specifica e utilizza un'API Gradio per estrarre dati testuali e numerici. Ecco le sue funzionalità principali:

- **Scansione della directory**: Cerca immagini nella directory `png` e le prepara per l'analisi.
- **Interazione con API Gradio**: Invia le immagini a un server Gradio per l'analisi e riceve risposte in formato JSON.
- **Gestione delle risposte**: Elabora le risposte ricevute dall'API, assicurandosi che siano in formato JSON valido.
- **Salvataggio dei risultati**: Salva le descrizioni delle immagini in file JSON nella directory `risultati`.
- **Debug e logging**: Fornisce messaggi di debug dettagliati per monitorare il flusso di lavoro e risolvere eventuali problemi.

## Licenza

Questo progetto è distribuito sotto la [Nome della Licenza] (inserisci il link alla licenza se disponibile).

## Autore

- Nome dell'Autore: Roberto Navoni
- Email: r.navoni@radionav.it

## Azienda

- Nome dell'Azienda: Laser Navigation 
- Sito Web: http://www.lasernavigation.it 