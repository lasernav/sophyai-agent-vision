---
title: Qwen2.5 VL 7B
emoji: 🔥
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 5.15.0
app_file: app.py
pinned: true
license: creativeml-openrail-m
short_description: Qwen2.5-VL-7B-Instruct
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# SophyAI-Agent-Qwen2.5VL-7B

Agente AI basato su Qwen2.5-VL-7B per l'estrazione di dati testuali e numerici dalle immagini, ottimizzato per l'inferenza rapida.

## Funzionalità

- **Estrazione Dati**: Estrae dati testuali e numerici da immagini
- **Analisi Veloce**: Ottimizzata per velocità di inferenza (fino a 5-7 token/s)
- **Modalità Duale**: Supporta interfaccia web interattiva o elaborazione batch di immagini
- **Dockerizzata**: Facilmente distribuibile tramite container Docker su qualsiasi piattaforma

## Utilizzo Standard

### Esecuzione locale

```bash
# Rendi eseguibile lo script
chmod +x run_local.sh

# Avvia il server Gradio
./run_local.sh server

# Oppure elabora le immagini in batch
./run_local.sh process
```

### Esecuzione con Docker

Per utilizzare l'applicazione con Docker, vedere la [documentazione Docker](docker/README.md) nella directory `docker/`.

## Struttura del Progetto

```
SophyAI-Agent-Qwen2.5VL-7B/
├── app.py                   # Server Gradio con interfaccia web
├── scan_analyze_png.py      # Script per analisi batch delle immagini
├── run_local.sh             # Script per esecuzione locale
├── requirements.txt         # Dipendenze Python
├── png/                     # Directory per le immagini da analizzare
├── risultati/               # Directory per i risultati dell'analisi
└── docker/                  # Directory con configurazione Docker
    ├── Dockerfile           # Definizione dell'immagine Docker
    ├── docker-compose.yml   # Configurazione dei servizi Docker
    ├── entrypoint.sh        # Script di avvio per il container
    └── README.md            # Documentazione specifica per Docker
```

## Requisiti

### Per utilizzo locale:
- Python 3.10+
- NVIDIA GPU con almeno 12GB di VRAM (consigliati 24GB+)
- CUDA 11.8+

### Per utilizzo Docker:
- Docker
- NVIDIA GPU con almeno 12GB di VRAM (consigliati 24GB+)
- NVIDIA Docker Runtime (nvidia-container-toolkit)

## Installazione

```bash
# Clona il repository (se applicabile)
# git clone https://github.com/tuouser/SophyAI-Agent-Qwen2.5VL-7B.git
# cd SophyAI-Agent-Qwen2.5VL-7B

# Crea ambiente virtuale e installa dipendenze
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Note su performance e ottimizzazioni

L'applicazione è stata ottimizzata per:

- Velocità di generazione dei token (circa 5-7 token/s su GPU adeguata)
- Efficienza nell'elaborazione delle immagini
- Riduzione della risoluzione per bilanciare qualità e performance

## Troubleshooting

- **Errore "CUDA initialization failed"**: Verifica che i driver NVIDIA siano installati correttamente.
- **Errore "OOM" (Out Of Memory)**: Prova con immagini più piccole o riduci la risoluzione di input modificando il codice.
- **Generazione lenta**: Verifica che altre applicazioni non stiano utilizzando la GPU.

## License
Questo progetto utilizza il modello Qwen2.5-VL-7B-Instruct soggetto a termini di licenza CreativeML OpenRAIL-M.

Da condividere in github con stecon2