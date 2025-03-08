# SophyAI-Agent-Qwen2.5VL-7B Docker

Questa directory contiene i file necessari per eseguire l'applicazione SophyAI-Agent-Qwen2.5VL-7B in un container Docker.

## Contenuto della directory

- `Dockerfile`: Definizione dell'immagine Docker basata su CUDA 12.1
- `docker-compose.yml`: Configurazione per avviare facilmente i servizi
- `entrypoint.sh`: Script di avvio del container
- `.dockerignore`: File che specifica quali file escludere durante la build

## Requisiti

- Docker
- NVIDIA GPU con almeno 12GB di VRAM (consigliati 24GB+)
- NVIDIA Docker Runtime (nvidia-container-toolkit)

## Quick Start

Dalla directory principale del progetto (non questa directory docker), puoi avviare l'applicazione in due modi:

### 1. Utilizzando Docker Compose (consigliato)

```bash
# Avvia il server Gradio
docker-compose -f docker/docker-compose.yml up sophyai-server

# Oppure avvia l'elaborazione batch delle immagini
docker-compose -f docker/docker-compose.yml up sophyai-process
```

### 2. Utilizzando Docker direttamente

```bash
# Costruisci l'immagine
docker build -t sophyai-agent:qwen2.5vl-7b -f docker/Dockerfile .

# Avvia il server Gradio
docker run --gpus all -it --rm  -p 7860:7860  -v png:/app/png -v risultati:/app/risultati sophyai-agent:qwen2.5vl-7b server

# Avvia il server Gradio con la cache del modello in locale non all'interno del docker
docker run --gpus all -it --rm  -v model:/data -p 7860:7860  -v png:/app/png -v risultati:/app/risultati sophyai-agent:qwen2.5vl-7b server

# Oppure avvia l'elaborazione batch delle immagini
docker run --gpus all -it --rm \
  -v $(pwd)/png:/app/png \
  -v $(pwd)/risultati:/app/risultati \
  sophyai-agent:qwen2.5vl-7b process
```

## Personalizzazione

### Variabili d'ambiente

Puoi personalizzare il comportamento dell'applicazione utilizzando variabili d'ambiente:

```bash
docker run --gpus all -it --rm \
  -p 7860:7860 \
  -e PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64" \
  -e OMP_NUM_THREADS=8 \
  -v $(pwd)/png:/app/png \
  -v $(pwd)/risultati:/app/risultati \
  sophyai-agent:qwen2.5vl-7b
```

### Personalizzazione del Dockerfile

Se desideri modificare l'immagine Docker, puoi:

1. Modificare il file `Dockerfile` per aggiungere o rimuovere pacchetti
2. Modificare `entrypoint.sh` per cambiare il comportamento di avvio
3. Aggiornare `docker-compose.yml` per configurare i servizi

## Troubleshooting

- **Errore "CUDA initialization failed"**: Verifica che i driver NVIDIA siano installati e che nvidia-container-toolkit sia configurato correttamente.
- **Errore "OOM" (Out Of Memory)**: Prova con immagini più piccole o riduci la risoluzione di input modificando il codice.
- **Container lento**: Verifica che la GPU sia accessibile al container e che non ci siano altri processi che utilizzano la GPU.

## Pulizia

```bash
# Rimuovi i container creati
docker-compose -f docker/docker-compose.yml down

# Rimuovi l'immagine Docker
docker rmi sophyai-agent:qwen2.5vl-7b
``` 