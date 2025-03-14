FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Imposta variabili d'ambiente per ridurre interazioni durante l'installazione
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Configura esplicitamente le variabili d'ambiente per i compilatori
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

# Installa le dipendenze di sistema
RUN apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    gcc \
    g++ \
    cmake \
    ninja-build \
    libcudnn8 \
    cuda-nvcc-12-1 \
    cuda-libraries-dev-12-1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verifica che il compilatore C sia disponibile
RUN gcc --version

# Crea directory per l'applicazione
WORKDIR /app

# Copia solo i file necessari per installare i requisiti
COPY requirements.txt ./

# Installa le dipendenze Python
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir hf_transfer

# Crea directory specifiche per l'applicazione
RUN mkdir -p /app/example_images /app/output /app/png /app/risultati

# Copia il codice dell'applicazione
COPY app.py ./
COPY scan_analyze_png.py ./
COPY docker/entrypoint.sh ./

# Rendi eseguibile lo script di avvio
RUN chmod +x entrypoint.sh

# Esponi la porta per Gradio
EXPOSE 7860

# Imposta le variabili d'ambiente per le ottimizzazioni CUDA
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV ACCELERATE_USE_DISK_OFFLOAD=1
ENV OMP_NUM_THREADS=4
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32,garbage_collection_threshold:0.5"

# Volume per i dati persistenti
VOLUME ["/app/png", "/app/risultati"]

# Esegui l'applicazione al lancio del container
ENTRYPOINT ["/app/entrypoint.sh"] 