#!/usr/bin/env python3
# Importazione iniziale solo per os, per impostare le variabili d'ambiente PRIMA di qualsiasi altro import
import os

# IMPORTANTE: Disabilitazione del modulo hf_transfer che causa errori
# Questa deve essere la PRIMA istruzione nel file, prima di qualsiasi altro import
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import gradio as gr
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TextIteratorStreamer, BitsAndBytesConfig
from transformers.image_utils import load_image
from transformers.generation.stopping_criteria import StoppingCriteriaList, StoppingCriteria
from threading import Thread, Lock
import time
import torch
import spaces
import threading
import subprocess
import psutil
import random
import gc
import tempfile
import shutil
from accelerate import init_empty_weights
from accelerate.utils import get_balanced_memory
import bitsandbytes as bnb
from concurrent.futures import ThreadPoolExecutor
import logging
import uuid
from queue import Empty  # Aggiungiamo l'importazione di Empty per gestire il timeout della coda

# Creazione di una classe personalizzata per sostituire MaxTimeStoppingCriteria mancante
class MaxTimeStoppingCriteria(StoppingCriteria):
    """
    Classe personalizzata che implementa un criterio di stop basato sul tempo massimo di generazione.
    Sostituisce la classe mancante nella libreria Transformers.
    """
    def __init__(self, max_time: float):
        self.max_time = max_time
        self.start_time = time.time()
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        elapsed = time.time() - self.start_time
        return elapsed > self.max_time
    
    def reset(self):
        self.start_time = time.time()

# Configurazione logging avanzato
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Impostiamo altre variabili d'ambiente PRIMA di importare o inizializzare qualsiasi modello
# Aggiungiamo un log esplicito per la disabilitazione di hf_transfer
logger.info("IMPOSTAZIONE VARIABILI D'AMBIENTE CRITICHE")
logger.info("Disabilitazione hf_transfer: HF_HUB_ENABLE_HF_TRANSFER=0")
# Ridondante, ma per sicurezza riconfermiamo la disabilitazione
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  
os.environ["ACCELERATE_USE_DISK_OFFLOAD"] = "1"  # Forza l'uso dell'offload su disco
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.5"
os.environ["OMP_NUM_THREADS"] = "4"  # Limita il numero di thread per le operazioni CPU

# Imposta la cache di Hugging Face in una directory esterna persistente
# Questo permette di riutilizzare i modelli scaricati tra riavvi del container
os.environ["TRANSFORMERS_CACHE"] = "/data/models_cache"
os.environ["HF_HOME"] = "/data/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/data/datasets_cache"

# Creiamo le directory persistenti se non esistono
for directory in ["/data/models_cache", "/data/huggingface", "/data/datasets_cache", "/data/offload"]:
    try:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory persistente creata/verificata: {directory}")
    except Exception as e:
        logger.warning(f"Impossibile creare directory {directory}: {e}")
        logger.warning("Potrebbe essere necessario avviare il container con un volume esterno montato su /data")

# Variabili globali per il monitoraggio
last_debug_time = time.time()
debug_interval = 20  # Aumentato da 10 a 20 secondi per ridurre overhead di log
active_requests = {}
request_lock = Lock()
memory_stats = {
    "ram": {"total": 0, "used": 0, "percent": 0},
    "gpu": {},
    "offload": 0,
    "timestamp": time.time()
}
stats_lock = Lock()

# Utilizziamo una directory persistente per l'offload su disco invece di una temporanea
offload_directory = "/data/offload"
logger.info(f"Directory persistente per offload: {offload_directory}")

def get_dir_size(path):
    """
    Calcola la dimensione totale di una directory in GB
    """
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size / (1024**3)  # Converti in GB
    except Exception as e:
        logger.error(f"Errore nel calcolare la dimensione della directory: {e}")
        return 0

def update_memory_stats():
    """
    Aggiorna le statistiche di memoria nel dizionario globale
    """
    global memory_stats
    
    with stats_lock:
        # Monitora l'utilizzo della RAM
        ram = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        memory_stats["ram"] = {
            "total": ram.total / (1024**3),
            "used": ram.used / (1024**3),
            "percent": ram.percent,
            "available": ram.available / (1024**3),
            "swap_total": swap.total / (1024**3),
            "swap_used": swap.used / (1024**3),
            "swap_percent": swap.percent
        }
        
        # Monitoraggio memoria del processo Python corrente
        process = psutil.Process(os.getpid())
        memory_stats["process"] = {
            "memory_rss": process.memory_info().rss / (1024**3),
            "cpu_percent": process.cpu_percent()
        }
        
        # Monitora l'uso dello spazio su disco per l'offload
        if os.path.exists(offload_directory):
            memory_stats["offload"] = get_dir_size(offload_directory)
        
        # GPU info
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,memory.free', '--format=csv,noheader'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                memory_stats["gpu"] = {}
                for line in result.stdout.strip().split('\n'):
                    if ', ' not in line:
                        continue
                    values = line.split(', ')
                    index = values[0]
                    used = values[1]
                    total = values[2]
                    free = values[3]
                    
                    # Estrai i numeri dalle stringhe (es. "10 MiB" -> 10)
                    used_val = float(used.split()[0])
                    total_val = float(total.split()[0])
                    free_val = float(free.split()[0])
                    
                    memory_stats["gpu"][index] = {
                        "used": used,
                        "total": total,
                        "free": free,
                        "used_val": used_val,
                        "total_val": total_val,
                        "free_val": free_val,
                        "percent": (used_val / total_val) * 100 if total_val > 0 else 0
                    }
        except Exception as e:
            logger.error(f"Errore nell'aggiornamento delle informazioni GPU: {e}")
        
        memory_stats["timestamp"] = time.time()
        
def log_memory_stats():
    """
    Logga le statistiche di memoria in modo formattato
    """
    global memory_stats
    
    with stats_lock:
        stats = memory_stats
        
        logger.info("\n" + "="*50)
        logger.info("STATO MEMORIA DEL SISTEMA:")
        logger.info(f"  RAM Totale:     {stats['ram']['total']:.2f} GB")
        logger.info(f"  RAM Utilizzata: {stats['ram']['used']:.2f} GB ({stats['ram']['percent']}%)")
        logger.info(f"  RAM Libera:     {stats['ram']['available']:.2f} GB ({100-stats['ram']['percent']}%)")
        logger.info(f"  SWAP Totale:    {stats['ram']['swap_total']:.2f} GB")
        logger.info(f"  SWAP Utilizzata:{stats['ram']['swap_used']:.2f} GB ({stats['ram']['swap_percent']}%)")
        logger.info(f"  Memoria usata da questo processo Python: {stats['process']['memory_rss']:.2f} GB")
        logger.info(f"  Memoria allocata su disco (offload): {stats['offload']:.2f} GB")
        
        if "gpu" in stats:
            logger.info("\nSTATO GPU:")
            for gpu_id, gpu_stats in stats["gpu"].items():
                logger.info(f"  GPU {gpu_id}: {gpu_stats['used']}/{gpu_stats['total']} utilizzata ({gpu_stats['percent']:.1f}%), {gpu_stats['free']} libera")
        
        # Informazioni richieste attive
        if active_requests:
            logger.info("\nRICHIESTE ATTIVE:")
            for req_id, req_info in active_requests.items():
                logger.info(f"  ID: {req_id[:8]}... | Iniziata: {req_info['start_time']:.1f}s fa | Testo: '{req_info['text'][:30]}...' | Immagini: {len(req_info['files'])}")
        logger.info("="*50)

def monitor_gpu():
    """
    Funzione per monitorare l'utilizzo della memoria GPU e RAM ogni 10 secondi
    """
    global last_debug_time
    
    while True:
        try:
            current_time = time.time()
            if current_time - last_debug_time >= debug_interval:
                # Aggiorna le statistiche
                update_memory_stats()
                # Logga le statistiche
                log_memory_stats()
                # Aggiorna il timestamp dell'ultimo debug
                last_debug_time = current_time
            
        except Exception as e:
            logger.error(f"Errore nel monitoraggio delle risorse: {e}")
            
        time.sleep(2)  # Controllo più frequente, ma log meno frequente

# Avvia il thread di monitoraggio GPU
monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
monitor_thread.start()

# Ottimizzazioni per CUDA
logger.info("Applicazione ottimizzazioni CUDA per migliorare performance")
# Forziamo CUDA a ottimizzare i kernel per il nostro hardware
torch.backends.cudnn.benchmark = True
# Permettiamo calcoli più veloci (meno precisi ma adeguati per inferenza)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Permette operazioni a precisione ridotta per matmul
torch.backends.cuda.flash_sdp_enabled = True  # Abilita flash attention scaled dot product

# Pre-allocazione memoria CUDA (evita frammentazione)
if torch.cuda.is_available():
    # Alloca un tensore grande per pre-allocare memoria e prevenire frammentazione
    dummy_tensor = torch.zeros(1024, 1024, 12, dtype=torch.float16, device="cuda")
    del dummy_tensor
    torch.cuda.empty_cache()

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"  # Versione più piccola e gestibile
#MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
# Creiamo una cartella specifica per il disk offload all'interno della directory persistente
disk_offload_dir = os.path.join(offload_directory, "disk_offload")
os.makedirs(disk_offload_dir, exist_ok=True)
logger.info(f"Directory persistente per disk offload creata: {disk_offload_dir}")

# Forzare offload di determinati layer
offload_layers = [
    "language_model.model.layers.20",
    "language_model.model.layers.21",
    "language_model.model.layers.22",
    "language_model.model.layers.23",
    "language_model.model.layers.24",
    "language_model.model.layers.25",
    "language_model.model.layers.26",
    "language_model.model.layers.27",
    "language_model.model.layers.28",
    "language_model.model.layers.29",
    "language_model.model.layers.30",
    "language_model.model.layers.31",
]

# Configurazione per l'offload manuale
logger.info(f"Configurando il modello per l'offload manuale")

# Funzione per forzare i pesi su disco
def force_weights_to_disk(model_component, directory, component_name):
    try:
        if hasattr(model_component, "state_dict"):
            # Creiamo un percorso per il file
            clean_name = component_name.replace(".", "_").replace("/", "_")
            file_path = os.path.join(directory, f"{clean_name}.pt")
            
            # Salviamo lo stato del componente su disco
            torch.save(model_component.state_dict(), file_path)
            logger.info(f"Salvato componente {component_name} su disco: {file_path}")
            
            # Forziamo garbage collection
            torch.cuda.empty_cache()
            gc.collect()
            return True
    except Exception as e:
        logger.warning(f"Impossibile salvare {component_name} su disco: {e}")
    return False

# Configurazione per quantizzazione a 4-bit
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Configurazione semplificata per il modello
logger.info("Caricamento del modello con configurazione semplificata")
try:
    # Verifica preventiva che hf_transfer sia effettivamente disabilitato
    logger.info(f"Verifica variabile d'ambiente: HF_HUB_ENABLE_HF_TRANSFER={os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', 'non impostata')}")
    
    # Disattiva esplicitamente l'uso di hf_transfer anche nelle API di huggingface_hub
    try:
        import huggingface_hub
        huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = False
        logger.info("Disabilitato hf_transfer anche nelle costanti di huggingface_hub")
    except Exception as e:
        logger.warning(f"Impossibile disabilitare hf_transfer nelle costanti: {e}")
    
    logger.info(f"Caricamento processor dal modello {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True,
        use_auth_token=False,
        local_files_only=False,
        resume_download=True,
        force_download=False,
        use_fast=True,
        token=None,  # Nessun token di autenticazione
        cache_dir="/data/models_cache"  # Utilizziamo la directory persistente
    )
    
    logger.info(f"Caricamento modello {MODEL_ID} con quantizzazione 4-bit")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",  # Usiamo "auto" invece di "balanced" per lasciare che HF ottimizzi
        offload_folder=offload_directory,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        use_auth_token=False,
        local_files_only=False,
        resume_download=True,
        force_download=False,
        use_safetensors=True,  # Preferisce versioni safetensors quando disponibili
        cache_dir="/data/models_cache"  # Utilizziamo la directory persistente
    ).eval()
    
    logger.info("Modello caricato con successo!")
except Exception as e:
    logger.error(f"ERRORE durante il caricamento del modello: {str(e)}")
    logger.error("Tentativo con configurazione alternativa...")
    try:
        # Disattiva esplicitamente qualsiasi uso di hf_transfer anche nelle funzioni di huggingface_hub
        try:
            from huggingface_hub import file_download
            # Proviamo a rendere innocua la funzione http_get che causa problemi
            original_http_get = file_download.http_get
            def safe_http_get(*args, **kwargs):
                # Rimuoviamo hf_transfer dai kwargs
                if 'use_hf_transfer' in kwargs:
                    kwargs['use_hf_transfer'] = False
                return original_http_get(*args, **kwargs)
            file_download.http_get = safe_http_get
            logger.info("Patched huggingface_hub.file_download.http_get per disabilitare hf_transfer")
        except Exception as patch_error:
            logger.warning(f"Impossibile patchare http_get: {patch_error}")
        
        # Prova con configurazione alternativa e parametri più semplici
        logger.info("Caricamento con configurazione minimale (senza accelerazioni)")
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir="/data/models_cache")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            cache_dir="/data/models_cache"
        ).eval()
        logger.info("Modello caricato con configurazione alternativa!")
    except Exception as e2:
        logger.error(f"ERRORE CRITICO: Impossibile caricare il modello: {str(e2)}")
        # Proviamo un'ultima strategia più aggressiva
        try:
            logger.warning("Tentativo con strategia di fallback estrema...")
            # Forziamo il download cache-only, senza richiedere file remoti
            from huggingface_hub import snapshot_download
            logger.info("Download locale del modello con snapshot_download")
            cache_dir = snapshot_download(
                MODEL_ID, 
                local_files_only=False,
                resume_download=True,
                use_auth_token=False,
                token=None,
                force_download=False,
                local_dir="/data/model_cache"  # Utilizziamo la directory persistente
            )
            logger.info(f"Modello scaricato in: {cache_dir}")
            
            # Ora carica dal percorso locale
            processor = AutoProcessor.from_pretrained(
                cache_dir,
                trust_remote_code=True,
                local_files_only=True  # Usa SOLO i file locali
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                cache_dir,
                trust_remote_code=True,
                device_map="auto",
                local_files_only=True,  # Usa SOLO i file locali
                low_cpu_mem_usage=True
            ).eval()
            logger.info("Modello caricato con strategia di fallback!")
        except Exception as e3:
            logger.error(f"Tutti i tentativi di caricamento falliti. Ultimo errore: {str(e3)}")
            raise RuntimeError(f"Impossibile caricare il modello dopo molteplici tentativi. Errori: \n1) {str(e)}\n2) {str(e2)}\n3) {str(e3)}")

# Forziamo un'ultima pulizia completa prima dell'inferenza
torch.cuda.empty_cache()
gc.collect()

# Configurazione aggiuntiva per l'uso della memoria
torch.cuda.set_per_process_memory_fraction(0.9, 0)  # 90% della prima GPU da 24GB
torch.cuda.set_per_process_memory_fraction(0.9, 1)  # 90% della seconda GPU da 24GB

# Disabilitiamo il calcolo dei gradienti per risparmiare memoria
torch.set_grad_enabled(False)

# Impostazione per convertire le immagini in bassa risoluzione prima dell'elaborazione
def preprocess_image_low_res(image_path):
    try:
        from PIL import Image
        import numpy as np
        import io
        
        # Carica l'immagine
        img = Image.open(image_path).convert('RGB')
        
        # Calcola le dimensioni per il ridimensionamento
        width, height = img.size
        max_dim = max(width, height)
        
        # Aumentato alla risoluzione massima supportata dal modello
        target_size = 1440  # Aumentato da 1024 a 1440 per massima qualità disponibile
        if max_dim > target_size:
            scale = target_size / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            logger.info(f"Immagine ridimensionata da {width}x{height} a {new_width}x{new_height}")
        
        # Compressione con qualità massima
        buffer = io.BytesIO()
        # Qualità aumentata al 95% per mantenere il massimo dei dettagli
        img.save(buffer, format="JPEG", quality=95, optimize=True)
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        
        # Calcola il rapporto di compressione
        original_size = os.path.getsize(image_path) / 1024  # KB
        buffer.seek(0, io.SEEK_END)
        compressed_size = buffer.tell() / 1024  # KB
        compression_ratio = (original_size - compressed_size) / original_size * 100
        
        logger.info(f"Compressione immagine: {original_size:.1f}KB → {compressed_size:.1f}KB ({compression_ratio:.1f}% riduzione)")
        
        return compressed_img
    except Exception as e:
        logger.error(f"Errore nel pre-processare l'immagine: {e}")
        return None

# Configurazione adattiva della modalità
TEXT_ONLY_MODE = False  # Inizialmente no
MAX_IMAGE_SIZE_MB = 10.0  # Aumentato da 5MB a 10MB per supportare immagini più grandi

def check_image_size(image_path):
    """Verifica la dimensione dell'immagine e decide se elaborarla"""
    try:
        size_mb = os.path.getsize(image_path) / (1024 * 1024)
        logger.info(f"Dimensione immagine: {size_mb:.2f} MB")
        if size_mb > MAX_IMAGE_SIZE_MB:
            logger.warning(f"Immagine troppo grande ({size_mb:.2f} MB > {MAX_IMAGE_SIZE_MB} MB), saltata")
            return False
        return True
    except Exception:
        return False

@spaces.GPU
def model_inference(input_dict, history):
    """
    Funzione di inferenza principale che elabora le richieste al modello
    """
    # Impostiamo subito l'utilizzo della GPU corretta e applichiamo ottimizzazioni CUDA
    # per massimizzare la velocità di questa specifica generazione
    if torch.cuda.is_available():
        device_id = 0  # Usiamo sempre la prima GPU che è spesso la più veloce
        torch.cuda.set_device(device_id)
        torch.cuda.synchronize()  # Sincronizziamo la GPU prima di iniziare
        
        # Assicuriamo che i kernel sono ottimizzati per questa inferenza
        if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            torch.cuda.amp.autocast(dtype=torch.float16, cache_enabled=True).__enter__()

    # Generazione ID univoco per questa richiesta
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    text = input_dict["text"]
    files = input_dict["files"]
    
    # Registra questa richiesta come attiva
    with request_lock:
        active_requests[request_id] = {
            "text": text,
            "files": files,
            "start_time": start_time,
            "last_update": start_time,
            "status": "starting"
        }
    
    # Log di inizio
    logger.info(f"[{request_id[:8]}] NUOVA RICHIESTA - Testo: '{text[:50]}...' - Files: {len(files)}")
    
    try:
        # Aggiorna stato
        with request_lock:
            active_requests[request_id]["status"] = "processing_input"
            
        # In modalità solo testo, ignoriamo completamente le immagini
        if TEXT_ONLY_MODE:
            files = []
            if "image" in text.lower() or "photo" in text.lower() or "picture" in text.lower():
                logger.info(f"[{request_id[:8]}] Modalità solo testo attiva, immagini ignorate")
                return "Mi dispiace, sono attualmente in modalità solo testo. Non posso elaborare immagini in questo momento."

        # Load images if provided
        images = []
        try:
            if len(files) > 0:
                logger.info(f"[{request_id[:8]}] Preprocessamento di {len(files)} immagini")
                with request_lock:
                    active_requests[request_id]["status"] = "preprocessing_images"
                
                # Preprocessing parallelo delle immagini
                def process_single_image(image_path):
                    logger.info(f"[{request_id[:8]}] Processando immagine: {image_path}")
                    img = preprocess_image_low_res(image_path)
                    return img
                
                with ThreadPoolExecutor(max_workers=2) as executor:
                    processed_images = list(executor.map(process_single_image, files[:2]))  # Limitiamo a 2 immagini
                
                # Filtriamo le immagini non valide (None)
                images = [img for img in processed_images if img is not None]
                logger.info(f"[{request_id[:8]}] Preprocessate {len(images)}/{len(files)} immagini con successo")
                
                # Aggiorna stato e forza l'aggiornamento delle statistiche di memoria
                with request_lock:
                    active_requests[request_id]["status"] = "images_preprocessed"
                    active_requests[request_id]["last_update"] = time.time()
                    
                # Forza aggiornamento debug se necessario
                current_time = time.time()
                if current_time - last_debug_time >= debug_interval:
                    update_memory_stats()
                    log_memory_stats()
                    globals()["last_debug_time"] = current_time
        except Exception as e:
            logger.error(f"[{request_id[:8]}] Errore nel caricamento delle immagini: {e}")
            images = []

        # Validate input
        if text == "" and not images:
            logger.warning(f"[{request_id[:8]}] Input non valido: testo e immagini mancanti")
            gr.Error("Please input a query and optionally image(s).")
            return
        if text == "" and images:
            logger.warning(f"[{request_id[:8]}] Input non valido: testo mancante")
            gr.Error("Please input a text query along with the image(s).")
            return

        # Prepare messages for the model
        logger.info(f"[{request_id[:8]}] Preparazione messaggio per il modello")
        with request_lock:
            active_requests[request_id]["status"] = "preparing_model_input"
            
        messages = [
            {
                "role": "system",
                "content": "Sei un assistente specializzato nell'estrazione dati dalle immagini. Rispondi SEMPRE in brevi frasi. Estrai SOLO dati essenziali in stile JSON ultra-minimale. Formato: valore, tipo, confidenza. EVITA descrizioni, introduzioni e conclusioni. Sii ESTREMAMENTE conciso."
            },
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in images],
                    {"type": "text", "text": "Estrai solo dati essenziali in formato JSON ultra-conciso" if not text or text.strip() == "" else text},
                ],
            }
        ]

        # Ottimizzazioni CUDA aggiuntive per generazione veloce
        if torch.cuda.is_available():
            # Preload CUDA kernels
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Impostazione allocation policy - compatibile con più versioni di PyTorch
            try:
                # Versione più recente di PyTorch
                if hasattr(torch.cuda, 'memory') and hasattr(torch.cuda.memory, 'caching_allocator_config'):
                    torch.cuda.memory.caching_allocator_config(max_split_size_mb=64)
                # Versione precedente di PyTorch
                elif hasattr(torch.cuda, 'caching_allocator_alloc'):
                    # Impostazione attraverso variabile d'ambiente per versioni precedenti
                    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
                    # Puliamo la cache prima di reimpostare
                    torch.cuda.empty_cache()
                else:
                    logger.info("Configurazione avanzata allocatore CUDA non disponibile nella versione corrente di PyTorch")
            except Exception as e:
                logger.warning(f"Non è stato possibile configurare l'allocatore CUDA: {str(e)}")
            
            # Disattiva le ottimizzazioni durante la generazione per mantenere grafici riscaldati
            torch.backends.cudnn.benchmark = False
            
            # Forza CUDA ad usare calcoli FP16 ancora più veloci e TensorCores quando possibile
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                pass  # Pre-warm autocast
                
            # Pre-compila alcuni kernel per FP16
            if hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
                try:
                    if hasattr(torch.cuda, 'set_float32_matmul_precision'):
                        torch.cuda.set_float32_matmul_precision('high')
                except Exception as e:
                    logger.warning(f"Impossibile impostare float32_matmul_precision: {str(e)}")
                
            # Disattiva il synchronize automatico
            if hasattr(torch.cuda, '_lazy_init'):
                torch.cuda._lazy_init()
        
        # Pre-warm GPU per migliorare il tempo della prima inferenza rapida
        if not getattr(model, "_pre_warmed", False):
            logger.info("Pre-riscaldamento GPU per prima inferenza rapida")
            with torch.no_grad():
                # Eseguiamo un piccolo forward pass per riscaldare la GPU
                dummy_input_ids = torch.ones((1, 1), device=model.device, dtype=torch.long)
                model(input_ids=dummy_input_ids)
                # Facciamo un po' di operazioni aggiuntive per inizializzare meglio i kernel
                for _ in range(5):  # Ripete alcune operazioni per inizializzare meglio i kernel
                    dummy_out = model(input_ids=dummy_input_ids)
                model._pre_warmed = True
        
        # Preparazione del prompt di input applicando il template di chat
        logger.info(f"[{request_id[:8]}] Applicazione template di chat ai messaggi")
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Aumentiamo la risoluzione massima per l'elaborazione del modello
        image_size = 1024  # Aumentato da 576 a 1024 per massimizzare la capacità di lettura del testo e dettagli
        inputs = processor(
            text=[prompt],
            images=images if images else None,
            image_size=image_size,
            return_tensors="pt",
            padding=True,
        )
        
        logger.info(f"[{request_id[:8]}] Immagini processate con risoluzione massima di {image_size}px")
        
        # Impostiamo priorità di elaborazione massima
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            torch.cuda.set_device(current_device)
            # Impostiamo la memoria in modo ottimale per operazioni a dimensioni fisse
            torch.cuda.memory_stats(current_device)  # Forza inizializzazione di alcuni cache CUDA
            # Aumentiamo la priorità dello stream CUDA
            with torch.cuda.stream(torch.cuda.Stream(priority=-1)):  # Priorità massima
                # Spostiamo gli input sul dispositivo appropriato
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
                
                # Pre-compiliamo il grafo computazionale per la prima parte dell'inferenza
                if hasattr(inputs, "input_ids") and torch.jit.is_tracing() == False:
                    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                        # Forza compilazione di alcuni kernel
                        _ = model(input_ids=inputs["input_ids"][:, :10], attention_mask=inputs["attention_mask"][:, :10])
                        torch.cuda.synchronize()  # Assicura che la compilazione sia completata
        
        # Impostiamo il garbage collection aggressivo durante l'inferenza
        torch.cuda.empty_cache()
        gc.collect()  # Forza la garbage collection Python
        
        # Log delle statistiche di memoria prima dell'inferenza
        if hasattr(torch.cuda, 'memory_stats'):
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"[{request_id[:8]}] Memoria CUDA prima dell'inferenza: {mem_reserved:.2f} GB riservati, {mem_allocated:.2f} GB allocati")
        
        # Forza un'ultima pulizia brutale prima di iniziare l'inferenza
        with request_lock:
            active_requests[request_id]["status"] = "starting_inference"
            
        torch.cuda.empty_cache()
        gc.collect()
        torch._C._cuda_clearCublasWorkspaces()  # Pulizia aggressiva workspaces CUDA
        
        if images:
            logger.info(f"[{request_id[:8]}] Elaborazione di immagini, forzo pulizia GPU...")
            torch.cuda.synchronize()  # Sincronizza per assicurarsi che la pulizia sia completata

        # Aggiorna statistiche e log
        current_time = time.time()
        if current_time - last_debug_time >= debug_interval:
            update_memory_stats()
            log_memory_stats()
            globals()["last_debug_time"] = current_time

        # Set up streamer for real-time output
        logger.info(f"[{request_id[:8]}] Configurazione streamer e parametri di generazione")
        # Configurazione ottimizzata dello streamer
        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=20,  # Aumentato da 2 a 20 secondi per dare più tempo all'inizio
            stride=2     # Ridotto da 4 a 2 per migliorare la reattività iniziale
        )
        
        # Parametri di generazione ottimizzati per qualità e decodifica del testo
        generation_kwargs = dict(
            inputs, 
            streamer=streamer, 
            max_new_tokens=1500,   # Aumentato da 1200 a 1500 per supportare risposte più dettagliate
            temperature=0.4,      # Aumentato leggermente per migliorare la creatività nell'estrazione dettagli
            repetition_penalty=1.15,  # Aumentato per evitare ripetizioni
            top_k=40,             # Aumentato da 30 a 40 per migliore variabilità
            top_p=0.95,           # Aumentato per una maggiore esplorazione 
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id,
            do_sample=True,
            num_beams=1,
            early_stopping=False,
            use_cache=True,
            max_time=240.0,       # Aumentato da 180 a 240 secondi per dare più tempo all'elaborazione di immagini ad alta risoluzione
            output_attentions=False,
            output_hidden_states=False,
            low_memory=True,
            # Parametri per velocità:
            attention_mask=inputs.get("attention_mask", None),
            return_dict_in_generate=False
        )

        # Start generation in a separate thread
        with request_lock:
            active_requests[request_id]["status"] = "running_inference"
            active_requests[request_id]["generation_start_time"] = time.time()
            active_requests[request_id]["last_token_time"] = time.time()
            active_requests[request_id]["tokens_generated"] = 0
            
        # Avvia thread di monitoraggio per timeout
        def monitor_generation_timeout():
            max_time_no_tokens = 45  # Ridotto da 60 a 45 secondi (più aggressivo)
            max_total_time = 150  # Ridotto da 180 a 150 secondi
            
            # Variabili per il calcolo velocità
            tokens_history = []
            last_report_time = time.time()
            report_interval = 15  # Report velocità solo ogni 15 secondi invece di 10
            
            # Flag di interruzione per la generazione
            interrupted = {"value": False}
            
            while request_id in active_requests and (active_requests[request_id]["status"].startswith("generating") or active_requests[request_id]["status"] == "running_inference"):
                with request_lock:
                    if request_id not in active_requests:
                        break
                        
                    current_time = time.time()
                    generation_time = current_time - active_requests[request_id]["generation_start_time"]
                    time_since_last_token = current_time - active_requests[request_id]["last_token_time"]
                    current_tokens = active_requests[request_id].get("tokens_generated", 0)
                    
                    # Aggiungiamo alla storia per calcolo velocità media
                    tokens_history.append((current_time, current_tokens))
                    # Manteniamo solo gli ultimi 10 secondi di storia
                    while tokens_history and tokens_history[0][0] < current_time - 10:
                        tokens_history.pop(0)
                    
                    # Calcola velocità media solo periodicamente per ridurre overhead di log
                    if current_time - last_report_time >= report_interval and len(tokens_history) >= 2:
                        oldest_time, oldest_tokens = tokens_history[0]
                        newest_time, newest_tokens = tokens_history[-1]
                        time_span = newest_time - oldest_time
                        if time_span > 0:
                            tokens_delta = newest_tokens - oldest_tokens
                            tokens_per_second = tokens_delta / time_span
                            logger.info(f"[{request_id[:8]}] Velocità generazione: {tokens_per_second:.2f} token/s ({tokens_delta} token in {time_span:.1f}s)")
                            last_report_time = current_time
                    
                    # Log periodici solo per generazioni molto lunghe (ridotto overhead)
                    if generation_time > 60 and int(generation_time) % 30 == 0:  # Ogni 30s dopo 1 min
                        logger.warning(f"[{request_id[:8]}] Generazione in corso da {generation_time:.1f}s, {current_tokens} token")
                    
                    # Verifica se siamo bloccati
                    if time_since_last_token > max_time_no_tokens:
                        logger.error(f"[{request_id[:8]}] TIMEOUT: Nessun nuovo token generato in {time_since_last_token:.1f}s")
                        interrupted["value"] = True
                        active_requests[request_id]["status"] = "timeout_no_progress"
                        active_requests[request_id]["interrupted"] = True
                        break
                        
                    # Verifica tempo totale
                    if generation_time > max_total_time:
                        logger.error(f"[{request_id[:8]}] TIMEOUT: Tempo massimo di generazione superato ({max_total_time}s)")
                        interrupted["value"] = True
                        active_requests[request_id]["status"] = "timeout_max_time"
                        active_requests[request_id]["interrupted"] = True
                        break
                
                # Check più frequente (ogni 2 secondi)
                time.sleep(2)
        
        # Avvia thread di monitoraggio
        timeout_thread = Thread(target=monitor_generation_timeout, daemon=True)
        timeout_thread.start()
            
        # Avvia thread di generazione
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Log che indica l'avvio della generazione
        logger.info(f"[{request_id[:8]}] Generazione avviata su dispositivo: {model.device}")

        # Stream the output
        buffer = ""
        yield "Elaborazione in corso..."
        
        # Token counter visibile all'utente
        token_display_counter = 0
        last_display_time = time.time()
        
        # Funzione per verificare se il testo contiene caratteri cinesi
        def contains_chinese(text):
            for char in text:
                if '\u4e00' <= char <= '\u9fff':
                    # Controlliamo che non sia un dato numerico o un simbolo importante
                    if not (char.isdigit() or char in '(),.+-*/=:;%$€£¥#@'):
                        return True
            return False
        
        # Funzione per filtrare i caratteri non latini
        def filter_non_latin(text):
            import re
            # Mantiene solo caratteri latini, numeri, punteggiatura e spazi comuni
            # Ma preserva i numeri e i simboli speciali anche se in mezzo a testo cinese
            filtered_text = re.sub(r'[^\x00-\x7F\xC0-\xFF0-9.,;:@#$%^&*()\[\]{}\-+=/\\|<>?!\'\"]+', ' ', text)
            return filtered_text
        
        token_count = 0
        max_allowed = 1500  # Aumentato per consentire risposte più lunghe
        chinese_detected = False
        last_progress_update = time.time()
        last_token_time = time.time()
        
        # Variabili per rilevare stallo
        token_history = []
        same_output_count = 0
        last_output_length = 0
        potential_loop_detected = False
        
        # Controlla se siamo in timeout o se il processo è stato interrotto
        is_timeout = lambda: (request_id in active_requests and 
                             (active_requests[request_id]["status"] == "timeout_no_progress" or 
                              active_requests[request_id]["status"] == "timeout_max_time" or
                              active_requests[request_id].get("interrupted", False)))
        
        # Inizializza un timeout di sicurezza generale
        start_streaming_time = time.time()
        max_streaming_time = 600  # 10 minuti massimo in totale
        
        try:
            # Gestiamo esplicitamente l'errore Empty, ma mantenendo il ciclo for semplice
            try:
                # Stampa un log per capire quando inizia l'iterazione
                logger.info(f"[{request_id[:8]}] Iniziando l'iterazione dello streamer...")
                
                # Usiamo un contatore per tenere traccia dei tentativi di iterazione
                iteration_start_time = time.time()
                
                # Utilizziamo il ciclo for originale che funzionava bene
                for new_text in streamer:
                    # Log al primo token ricevuto
                    if token_count == 0:
                        first_token_time = time.time() - iteration_start_time
                        logger.info(f"[{request_id[:8]}] Primo token ricevuto dopo {first_token_time:.2f}s")
                    
                    # Aggiorna lo stato e il timestamp dell'ultimo token
                    current_time = time.time()
                    with request_lock:
                        if request_id in active_requests:
                            active_requests[request_id]["status"] = f"generating_token_{token_count}"
                            active_requests[request_id]["last_update"] = current_time
                            active_requests[request_id]["last_token_time"] = current_time
                            active_requests[request_id]["tokens_generated"] = token_count + 1
                    
                    # Aggiorna tempo ultimo token generato e conta i token
                    last_token_time = current_time
                    token_count += 1
                    
                    # Aggiunta del nuovo testo
                    if new_text:
                        buffer += new_text
                        
                        # Aggiorniamo il contatore visibile ogni secondo
                        if current_time - last_display_time >= 1.0:
                            token_display_counter = token_count
                            # Aggiorniamo la velocità nel buffer se siamo ancora nella fase iniziale (primi 100 token)
                            if token_count < 100:
                                display_text = buffer + f"\n[Token generati: {token_display_counter}]"
                                yield display_text
                                # Reset del timer visuale
                                last_display_time = current_time
                    
                    # Termina dopo token max
                    if token_count >= max_allowed:
                        logger.info(f"[{request_id[:8]}] Limite token raggiunto ({token_count}), terminazione della generazione")
                        break
                        
                    # Mostra il buffer aggiornato (ogni token)
                    yield buffer
                
                # Log se l'iterazione termina normalmente
                logger.info(f"[{request_id[:8]}] Iterazione streamer completata normalmente con {token_count} token")
                
            except Empty:
                # Cattura solo l'eccezione Empty, che indica un timeout nella coda
                elapsed = time.time() - iteration_start_time
                logger.warning(f"[{request_id[:8]}] TIMEOUT di {elapsed:.2f}s nella coda del streamer. Token generati: {token_count}")
                
                if token_count == 0:
                    # Se il thread di generazione è ancora attivo, attendiamo
                    if thread.is_alive():
                        logger.info(f"[{request_id[:8]}] Thread di generazione ancora attivo, attendiamo...")
                        yield "La generazione richiede più tempo del previsto. Attendere..."
                        
                        # Tentiamo di cambiare il timeout e riprovare
                        try:
                            # Proviamo a estrarre manualmente alcuni token
                            retry_timeout = 30  # 30 secondi di attesa
                            retry_start = time.time()
                            
                            while thread.is_alive() and time.time() - retry_start < retry_timeout:
                                try:
                                    # Tentiamo di ottenere un token
                                    text = streamer.text_queue.get(timeout=5)
                                    if text:
                                        buffer += text
                                        token_count += 1
                                        logger.info(f"[{request_id[:8]}] Recuperato token: '{text}'")
                                        yield buffer
                                except Empty:
                                    # Continuiamo a provare
                                    time.sleep(1)
                                    continue
                            
                            # Se ancora nessun token
                            if token_count == 0:
                                logger.error(f"[{request_id[:8]}] Impossibile recuperare token nonostante il retry")
                                yield "Nessun token generato dopo un'attesa estesa. Riprova con un'altra richiesta."
                        except Exception as retry_error:
                            logger.error(f"[{request_id[:8]}] Errore durante il retry: {str(retry_error)}")
                            yield "Si è verificato un errore durante il tentativo di recupero. Riprova."
                    else:
                        yield "Nessun token generato. Il modello potrebbe avere difficoltà con questa richiesta. Prova a modificarla."
                else:
                    # Se abbiamo già alcuni token, restituiamo ciò che abbiamo raccolto finora
                    logger.info(f"[{request_id[:8]}] Generazione interrotta ma restituisco {token_count} token generati")
                    buffer += "\n\n[Generazione interrotta, risposta parziale]"
                    yield buffer

        except Exception as e:
            logger.error(f"[{request_id[:8]}] Errore nel ciclo di streaming: {str(e)}")
            logger.exception("Dettaglio errore:")
            # Forniamo almeno qualche feedback all'utente
            if token_count == 0:
                yield "Si è verificato un errore durante la generazione. Riprova."
            else:
                buffer += f"\n\n[Errore durante lo streaming: {str(e)}]"
                yield buffer
        
        # Aggiornamento finale dello stato
        with request_lock:
            if request_id in active_requests:
                active_requests[request_id]["status"] = "completed"
                active_requests[request_id]["completion_time"] = time.time()
                active_requests[request_id]["tokens_generated"] = token_count
                
                # Dopo 5 secondi, rimuovi questa richiesta dalla lista delle attive
                def cleanup_request():
                    time.sleep(5)
                    with request_lock:
                        if request_id in active_requests:
                            del active_requests[request_id]
                            
                cleanup_thread = Thread(target=cleanup_request)
                cleanup_thread.daemon = True
                cleanup_thread.start()
        
        # Log finale
        total_time = time.time() - start_time
        logger.info(f"[{request_id[:8]}] RICHIESTA COMPLETATA - {token_count} token generati in {total_time:.2f} secondi")
        
        # Aggiorna statistiche e log
        current_time = time.time()
        if current_time - last_debug_time >= debug_interval:
            update_memory_stats()
            log_memory_stats()
            globals()["last_debug_time"] = current_time
            
    except Exception as e:
        logger.error(f"[{request_id[:8]}] ERRORE DURANTE L'ELABORAZIONE: {str(e)}")
        with request_lock:
            if request_id in active_requests:
                active_requests[request_id]["status"] = "failed"
                active_requests[request_id]["error"] = str(e)
        return f"Si è verificato un errore durante l'elaborazione: {str(e)}"


# Personalizzazione dell'interfaccia per renderla più accattivante
custom_css = """
body {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    font-family: 'Segoe UI', 'Roboto', sans-serif;
    color: #333;
    margin: 0;
    padding: 0;
    min-height: 100vh;
}

h1, h2, h3, h4, h5, h6 {
    color: #1a5276;
    font-weight: 600;
}

/* Adatta a tutto schermo */
.gradio-container {
    border-radius: 12px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    padding: 25px;
    background-color: rgba(255, 255, 255, 0.95);
    margin: 20px;
    width: calc(100% - 40px) !important;
    max-width: none !important;
}

button {
    background: linear-gradient(to right, #1a5276, #2980b9);
    color: white;
    border: none;
    padding: 12px 24px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    font-weight: 500;
    margin: 5px 3px;
    transition: all 0.3s ease;
    cursor: pointer;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

button:hover {
    background: linear-gradient(to right, #2980b9, #3498db);
    transform: translateY(-2px);
    box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
}

button:active {
    transform: translateY(1px);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Stile per pulsanti di invio ed allegato - migliora allineamento */
[data-testid="send-btn"] svg,
[data-testid="attach-button"] svg {
    display: inline-block;
    vertical-align: middle;
    position: relative;
    top: -2px;
    margin-right: 6px;
}

/* Correggi allineamento testo nei pulsanti */
[data-testid="send-btn"],
[data-testid="attach-button"] {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Stile per i messaggi di chat */
.message-wrap {
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 15px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.message-wrap.user {
    background-color: #f1f8ff;
    border-left: 4px solid #2980b9;
}

.message-wrap.bot {
    background-color: #f9f9f9;
    border-left: 4px solid #16a085;
}

/* Cambia FAKE API in Debug */
button[class*="fake-api"] {
    background: linear-gradient(to right, #6b5b95, #9b59b6) !important;
    padding: 8px 16px !important;
    font-weight: bold !important;
}
"""

demo = gr.ChatInterface(
    fn=model_inference,
    title="**SophyAI Agent : Qwen2.5-VL-7B-Instruct**",
    description="### Analisi di documenti e immagini alimentata da intelligenza artificiale",
    textbox=gr.MultimodalTextbox(label="Inserisci la tua query", file_types=["image"], file_count="multiple"),
    stop_btn="Interrompi Generazione",
    multimodal=True,
    cache_examples=False,
    css=custom_css
)

# Modifica alla chiamata launch per permettere connessioni dall'esterno del container Docker
logger.info("Avvio server Gradio su 0.0.0.0:7860 per permettere connessioni esterne al container Docker")
demo.launch(
    debug=False,         # Disabilitiamo la modalità debug per evitare il pulsante FAKE API
    server_name="0.0.0.0", # Ascolta su tutte le interfacce di rete, non solo localhost
    server_port=7860,    # Specifica esplicitamente la porta standard di Gradio
    share=False,         # Non attivare condivisione Gradio (non necessaria in Docker)
    inbrowser=False,     # Non aprire automaticamente un browser (inutile in Docker)
    show_error=True,     # Mostra errori dettagliati
    max_threads=40       # Aumenta il numero di thread per gestire più richieste simultanee
)