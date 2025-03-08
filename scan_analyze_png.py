"""
Semplice applicazione che:
1. Scansiona la directory 'png'
2. Invia ogni immagine all'API Gradio Qwen2.5-VL
3. Salva la descrizione ritornata in un file JSON con lo stesso nome dell'immagine
"""

import os
import json
import time
import re
import sys
import datetime
import threading
from gradio_client import Client, handle_file

# -------- CONFIGURAZIONE --------
# Directory contenente le immagini da analizzare
PNG_DIRECTORY = "png"

# URL del server Gradio
GRADIO_SERVER = "http://127.0.0.1:7860/"

# Domanda da porre per ogni immagine
QUESTION = """Estrai TUTTI i dati testuali e numerici visibili nell'immagine in maniera COMPLETA. 
Rispondi ESCLUSIVAMENTE con un oggetto JSON valido, rigorosamente in questo formato:
{
  "dati_estratti": [
    {"valore": "testo o numero", "tipo": "testo/numero/data", "confidenza": 5},
    {"valore": "altro testo", "tipo": "testo", "confidenza": 4}
  ],
  "metadata": {
    "tipo_documento": "tipo di documento",
    "qualità_immagine": "alta/media/bassa"
  }
}
È FONDAMENTALE rispondere SOLO con JSON valido, senza testo introduttivo o conclusivo."""

# Nome dell'endpoint API (deve corrispondere a quello nel server Gradio)
API_NAME = "/chat"  # Endpoint corretto per l'app Qwen2.5-VL

# Directory per i risultati
RESULTS_DIR = "risultati"

# Livello di debug (0=minimo, 1=normale, 2=dettagliato, 3=molto dettagliato)
DEBUG_LEVEL = 3

# -------- FUNZIONI DI UTILITÀ --------

def debug_print(level, message, force_flush=False):
    """Stampa messaggi di debug in base al livello richiesto"""
    if level <= DEBUG_LEVEL:
        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"[{timestamp}] {'#'*level} {message}")
        if force_flush:
            sys.stdout.flush()

def timeit(func):
    """Decoratore per misurare il tempo di esecuzione di una funzione"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        debug_print(2, f"INIZIO: {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        debug_print(2, f"FINE: {func.__name__} - Tempo: {end_time - start_time:.2f} secondi")
        return result
    return wrapper

# -------- FUNZIONI --------

@timeit
def scan_png_directory():
    """Scansiona la directory PNG e restituisce l'elenco di tutte le immagini."""
    debug_print(1, f"Scansione della directory: {PNG_DIRECTORY}")
    
    if not os.path.exists(PNG_DIRECTORY):
        debug_print(1, f"Errore: La directory {PNG_DIRECTORY} non esiste.")
        return []
        
    # Lista delle estensioni di immagine da cercare
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']
    
    # Trova tutte le immagini nella directory
    images = []
    for file in os.listdir(PNG_DIRECTORY):
        file_path = os.path.join(PNG_DIRECTORY, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in extensions:
                images.append(file_path)
                debug_print(3, f"Trovata immagine: {file_path}")
                
    debug_print(1, f"Trovate {len(images)} immagini.")
    return images

@timeit
def analyze_image(image_path, client):
    """
    Invia un'immagine all'API Gradio Qwen2.5-VL e restituisce la descrizione.
    """
    debug_print(1, f"🔍 ANALISI IMMAGINE: {image_path}", True)
    
    try:
        # Prepara l'immagine con handle_file per garantire una corretta gestione
        debug_print(2, "Preparazione file immagine...")
        start_time = time.time()
        handled_file = handle_file(image_path)
        debug_print(2, f"File preparato in {time.time() - start_time:.2f} secondi: {type(handled_file)}")
        
        if os.path.exists(image_path):
            # Ottieni informazioni sul file
            file_size = os.path.getsize(image_path) / 1024 / 1024  # in MB
            debug_print(2, f"Dimensione file: {file_size:.2f} MB")
        
        # Formato del messaggio per ChatInterface
        debug_print(1, "Analisi interfaccia ChatInterface di Gradio...")
        
        # Per debug, mostra tutti gli endpoint disponibili
        debug_print(2, f"Endpoint disponibili:")
        for i, endpoint in enumerate(client.endpoints):
            debug_print(2, f"  {i}: {endpoint}")
        
        # Tentativo specifico per ChatInterface
        debug_print(1, "⏳ INIZIO COMUNICAZIONE CON SERVER GRADIO...", True)
        
        # Crea un messaggio nel formato per ChatInterface
        message = {"text": QUESTION, "files": [handled_file]}
        debug_print(2, f"Messaggio preparato: text={QUESTION[:50]}... | files={len(message['files'])} elementi")
        
        # Variabili condivise tra i thread per il timeout
        response_data = {"result": None, "completed": False, "last_update_time": time.time(), "bytes_received": 0}
        
        # Funzione che esegue la chiamata API effettiva
        def api_call_with_monitoring():
            try:
                debug_print(2, "Tentativo con formato ChatInterface (metodo predict)")
                debug_print(1, "⚙️ INIZIO ELABORAZIONE SERVER - attendere risposta...", True)
                
                try:
                    # Chiamata API effettiva
                    result = client.predict(
                        message,  # Un singolo parametro che contiene testo e file
                        None,     # History (può essere None per la prima iterazione)
                        api_name="/chat"  # Prova direttamente con /chat
                    )
                    response_data["result"] = result
                    response_data["completed"] = True
                    debug_print(2, "Chiamata API riuscita con formato ChatInterface!")
                except Exception as e1:
                    debug_print(2, f"Errore con formato ChatInterface: {str(e1)}")
                    try:
                        # Altri tentativi di formato come prima...
                        debug_print(2, "Provo approccio alternativo...")
                        result = client.predict(
                            QUESTION,
                            handled_file,
                            api_name="/"
                        )
                        response_data["result"] = result
                        response_data["completed"] = True
                        debug_print(2, "Chiamata API riuscita con formato alternativo!")
                    except Exception as e2:
                        debug_print(2, f"Errore con formato alternativo: {str(e2)}")
                        # Gli altri tentativi possono continuare qui...
                        response_data["completed"] = True  # Impostiamo completed anche in caso di errore
            except Exception as e:
                debug_print(1, f"❌ Errore durante la chiamata API: {str(e)}")
                response_data["completed"] = True
                response_data["error"] = str(e)
        
        # Avvia thread di chiamata API
        api_thread = threading.Thread(target=api_call_with_monitoring)
        api_thread.daemon = True  # Consenti al thread principale di terminare anche se questo è ancora in esecuzione
        api_thread.start()
        
        # Monitoraggio del progresso e timeout
        api_start_time = time.time()
        max_timeout = 180  # Timeout massimo in secondi (3 minuti)
        stall_timeout = 60  # Timeout di stallo in secondi (1 minuto senza nuovi dati)
        polling_interval = 5  # Controlla ogni 5 secondi
        
        dots = 0
        
        # Loop di monitoraggio - uscirà quando la risposta è completa o dopo timeout
        while not response_data["completed"]:
            elapsed = time.time() - api_start_time
            dots = (dots % 10) + 1
            
            # Aggiorna l'indicatore di progresso
            debug_print(1, f"⏱️ Attesa risposta... {elapsed:.1f}s {'.'*dots}", True)
            
            # Verifica timeout generale
            if elapsed > max_timeout:
                debug_print(1, f"⚠️ TIMEOUT FORZATO: L'elaborazione ha superato {max_timeout} secondi. Interruzione!", True)
                # Quando si verifica un timeout, restituisci un messaggio parziale o un errore
                break
            
            # Dormi per un po' prima di controllare di nuovo
            time.sleep(polling_interval)
            
        # Verifica se il thread ha completato con successo
        if not api_thread.is_alive() and response_data["result"] is not None:
            result = response_data["result"]
            debug_print(1, f"✅ RISPOSTA RICEVUTA dopo {time.time() - api_start_time:.2f} secondi!", True)
        else:
            # Thread bloccato o nessun risultato - forza terminazione
            debug_print(1, f"⚠️ Timeout: Il processo di generazione sembra bloccato o ha impiegato troppo tempo", True)
            raise Exception("Timeout durante la comunicazione con il server Gradio - possibile loop infinito rilevato")
        
        # Controllo lunghezza risposta
        debug_print(2, "Elaborazione risposta ricevuta...")
        result_length = len(str(result)) if result else 0
        debug_print(1, f"Risposta ricevuta: {result_length} caratteri")
        
        # Debug approfondito sulla struttura della risposta
        debug_print(2, f"Tipo risposta: {type(result).__name__}")
        
        if isinstance(result, list):
            debug_print(2, f"Risposta è una lista di {len(result)} elementi")
            for i, item in enumerate(result):
                debug_print(3, f"Elemento {i}: tipo={type(item).__name__}")
                
        elif isinstance(result, dict):
            debug_print(2, f"Risposta è un dizionario con chiavi: {list(result.keys())}")
            
        # Se la risposta è una lista o un dict, prendi l'elemento che contiene la risposta testuale
        if isinstance(result, list) and len(result) > 0:
            debug_print(2, "Estrazione dell'ultimo elemento dalla lista")
            result = result[-1]  # Spesso l'ultimo elemento è la risposta
        
        # Se è ancora un dizionario, estrai il testo
        if isinstance(result, dict):
            debug_print(2, "Estrazione del testo dal dizionario")
            if 'output' in result:
                debug_print(3, "Chiave 'output' trovata nel dizionario")
                result = str(result.get('output'))
            else:
                debug_print(3, f"Chiave 'output' non trovata. Chiavi disponibili: {list(result.keys())}")
                result = str(result)  # Converti l'intero dizionario in stringa
        
        # Assicurati che la risposta sia una stringa
        result = str(result)
        
        if result_length > 0:
            debug_print(2, f"Primi 100 caratteri: {result[:100]}...")
            if result_length > 100:
                debug_print(2, f"Ultimi 100 caratteri: ...{result[-100:]}")
        
        # Verifica se la risposta è completa
        if result_length < 100:
            debug_print(1, "⚠️ ATTENZIONE: La risposta sembra troppo breve!")
        
        debug_print(1, "✅ ANALISI IMMAGINE COMPLETATA", True)
        return result
    except Exception as e:
        debug_print(1, f"❌ ERRORE durante l'analisi dell'immagine {image_path}: {str(e)}")
        # Restituisce un errore strutturato invece di una stringa
        return f"Errore: {str(e)}"

@timeit
def extract_json_from_text(text):
    """
    Tenta di estrarre un oggetto JSON valido dal testo, anche se circondato da altro testo.
    """
    if not text:
        debug_print(2, "Testo vuoto, impossibile estrarre JSON")
        return None
    
    # Cerca testo che sembra JSON (inizia con { e finisce con })
    debug_print(2, "Ricerca pattern JSON nel testo...")
    json_pattern = re.compile(r'(\{.*\})', re.DOTALL)
    match = json_pattern.search(text)
    
    if match:
        json_text = match.group(1)
        debug_print(3, f"Pattern JSON trovato: {json_text[:100]}...")
        try:
            # Prova a parsare come JSON
            debug_print(2, "Tentativo di parsing JSON...")
            json_obj = json.loads(json_text)
            debug_print(2, "Parsing JSON completato con successo")
            return json_obj
        except json.JSONDecodeError as e:
            # Se fallisce, prova a pulire ulteriormente il testo
            debug_print(2, f"Errore parsing JSON: {str(e)}")
            debug_print(2, "Tentativo di pulizia JSON...")
            try:
                # Pulisci backslash, singoli apici e altri caratteri problematici
                cleaned = json_text.replace('\\', '').replace("'", '"')
                # Aggiusta virgole trailing problematiche
                cleaned = re.sub(r',\s*}', '}', cleaned)
                cleaned = re.sub(r',\s*]', ']', cleaned)
                debug_print(3, f"JSON pulito: {cleaned[:100]}...")
                json_obj = json.loads(cleaned)
                debug_print(2, "Parsing JSON pulito completato con successo")
                return json_obj
            except Exception as clean_error:
                debug_print(2, f"Errore pulizia JSON: {str(clean_error)}")
                debug_print(2, "Impossibile correggere il JSON")
    else:
        debug_print(2, "Nessun pattern JSON trovato nel testo")
    return None

@timeit
def save_to_json(image_path, description):
    """
    Salva la descrizione dell'immagine in un file JSON con lo stesso nome dell'immagine.
    """
    # Crea il nome del file JSON
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    json_file = os.path.join(RESULTS_DIR, f"{base_name}.json")
    debug_print(2, f"Preparazione file JSON: {json_file}")
    
    # Estrai JSON dalla risposta se possibile
    debug_print(2, "Tentativo di estrazione JSON dalla risposta...")
    extracted_json = extract_json_from_text(description)
    
    # Prepara i dati JSON
    data = {
        "image": image_path,
        "raw_response": description,
        "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "response_length": len(description) if description else 0
    }
    
    # Aggiungi il JSON estratto se valido
    if extracted_json:
        data["extracted_data"] = extracted_json
        data["is_valid_json"] = True
        debug_print(1, "✅ Estratto JSON valido dalla risposta")
    else:
        data["is_valid_json"] = False
        debug_print(1, "❌ Non è stato possibile estrarre JSON valido dalla risposta")
    
    # Salva il file JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    debug_print(1, f"Risultato salvato in: {json_file}")
    return json_file

@timeit
def main():
    """Funzione principale"""
    # Dichiarazione global all'inizio della funzione
    global API_NAME
    
    debug_print(1, "=" * 50)
    debug_print(1, "ANALISI IMMAGINI CON API GRADIO QWEN2.5-VL")
    debug_print(1, f"Livello debug: {DEBUG_LEVEL}")
    debug_print(1, "=" * 50)
    
    # Crea la directory per i risultati se non esiste
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        debug_print(2, f"Directory {RESULTS_DIR} creata")
    
    debug_print(1, "Scansione directory in corso...")
    # Trova tutte le immagini
    images = scan_png_directory()
    if not images:
        debug_print(1, "Nessuna immagine trovata. Termino.")
        return
        
    # Connessione al server Gradio
    debug_print(1, f"Connessione al server Gradio: {GRADIO_SERVER}")
    try:
        start_time = time.time()
        client = Client(GRADIO_SERVER)
        end_time = time.time()
        debug_print(1, f"✅ Connessione stabilita in {end_time - start_time:.2f} secondi")
        
        # Mostra informazioni sui componenti disponibili
        debug_print(1, "DETTAGLI INTERFACCIA GRADIO:")
        try:
            debug_print(2, f"Endpoint disponibili: {[str(ep) for ep in client.endpoints]}")
        except Exception as e:
            debug_print(2, f"Non è stato possibile ottenere gli endpoint come stringhe: {str(e)}")
            
        try:
            # Stampa informazioni più dettagliate sugli endpoint
            for i, endpoint in enumerate(client.endpoints):
                debug_print(2, f"Endpoint {i}: {endpoint}")
                try:
                    # Verifica anche i dettagli dei componenti se possibile
                    if hasattr(endpoint, "component"):
                        debug_print(3, f"  Tipo componente: {endpoint.component.get('type', 'sconosciuto')}")
                except Exception as e:
                    debug_print(3, f"  Errore nel ricavare dettagli componente: {str(e)}")
        except Exception as e:
            debug_print(2, f"Errore nell'analisi degli endpoint: {str(e)}")
            
        debug_print(1, "Uso l'interfaccia predefinita senza specificare endpoint.")
                
    except Exception as e:
        debug_print(1, f"❌ Impossibile connettersi al server Gradio: {str(e)}")
        return
        
    # Analisi di ogni immagine
    results = []
    success_count = 0
    json_valid_count = 0
    
    total_start_time = time.time()
    
    for i, image_path in enumerate(images):
        debug_print(1, f"\nIMAGINE {i+1}/{len(images)}: {image_path}")
        
        img_start_time = time.time()
        try:
            # Ottieni la descrizione dell'immagine
            description = analyze_image(image_path, client)
            
            # Salva il risultato in un file JSON
            json_file = save_to_json(image_path, description)
            
            # Aggiungi il risultato alla lista
            result_info = {
                "image": image_path,
                "json_file": json_file,
                "status": "success",
                "processing_time": time.time() - img_start_time,
                "response_length": len(description) if description else 0
            }
            
            # Verifica se abbiamo estratto JSON valido
            extracted_json = extract_json_from_text(description)
            if extracted_json:
                result_info["json_valid"] = True
                json_valid_count += 1
            else:
                result_info["json_valid"] = False
                
            results.append(result_info)
            success_count += 1
            
            # Mostra tempo elaborazione
            img_elapsed = time.time() - img_start_time
            debug_print(1, f"✅ Immagine elaborata in {img_elapsed:.2f} secondi")
            
            # Stima del tempo rimanente
            elapsed_so_far = time.time() - total_start_time
            images_left = len(images) - (i + 1)
            avg_time_per_image = elapsed_so_far / (i + 1)
            estimated_time_left = avg_time_per_image * images_left
            
            debug_print(1, f"⏱️ Tempo stimato rimanente: {estimated_time_left/60:.1f} minuti ({images_left} immagini)")
            
        except Exception as e:
            debug_print(1, f"❌ ERRORE nell'elaborazione dell'immagine {image_path}: {str(e)}")
            results.append({
                "image": image_path,
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - img_start_time
            })
        
        # Pausa più lunga tra le richieste per evitare sovraccarichi e dare tempo al modello
        pause_time = 10
        debug_print(1, f"Pausa di {pause_time} secondi prima della prossima immagine...")
        time.sleep(pause_time)
        
    # Tempo totale
    total_elapsed = time.time() - total_start_time
    
    # Crea un file di riepilogo
    summary_file = os.path.join(RESULTS_DIR, "riepilogo.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_images": len(images),
            "successful_analyses": success_count,
            "valid_json_responses": json_valid_count,
            "failed_analyses": len(images) - success_count,
            "total_processing_time_seconds": total_elapsed,
            "average_time_per_image": total_elapsed / len(images) if len(images) > 0 else 0,
            "results": results
        }, f, ensure_ascii=False, indent=2)
        
    debug_print(1, "\n" + "=" * 50)
    debug_print(1, f"ANALISI COMPLETATA: {success_count}/{len(images)} immagini analizzate con successo")
    debug_print(1, f"Risposte con JSON valido: {json_valid_count}/{success_count}")
    debug_print(1, f"Tempo totale: {total_elapsed/60:.2f} minuti")
    debug_print(1, f"Tempo medio per immagine: {total_elapsed/len(images):.2f} secondi")
    debug_print(1, f"I risultati sono stati salvati nella directory '{RESULTS_DIR}'")
    debug_print(1, "=" * 50)

if __name__ == "__main__":
    main() 