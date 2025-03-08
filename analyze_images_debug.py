import os
import json
import time
import sys
import traceback
from pathlib import Path
from gradio_client import Client, handle_file

def scan_image_directory(directory_path="png", extensions=".jpg,.jpeg,.png"):
    """
    Scandisce una directory alla ricerca di immagini.
    
    Args:
        directory_path: Percorso della directory da scansionare
        extensions: Estensioni delle immagini da cercare
    
    Returns:
        Lista di percorsi completi delle immagini trovate
    """
    print(f"[DEBUG] Inizio scansione della directory: {directory_path}")
    
    if not os.path.exists(directory_path):
        print(f"[ERRORE] La directory '{directory_path}' non esiste")
        return []
    
    if not os.path.isdir(directory_path):
        print(f"[ERRORE] '{directory_path}' non è una directory")
        return []
    
    # Converte le estensioni in una lista
    ext_list = [ext.strip().lower() for ext in extensions.split(",")]
    print(f"[DEBUG] Cerco file con estensioni: {ext_list}")
    
    # Trova tutte le immagini nella directory
    image_files = []
    for root, dirs, files in os.walk(directory_path):
        print(f"[DEBUG] Esaminando sottodirectory: {root}")
        print(f"[DEBUG] Contiene {len(files)} file")
        
        for file in files:
            # Controlla se il file ha un'estensione di immagine
            if any(file.lower().endswith(ext) for ext in ext_list):
                full_path = os.path.join(root, file)
                print(f"[DEBUG] Trovata immagine: {full_path}")
                image_files.append(full_path)
    
    print(f"[INFO] Trovate {len(image_files)} immagini nella directory '{directory_path}'")
    return image_files

def get_available_endpoints(server_url):
    """
    Ottiene l'elenco degli endpoint disponibili sul server Gradio.
    """
    try:
        client = Client(server_url)
        print(f"[DEBUG] Connesso al server Gradio: {server_url}")
        
        # Ottieni le informazioni sugli endpoint
        endpoints = client.endpoints
        print(f"[DEBUG] Endpoint disponibili: {endpoints}")
        
        return endpoints
    except Exception as e:
        print(f"[ERRORE] Impossibile connettersi al server o recuperare gli endpoint: {str(e)}")
        return []

def analyze_image_with_phi4(image_path, question="Descrivi in dettaglio questa immagine", server_url="http://127.0.0.1:7860/"):
    """
    Analizza un'immagine con il modello Phi-4 multimodale tramite Client Gradio.
    
    Args:
        image_path: Percorso dell'immagine da analizzare
        question: Domanda da porre al modello sull'immagine
        server_url: URL del server Gradio con il modello Phi-4
    
    Returns:
        Risposta del modello Phi-4
    """
    try:
        print(f"[DEBUG] Iniziando analisi dell'immagine: {image_path}")
        print(f"[DEBUG] Connessione al server: {server_url}")
        
        # Verifica che il file esista
        if not os.path.exists(image_path):
            print(f"[ERRORE] Il file {image_path} non esiste")
            return f"Errore: File non trovato."
        
        client = Client(server_url)
        print(f"[DEBUG] Client Gradio inizializzato")
        
        # Prepara il file immagine usando handle_file per garantire la corretta gestione
        print(f"[DEBUG] Preparo l'immagine con handle_file")
        try:
            handled_file = handle_file(image_path)
            print(f"[DEBUG] Immagine preparata correttamente")
        except Exception as file_error:
            print(f"[ERRORE] Problema con handle_file: {str(file_error)}")
            # Fallback: usa il percorso diretto
            handled_file = image_path
            print(f"[DEBUG] Fallback: uso il percorso diretto del file")
        
        # Recupera gli endpoint disponibili
        print(f"[DEBUG] Recupero degli endpoint disponibili sul server")
        endpoints = client.endpoints
        print(f"[DEBUG] Endpoint disponibili: {endpoints}")
        
        # Verifica quale endpoint usare
        if "/phi4_demo" in endpoints:
            api_name = "/phi4_demo"
        else:
            print(f"[AVVISO] Endpoint /phi4_demo non trovato, uso il primo disponibile")
            if endpoints:
                api_name = endpoints[0]
            else:
                print(f"[ERRORE] Nessun endpoint disponibile")
                return "Errore: Nessun endpoint disponibile sul server Gradio."
        
        print(f"[DEBUG] Utilizzo endpoint: {api_name}")
        
        # Mostra i parametri che stiamo inviando
        print(f"[DEBUG] Parametri inviati:")
        print(f"  - file: {handled_file}")
        print(f"  - question: {question}")
        print(f"  - processing_type: Image")
        
        # Tenta la chiamata API
        print(f"[DEBUG] Esecuzione della chiamata predict() al server Gradio")
        result = client.predict(
            handled_file,                  # File immagine
            question,                      # Domanda
            "Image",                       # Tipo di elaborazione
            api_name=api_name              # Nome dell'API
        )
        
        print(f"[DEBUG] Risposta ricevuta dal server")
        print(f"[DEBUG] Tipo di risposta: {type(result)}")
        print(f"[DEBUG] Risposta (inizio): {str(result)[:100]}...")
        
        # Attendi brevemente per non sovraccaricare il server
        time.sleep(1)
        
        return result
    except Exception as e:
        print(f"[ERRORE] Eccezione nell'analisi dell'immagine:")
        traceback.print_exc()
        return f"Errore nell'elaborazione: {str(e)}"

def save_result_as_json(image_path, analysis_result):
    """
    Salva il risultato dell'analisi in un file JSON con lo stesso nome dell'immagine.
    
    Args:
        image_path: Percorso dell'immagine analizzata
        analysis_result: Risultato dell'analisi
    
    Returns:
        Percorso del file JSON creato
    """
    try:
        # Crea il nome del file JSON
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        json_filename = f"{image_name}.json"
        print(f"[DEBUG] Salvataggio risultato in: {json_filename}")
        
        # Prepara i dati da salvare
        data = {
            "image_path": image_path,
            "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_result": analysis_result
        }
        
        # Salva il file JSON
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"[INFO] Risultato salvato in '{json_filename}'")
        return json_filename
    except Exception as e:
        print(f"[ERRORE] Errore nel salvataggio del risultato:")
        traceback.print_exc()
        return None

def main():
    # Directory da scansionare
    directory = "png"
    
    # Domanda da porre al modello per ogni immagine
    question = "Descrivi dettagliatamente questa immagine. Identifica oggetti, persone, colori, e il contesto generale."
    
    # URL del server Gradio che ospita il modello Phi-4
    server_url = "http://127.0.0.1:7860/"
    
    print(f"\n{'='*80}")
    print(f"INIZIO DEBUG ANALISI IMMAGINI")
    print(f"{'='*80}\n")
    
    print(f"[INFO] Controllo connettività con il server Gradio...")
    endpoints = get_available_endpoints(server_url)
    if not endpoints:
        print(f"[ERRORE] Impossibile procedere senza una connessione al server Gradio.")
        return
    
    print(f"[INFO] Inizio scansione della directory '{directory}'...")
    
    # Trova tutte le immagini nella directory
    image_files = scan_image_directory(directory)
    
    if not image_files:
        print("[AVVISO] Nessuna immagine trovata. Operazione terminata.")
        return
    
    # Crea la directory per i risultati se non esiste
    results_dir = "risultati_analisi"
    os.makedirs(results_dir, exist_ok=True)
    print(f"[DEBUG] Directory per i risultati: {results_dir}")
    
    # Cambia directory per salvare i risultati nella cartella apposita
    original_dir = os.getcwd()
    os.chdir(results_dir)
    print(f"[DEBUG] Directory di lavoro cambiata in: {os.getcwd()}")
    
    # Analizza ogni immagine e salva il risultato
    results = []
    for idx, image_path in enumerate(image_files):
        print(f"\n{'-'*80}")
        print(f"[INFO] Elaborazione immagine {idx+1}/{len(image_files)}: {image_path}")
        print(f"{'-'*80}")
        
        # Analizza l'immagine con Phi-4
        analysis_result = analyze_image_with_phi4(image_path, question, server_url)
        
        # Salva il risultato in un file JSON
        json_file = save_result_as_json(image_path, analysis_result)
        
        # Aggiungi il risultato alla lista dei risultati
        if json_file:
            results.append({
                "image": image_path,
                "json_file": os.path.join(results_dir, json_file)
            })
    
    # Torna alla directory originale
    print(f"[DEBUG] Ritorno alla directory originale: {original_dir}")
    os.chdir(original_dir)
    
    # Mostra un riepilogo
    print(f"\n{'-'*80}")
    print(f"[INFO] Analisi completata. Analizzate {len(results)}/{len(image_files)} immagini.")
    print(f"[INFO] I risultati sono stati salvati nella directory '{results_dir}'.")
    
    # Crea un file di riepilogo
    summary_file = os.path.join(results_dir, "riepilogo_analisi.json")
    print(f"[DEBUG] Creazione file di riepilogo: {summary_file}")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_images": len(image_files),
            "analyzed_images": len(results),
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] Riepilogo salvato in '{summary_file}'")
    print(f"\n{'='*80}")
    print(f"FINE DEBUG ANALISI IMMAGINI")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main() 