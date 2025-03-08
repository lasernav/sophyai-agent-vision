import os
import json
import time
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
    if not os.path.exists(directory_path):
        print(f"Errore: La directory '{directory_path}' non esiste")
        return []
    
    if not os.path.isdir(directory_path):
        print(f"Errore: '{directory_path}' non è una directory")
        return []
    
    # Converte le estensioni in una lista
    ext_list = [ext.strip().lower() for ext in extensions.split(",")]
    
    # Trova tutte le immagini nella directory
    image_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            # Controlla se il file ha un'estensione di immagine
            if any(file.lower().endswith(ext) for ext in ext_list):
                full_path = os.path.join(root, file)
                image_files.append(full_path)
    
    print(f"Trovate {len(image_files)} immagini nella directory '{directory_path}'")
    return image_files

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
        print(f"Analisi di '{image_path}'...")
        client = Client(server_url)
        
        # Prepara il payload con l'immagine
        result = client.predict(
            file=image_path,                      # File immagine
            question=question,                    # Domanda
            processing_type="Image",              # Tipo di elaborazione
            api_name="/phi4_demo"                 # Nome dell'API
        )
        
        # Attendi brevemente per non sovraccaricare il server
        time.sleep(1)
        
        return result
    except Exception as e:
        print(f"Errore nell'analisi dell'immagine '{image_path}': {str(e)}")
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
        
        # Prepara i dati da salvare
        data = {
            "image_path": image_path,
            "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_result": analysis_result
        }
        
        # Salva il file JSON
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Risultato salvato in '{json_filename}'")
        return json_filename
    except Exception as e:
        print(f"Errore nel salvataggio del risultato per '{image_path}': {str(e)}")
        return None

def main():
    # Directory da scansionare
    directory = "png"
    
    # Domanda da porre al modello per ogni immagine
    question = "Descrivi dettagliatamente questa immagine. Identifica oggetti, persone, colori, e il contesto generale."
    
    # URL del server Gradio che ospita il modello Phi-4
    server_url = "http://127.0.0.1:7860/"
    
    print(f"Inizio scansione della directory '{directory}'...")
    
    # Trova tutte le immagini nella directory
    image_files = scan_image_directory(directory)
    
    if not image_files:
        print("Nessuna immagine trovata. Operazione terminata.")
        return
    
    # Crea la directory per i risultati se non esiste
    results_dir = "risultati_analisi"
    os.makedirs(results_dir, exist_ok=True)
    
    # Cambia directory per salvare i risultati nella cartella apposita
    original_dir = os.getcwd()
    os.chdir(results_dir)
    
    # Analizza ogni immagine e salva il risultato
    results = []
    for image_path in image_files:
        print(f"\nAnalisi dell'immagine {image_path}...")
        
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
    os.chdir(original_dir)
    
    # Mostra un riepilogo
    print(f"\nAnalisi completata. Analizzate {len(results)} immagini.")
    print(f"I risultati sono stati salvati nella directory '{results_dir}'.")
    
    # Crea un file di riepilogo
    summary_file = os.path.join(results_dir, "riepilogo_analisi.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_images": len(image_files),
            "analyzed_images": len(results),
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Riepilogo salvato in '{summary_file}'")

if __name__ == "__main__":
    main() 