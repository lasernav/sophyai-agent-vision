#!/bin/bash

# Funzione per mostrare un banner di avvio
show_banner() {
    echo "========================================================"
    echo "        SophyAI-Agent-Qwen2.5VL-7B                     "
    echo "        Powered by Qwen/Qwen2.5-VL-7B-Instruct         "
    echo "========================================================"
    echo "Inizializzazione in corso..."
}

# Mostra informazioni sull'ambiente e sulle GPU disponibili
show_environment_info() {
    echo "Informazioni sul sistema:"
    echo "- $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '"')"
    echo "- Python: $(python3 --version)"
    echo "- Pip: $(pip3 --version | awk '{print $1" "$2}')"
    
    if command -v nvidia-smi &> /dev/null; then
        echo -e "\nInformazioni GPU:"
        nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader
    else
        echo -e "\nATTENZIONE: NVIDIA Driver non rilevati. L'applicazione richiede GPU con supporto CUDA."
    fi
}

# Avvia l'app in base al comando fornito
run_app() {
    if [ "$1" == "server" ]; then
        echo -e "\nAvvio del server Gradio..."
        exec python3 app.py
    elif [ "$1" == "process" ]; then
        echo -e "\nAvvio dell'elaborazione immagini..."
        exec python3 scan_analyze_png.py
    else
        echo -e "\nAvvio del server Gradio (modalità predefinita)..."
        exec python3 app.py
    fi
}

# Funzione principale
main() {
    show_banner
    show_environment_info
    
    # Verifica se ci sono argomenti passati al container
    if [ $# -gt 0 ]; then
        run_app $1
    else
        run_app "server"
    fi
}

# Esegui la funzione principale con tutti gli argomenti passati
main "$@" 