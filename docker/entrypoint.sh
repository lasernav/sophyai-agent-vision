#!/bin/bash
set -e

# Imposta le variabili d'ambiente per i compilatori
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# Visualizza info di debug
gcc --version
echo "CC=$CC"
echo "CXX=$CXX"

echo "========================================================" 
echo "        SophyAI-Agent-Qwen2.5VL-7B"
echo "        Powered by Qwen/Qwen2.5-VL-7B-Instruct"
echo "========================================================"

# Informazioni di sistema
echo "Inizializzazione in corso..."
echo "Informazioni sul sistema:"
echo "- $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | sed 's/"//g')"
echo "- Python: $(python3 --version)"
echo "- Pip: $(pip --version | cut -d' ' -f1,2)"
echo ""

# Mostra info GPU
echo "Informazioni GPU:"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader | sed 's/,/, /g'
echo ""

# Avvia l'applicazione in base al comando
if [ "$1" = "server" ]; then
    echo "Avvio del server Gradio..."
    exec python3 app.py
elif [ "$1" = "analyze" ]; then
    echo "Avvio analisi immagini batch..."
    exec python3 scan_analyze_png.py
else
    echo "Comando non riconosciuto. Comandi disponibili:"
    echo "  server  - Avvia il server Gradio"
    echo "  analyze - Avvia analisi di immagini in modalità batch"
    exit 1
fi 