#!/bin/bash

# Verifica se è richiesto l'ambiente virtuale
if [ ! -d "venv" ]; then
  echo "Creazione ambiente virtuale Python..."
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  source venv/bin/activate
fi

# Funzione per terminare correttamente
cleanup() {
  echo "Terminazione applicazione..."
  # Termina eventuali processi in background
  if [ ! -z "$SERVER_PID" ]; then
    kill $SERVER_PID 2>/dev/null
  fi
  deactivate
  exit 0
}

# Imposta la gestione dei segnali per terminare correttamente
trap cleanup SIGINT SIGTERM

# Verifica il comando richiesto
if [ "$1" == "server" ] || [ "$1" == "" ]; then
  echo "Avvio del server Gradio..."
  python app.py
elif [ "$1" == "process" ]; then
  echo "Avvio elaborazione batch delle immagini..."
  python scan_analyze_png.py
else
  echo "Comando non riconosciuto. Utilizzo: ./run_local.sh [server|process]"
  echo "  - server: avvia il server Gradio (default)"
  echo "  - process: esegue l'elaborazione batch delle immagini"
  exit 1
fi 