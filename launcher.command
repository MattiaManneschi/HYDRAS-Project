#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# HYDRAS Live Sim — launcher DOPPIO-CLICCABILE (macOS).
#
# Fai doppio clic su questo file in Finder: si apre il Terminale e parte tutto.
# (Un .sh normale si aprirebbe nell'editor; l'estensione .command lo rende
#  eseguibile con un doppio clic.)
#
# Scarica src/live_sim.py dal repo GitHub (URL FISSO, nessun 'git' richiesto) e lo
# avvia; da lì live_sim.py scarica il resto (requisiti, script, dati, modelli).
# Richiede: Python 3 + curl (incluso in macOS).
#
# Prima esecuzione di un file scaricato: se macOS lo blocca ("sviluppatore non
# identificato"), fai clic destro → Apri una volta per sbloccarlo.
# ─────────────────────────────────────────────────────────────────────────────
set -e

REPO="MattiaManneschi/HYDRAS-Project"
BRANCH="master"
RAW_URL="https://raw.githubusercontent.com/${REPO}/${BRANCH}/src/live_sim.py"

# Cartella d'installazione: $HYDRAS_HOME se impostata, altrimenti quella di questo file.
HOME_DIR="${HYDRAS_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)}"
mkdir -p "$HOME_DIR/src"
TARGET="$HOME_DIR/src/live_sim.py"

if [[ ! -s "$TARGET" ]]; then
  echo "Scarico live_sim.py…"
  if command -v curl >/dev/null 2>&1; then
    curl -fSL -o "$TARGET" "$RAW_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$TARGET" "$RAW_URL"
  else
    echo "Errore: serve curl oppure wget."; read -n 1 -s -r -p "Premi un tasto…"; exit 1
  fi
fi

PYTHON="$(command -v python3 || command -v python || true)"
if [[ -z "$PYTHON" ]]; then
  echo "Errore: Python non è installato. Installa Python 3 e riprova."
  read -n 1 -s -r -p "Premi un tasto per chiudere…"; exit 1
fi

export HYDRAS_HOME="$HOME_DIR"

# Avvia l'app (NON con exec: così, quando chiudi la GUI, lo script prosegue e
# chiude anche questa finestra del Terminale).
TTY_NAME="$(tty 2>/dev/null || true)"
"$PYTHON" "$TARGET" "$@"

# App chiusa: chiudi la finestra del Terminale che ha eseguito questo launcher.
# Aggancio la finestra al tty di questo script (non alla "frontmost"), in modo
# detached e con un piccolo ritardo: a quel punto la shell è già uscita, quindi
# la finestra si chiude senza il prompt "terminare i processi?".
# NB: la prima volta macOS chiede il permesso di controllare Terminale (una volta).
if [[ "$TTY_NAME" == /dev/* ]]; then
  ( sleep 0.4
    /usr/bin/osascript \
      -e "set theTTY to \"$TTY_NAME\"" \
      -e 'tell application "Terminal"' \
      -e '  repeat with w in windows' \
      -e '    repeat with t in tabs of w' \
      -e '      if tty of t is theTTY then close w' \
      -e '    end repeat' \
      -e '  end repeat' \
      -e 'end tell' >/dev/null 2>&1
  ) >/dev/null 2>&1 &
fi
