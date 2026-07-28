#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# HYDRAS Live Sim — launcher standalone (macOS / Linux).
#
# Scarica src/live_sim.py dal repo GitHub (URL FISSO: nessun 'git' richiesto, così
# funziona anche su una macchina vuota dove hai copiato solo questo file) e lo
# avvia. Da lì è live_sim.py stesso a scaricare il resto (requisiti, gli altri
# script, dati e modelli) dentro la cartella d'installazione.
#
# Uso:   ./launcher.sh                       (installa/avvia nella cartella di questo file)
#        HYDRAS_HOME=/percorso ./launcher.sh (installa/avvia altrove)
# Richiede: Python 3 + curl (o wget) sulla macchina.
# ─────────────────────────────────────────────────────────────────────────────
set -e

REPO="MattiaManneschi/HYDRAS-Project"
BRANCH="master"
RAW_URL="https://raw.githubusercontent.com/${REPO}/${BRANCH}/src/live_sim.py"

# Cartella d'installazione: $HYDRAS_HOME se impostata, altrimenti quella del launcher.
HOME_DIR="${HYDRAS_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)}"
mkdir -p "$HOME_DIR/src"
TARGET="$HOME_DIR/src/live_sim.py"

# Scarica live_sim.py solo se manca (se già presente, usa la copia locale).
if [[ ! -s "$TARGET" ]]; then
  echo "Scarico live_sim.py…"
  if command -v curl >/dev/null 2>&1; then
    curl -fSL -o "$TARGET" "$RAW_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$TARGET" "$RAW_URL"
  else
    echo "Errore: serve curl oppure wget per scaricare live_sim.py."; exit 1
  fi
fi

PYTHON="$(command -v python3 || command -v python || true)"
if [[ -z "$PYTHON" ]]; then
  echo "Errore: Python non è installato. Installa Python 3 e riprova."; exit 1
fi

export HYDRAS_HOME="$HOME_DIR"
exec "$PYTHON" "$TARGET" "$@"
