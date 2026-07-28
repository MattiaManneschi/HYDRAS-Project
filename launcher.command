#!/usr/bin/env bash
set -e

REPO="MattiaManneschi/HYDRAS-Project"
BRANCH="master"
RAW_URL="https://raw.githubusercontent.com/${REPO}/${BRANCH}/src/live_sim.py"

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

TTY_NAME="$(tty 2>/dev/null || true)"
"$PYTHON" "$TARGET" "$@"

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
