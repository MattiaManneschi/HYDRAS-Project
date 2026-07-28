#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Crea gli archivi split del SUBSET LIVE (held-out) da caricare come asset di una
# GitHub Release, per far scaricare i dati a live_sim senza Google Drive/gdown.
#
# Subset = concentrazioni held-out (SRC107-132, 4 versioni) + 1 corrente per
# versione + 4 vento + il CSV coordinate. ~7,4 GB → pezzi da <2 GB (limite asset).
#
# Uso:   ./make_data_release.sh [DATA_DIR] [OUT_DIR]
#   DATA_DIR : cartella dati (default: data)
#   OUT_DIR  : dove scrivere i pezzi (default: cartella corrente)
#
# Poi carica i pezzi su una Release, es.:
#   gh release create data-v1 hydras_data.part-* \
#      --title "Dati held-out (subset live)" --notes "Subset per live_sim"
# oppure via web: Releases → Draft a new release → tag data-v1 → trascina i file.
# ─────────────────────────────────────────────────────────────────────────────
set -e

DATA_DIR="${1:-data}"
OUT_DIR="${2:-.}"
PREFIX="$(cd "$OUT_DIR" && pwd)/hydras_data.part-"
SPLIT_BYTES=$((1900 * 1024 * 1024))   # 1,9 GB per pezzo (< 2 GB limite asset)

if [[ ! -d "$DATA_DIR" ]]; then echo "Errore: cartella '$DATA_DIR' non trovata."; exit 1; fi
cd "$DATA_DIR"

# Raccoglie i file del subset (percorsi relativi a data/, così l'estrazione ricrea
# la struttura Output_HD_FaseII_CL2_V*/…, Vento_V0-V3/…, CSV).
FILES=()
for f in Output_HD_FaseII_CL2_V*/*_Conc_10mGrid.nc; do
  [[ -e "$f" ]] || continue
  src=$(basename "$f" | sed -E 's/.*SRC0*([0-9]+)_Conc.*/\1/')
  if [[ "$src" =~ ^[0-9]+$ && "$src" -gt 106 ]]; then FILES+=("$f"); fi
done
for f in Output_HD_FaseII_CL2_V*/*SRC000_U_V_10mGrid.nc; do [[ -e "$f" ]] && FILES+=("$f"); done
for f in Vento_V0-V3/CI_WIND_faseII_V*.txt; do [[ -e "$f" ]] && FILES+=("$f"); done
[[ -f Coordinate_Sorgenti_FaseII.csv ]] && FILES+=("Coordinate_Sorgenti_FaseII.csv")

if [[ ${#FILES[@]} -eq 0 ]]; then echo "Errore: nessun file del subset trovato in '$DATA_DIR'."; exit 1; fi
echo "File nel subset: ${#FILES[@]}"

rm -f "${PREFIX}"*
echo "Creo l'archivio e lo spezzo in pezzi da $((SPLIT_BYTES/1024/1024)) MB…"
tar cf - "${FILES[@]}" | split -b "$SPLIT_BYTES" - "$PREFIX"

echo
echo "Pezzi creati:"
ls -la "${PREFIX}"* | awk '{printf "  %s  (%.2f GB)\n", $NF, $5/1073741824}'
echo
echo "Carica su una Release con tag 'data-v1':"
echo "  gh release create data-v1 ${PREFIX}* --title \"Dati held-out (subset live)\" --notes \"Subset per live_sim\""
