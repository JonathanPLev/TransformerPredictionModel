#!/usr/bin/env bash
set -euo pipefail

KAGGLE_DATASET="eoinamoore/historical-nba-data-and-player-box-scores"
# NEW NBA-Warehouse folder
GDRIVE_FOLDER_URL="https://drive.google.com/drive/folders/1MzLBNBKa82FIo7qYoS6BKte3DM8-Wbyp?usp=sharing"
OUT_DIR="Data"

mkdir -p "$OUT_DIR"

echo "==> Downloading Kaggle data..."


# Games.csv (Kaggle)
kaggle datasets download -d "$KAGGLE_DATASET" -f Games.csv --force

if [[ -f Games.csv.zip ]]; then
  unzip -o Games.csv.zip -d "$OUT_DIR" >/dev/null
  rm -f Games.csv.zip
elif [[ -f Games.csv ]]; then
  mv -f Games.csv "$OUT_DIR/Games.csv"
else
  echo "ERROR: Games.csv not found after Kaggle download"
  exit 1
fi


# PlayerStatistics.csv (Kaggle)

kaggle datasets download -d "$KAGGLE_DATASET" -f PlayerStatistics.csv --force

if [[ -f PlayerStatistics.csv.zip ]]; then
  unzip -o PlayerStatistics.csv.zip -d "$OUT_DIR" >/dev/null
  rm -f PlayerStatistics.csv.zip
elif [[ -f PlayerStatistics.csv ]]; then
  mv -f PlayerStatistics.csv "$OUT_DIR/PlayerStatistics.csv"
else
  echo "ERROR: PlayerStatistics.csv not found after Kaggle download"
  exit 1
fi


# InjuryData.csv (Google Drive – NBA-Warehouse)

echo "==> Downloading Injury data from NBA-Warehouse (Google Drive)..."

if ! command -v gdown >/dev/null 2>&1; then
  echo "ERROR: gdown not found."
  echo "Install with: pipx install gdown"
  exit 1
fi

# Download folder but KEEP Kaggle files intact
TMP_DIR="$(mktemp -d)"
gdown --folder "$GDRIVE_FOLDER_URL" -O "$TMP_DIR"

# Move only InjuryData.csv
if [[ -f "$TMP_DIR/InjuryData.csv" ]]; then
  mv -f "$TMP_DIR/InjuryData.csv" "$OUT_DIR/InjuryData.csv"
else
  echo "ERROR: InjuryData.csv not found in NBA-Warehouse folder"
  exit 1
fi

rm -rf "$TMP_DIR"

echo ""
echo "✅ Data refresh complete:"
ls -lh "$OUT_DIR" | egrep "Games.csv|PlayerStatistics.csv|InjuryData.csv"