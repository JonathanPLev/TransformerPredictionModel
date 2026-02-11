# NBA Data Warehouse (PostgreSQL)

This module contains a PostgreSQL data warehouse for NBA games, player statistics, and injury data.
All data is downloaded locally and is **not stored in GitHub**.

---

## Folder Structure

- `Data/` – CSV data files (downloaded locally, gitignored)
- `sql/` – SQL migrations (create tables / constraints)
- `ops_sql/` – operational SQL (refresh / upsert logic)
- `download_data.sh` – downloads Kaggle data (and optionally injury data)
- `refresh_daily.sh` – end-to-end refresh (download + DB upsert)
- `docker-compose.yml` – local DB stack that mounts `Data/` into the DB container

---

## Data Sources

The warehouse pulls data from two external sources:

### Kaggle (NBA statistics)

Used for:
- `Games.csv`
- `PlayerStatistics.csv`

Dataset:
- eoinamoore/historical-nba-data-and-player-box-scores

Kaggle is treated as the source of truth for core NBA statistics.

### Google Drive (NBA-Warehouse)

Used for:
- `InjuryData.csv` (optional/manual refresh)

Public folder:
https://drive.google.com/drive/folders/1MzLBNBKa82FIo7qYoS6BKte3DM8-Wbyp?usp=sharing

By default, the daily refresh **does not** re-download `InjuryData.csv`.
To refresh it manually, run `DOWNLOAD_INJURY=1 ./download_data.sh`.

---

## Setup

### 1. Prerequisites

The following tools are required:

- Docker and Docker Compose
- Kaggle CLI
- (Optional) `gdown` (only needed if refreshing injury data)

Install tools:
```bash
pipx install kaggle==1.5.16
pipx install gdown
```

Kaggle authentication must be configured using an API key located at:
```text
~/.kaggle/kaggle.json
```

### 2. Download and refresh the data

From this directory:
```bash
cd src/nba_dataset
./download_data.sh
```

This script will:
- Download `Games.csv` and `PlayerStatistics.csv` from Kaggle
- Download `InjuryData.csv` from Google Drive
- Overwrite existing local CSVs in `Data/`

Notes:
- The script creates/overwrites files in `Data/`.
- If you prefer isolation, create a Python virtual environment and install `gdown` and `kaggle` there before running the script.

### 3. Start the database

After the data has been downloaded, start PostgreSQL:
```bash
cd src/nba_dataset
docker compose up -d
```

The database container reads CSV files directly from the `Data/` directory via the volume mount defined in `docker-compose.yml`.