#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

echo "[$(date)] NBA refresh starting..."

# 1) Download latest CSVs
./download_data.sh

# 2) Ensure DB is up
docker compose up -d

# 3) Sanity check: tables must exist 
docker compose exec -T db psql -U nba -d nba -c "\dt" | grep -q "games" || {
  echo "ERROR: DB tables not found. Run one-time migration on a fresh DB volume."
  exit 1
}

# 4) Refresh data
docker compose exec -T db psql -U nba -d nba -f /ops_sql/refresh_from_csv.sql

echo "[$(date)] NBA refresh finished."