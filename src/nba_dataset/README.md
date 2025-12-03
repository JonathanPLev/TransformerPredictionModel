# NBA Data Warehouse (PostgreSQL)

This module contains a PostgreSQL database setup for NBA games, player statistics, and injury data.

## Folder Structure

- `Data/` – CSV data files (downloaded locally, not stored in GitHub)
- `sql/` – SQL scripts to create tables and load data
- `download_data.sh` – script that pulls the dataset from Google Drive (run this before starting Docker)

## Setup

### 1. Download the data

The CSVs are stored on Google Drive (too large for GitHub).

 The Google Drive folder is publicly accessible. 

 here is the link: https://drive.google.com/drive/folders/1F9YNF-VUxSssIhwXw7mwohnhAuyimsMm?usp=sharing 
 
 
 Instructions on how to make Google Drive files publicly accessible are provided below.

From this folder (`src/nba_dataset`), run:

```bash
./download_data.sh
