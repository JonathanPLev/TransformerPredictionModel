#!/bin/bash
set -e

echo "Creating Data folder..."
mkdir -p Data

echo "Downloading CSVs from Google Drive folder..."
gdown --folder "https://drive.google.com/drive/folders/1F9YNF-VUxSssIhwXw7mwohnhAuyimsMm?usp=sharing" -O Data

echo "Done! Your Data/ folder now contains:"
ls Data


