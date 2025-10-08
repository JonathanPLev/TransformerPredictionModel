# Random Forest NBA Predictor

This directory contains the Random Forest model implementation for predicting NBA player performance.

## 🚀 Quick Start

### Step 1: Create Training Data
**IMPORTANT**: Before training any models, you must first create the training dataset.

```bash
cd ../data_generation
python nba_pipeline_data.py
```

This will create:
- `NBA_Multi_Player_Training_Data.csv` - Multi-player dataset
- `player_id_mapping.csv` - Player name to ID mapping

### Step 2: Train the Model

```bash
cd ../random_forest
python random_forest_trainer.py
```

This will:
- Automatically detect if multi-player data exists
- Train using cross-validation
- Save the best model with metadata

### Step 3: Make Predictions

```bash
# Basic prediction
python predict_player.py "Tyler Herro"

# Detailed prediction with game context
python predict_player.py "Tyler Herro" --opponent LAL --home --line 26.5

# List all available players
python predict_player.py --list-players "Any Name"
```

## 📁 Files Overview

- **`random_forest_trainer.py`** - Main training script with cross-validation
- **`multi_player_predictor.py`** - Predictor class for any NBA player
- **`predict_player.py`** - Command-line interface for predictions
- **`generate_test_data.py`** - Utility for generating test data

## 🎯 Model Features

- **Multi-player training**: Learns from 20+ NBA players
- **Cross-validation**: 5-fold CV for robust evaluation
- **Smart caching**: Loads existing models automatically
- **Player lookup**: Convert player names to predictions
- **Game context**: Factor in opponent, home/away, projected line

## 🔧 Model Management

Models are automatically saved under `models`.

## ⚠️ Requirements

1. Must run `nba_pipeline_data.py` first to create training data
2. Requires NBA API access (handled automatically)
3. Minimum 20 games per player for reliable predictions
4. Currently uses 20 players for the pipeline.
