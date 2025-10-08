# NBA Data Pipeline

This directory contains the data generation pipeline for creating NBA player training datasets.

## 🏀 Pipeline Options

### Option 1: Multi-Player Pipeline (Recommended)

Creates a comprehensive dataset with multiple NBA players from the 2024-25 season.

```bash
python nba_pipeline_data.py
```

**What it does:**
- Gets all players from 2024-25 season with 20+ games
- Processes each player's career statistics
- Creates features: rolling averages, team stats, opponent data
- Combines all players into one large dataset
- Saves `NBA_Multi_Player_Training_Data.csv` and `player_id_mapping.csv`

**Configuration:**
```python
# In nba_pipeline_data.py main section
multi_pipeline = MultiPlayerPipeline(
    season='2024-25',      # Target season
    min_games=20,          # Minimum games to include player
)
dataset = multi_pipeline.create_multi_player_dataset(max_players=20)
```

### Option 2: Single-Player Pipeline

Creates a dataset for one specific player (like Franz Wagner).

```python
# Uncomment the single-player section in nba_pipeline_data.py
pipeline = StandardizedTrainingPipeline(
    player_name="Franz Wagner", 
    seasons_range=('2014-15', '2022-23')
)
pipeline.prepare_training_data()
pipeline.training_data.to_csv("Franz_Wagner_training_data.csv", index=False)
```

## 📊 Generated Features

Both pipelines create the same feature set:

### Player Performance Features
- **Rolling averages**: Points, assists, rebounds over last 10/20 games
- **Lag features**: Previous game statistics
- **Projected line**: Rolling 10-game average for points prediction
- **Target variable**: `Beats_Projected_Line` (1 if over, 0 if under)

### Game Context Features
- **Date features**: Year, month, day, day of week, day of year
- **Team matchup**: Opponent team (one-hot encoded)
- **Home/Away**: Game location indicator
- **Win/Loss**: Team performance in that game

### Team Statistics Features
- **Historical team stats**: Merged from 2014-15 to 2022-23 seasons
- **Opponent strength**: Team performance metrics

## 📁 Output Files

### Multi-Player Pipeline
- **`NBA_Multi_Player_Training_Data.csv`** - Combined dataset (~13,000+ games)
- **`player_id_mapping.csv`** - Player name to NBA ID mapping

### Single-Player Pipeline  
- **`Franz_Wagner_training_data.csv`** - Individual player dataset (~300+ games)

## ⚙️ Configuration Options

### MultiPlayerPipeline Parameters
```python
MultiPlayerPipeline(
    season='2024-25',           # NBA season to get players from
    seasons_range=('2014-15', '2022-23'),  # Range for team stats
    min_games=20,               # Minimum games played to include
)

# In create_multi_player_dataset()
max_players=20                  # Limit number of players (for testing)
```

### StandardizedTrainingPipeline Parameters
```python
StandardizedTrainingPipeline(
    player_name="Franz Wagner",  # Full player name
    seasons_range=('2014-15', '2022-23')  # Team stats date range
)
```

## 🚀 Usage Examples

### Quick Multi-Player Dataset
```bash
# Default: 20 players, 20+ games each
python nba_pipeline_data.py
```

### Custom Multi-Player Dataset
```python
# Edit nba_pipeline_data.py main section:
multi_pipeline = MultiPlayerPipeline(season='2024-25', min_games=30)
dataset = multi_pipeline.create_multi_player_dataset(max_players=50)
```

### Single Player Dataset
```python
# Edit nba_pipeline_data.py main section:
# Comment out multi-player section
# Uncomment single-player section
pipeline = StandardizedTrainingPipeline("Jayson Tatum")
pipeline.prepare_training_data()
pipeline.training_data.to_csv("Jayson_Tatum_training_data.csv", index=False)
```

## 🔧 Dependencies

- `nba_api` - NBA statistics API
- `pandas` - Data manipulation
- `utils.shared_utils` - Custom utility functions for player/team data

## ⚠️ Important Notes

1. **NBA API Rate Limits**: 0.6 second delay between player requests
2. **Season Format**: Use format like '2024-25', not '2024-2025'
3. **Player Names**: Use full names like "Franz Wagner", not "F. Wagner"
4. **Internet Required**: Fetches live data from NBA API

## 🐛 Troubleshooting

### "No qualified players found"
- Check season format ('2024-25')
- Verify NBA API is accessible
- Try reducing `min_games` parameter

### "Player not found"
- Use exact player name from NBA roster
- Check spelling and capitalization
- Player must have played in specified season

### "Rate limit exceeded"
- Reduce `max_players` parameter
- Increase delay in `time.sleep()` calls
