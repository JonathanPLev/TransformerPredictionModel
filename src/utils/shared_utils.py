"""
Shared utilities for NBA prediction pipelines
"""
import pandas as pd
import os
from datetime import datetime
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo
from team_name_map import team_name_map 

class PlayerUtils:
    """Utilities for player information and validation"""
    
    @staticmethod
    def get_player_id(player_name):
        """Get player ID from player name with error handling"""
        try:
            player_info = players.find_players_by_full_name(player_name)
            if not player_info:
                raise ValueError(f"Player '{player_name}' not found!")
            return player_info[0]['id']
        except Exception as e:
            raise ValueError(f"Error finding player '{player_name}': {str(e)}")
    
    @staticmethod
    def get_player_team(player_id):
        """Get current team abbreviation for a player"""
        try:
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            player_data = player_info.get_data_frames()[0]
            return player_data['TEAM_ABBREVIATION'].iloc[0]
        except Exception as e:
            raise ValueError(f"Error getting team for player ID {player_id}: {str(e)}")
    
    @staticmethod
    def get_available_models():
        """Get list of available trained models"""
        models = {}
        
        # Try to load metadata from JSON file
        if os.path.exists("model_metadata.json"):
            try:
                import json
                with open("model_metadata.json", "r") as f:
                    metadata = json.load(f)
                    
                # Convert metadata to the expected format
                for player_name, info in metadata.items():
                    models[player_name] = {
                        "model_path": info["model_path"],
                        "accuracy": info["accuracy"]
                    }
                    
            except Exception as e:
                print(f"Error reading model metadata: {e}")
        
        return models
    
    @staticmethod
    def save_model_metadata(player_name, model_path, accuracy, trained_date=None):
        """Save model metadata to JSON file"""
        import json
        
        metadata = {}
        if os.path.exists("model_metadata.json"):
            try:
                with open("model_metadata.json", "r") as f:
                    metadata = json.load(f)
            except Exception:
                pass
        
        if trained_date is None:
            trained_date = datetime.now().strftime("%Y-%m-%d")
        
        metadata[player_name] = {
            "model_path": model_path,
            "accuracy": accuracy,
            "trained_date": trained_date
        }
        
        with open("model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

class DataProcessor:
    """Common data processing functionality"""
    
    def __init__(self, player_id, player_team_abbr):
        self.player_id = player_id
        self.player_team_abbr = player_team_abbr
    
    @staticmethod
    def assign_season(date):
        """Assigns the season based on the date."""
        year = date.year
        month = date.month

        if month <= 6:
            season = f"{year - 1}-{str(year)[-2:]}"
        else:
            season = f"{year}-{str(year + 1)[-2:]}"
        
        return season
    
    def get_team_stats(self, start_szn, end_szn, data_path="training_data/team_stats_traditional_rs.csv"):
        """Load and filter team stats data"""
        try:
            # Try multiple possible paths for team stats
            possible_paths = [
                data_path,
                "NBA-dataset-stats-player-team/team/team_stats_advanced_rs.csv",
                "../NBA-dataset-stats-player-team/team/team_stats_advanced_rs.csv",
                "NBA-dataset-stats-player-team/team/team_stats_misc_rs.csv",
                "../NBA-dataset-stats-player-team/team/team_stats_misc_rs.csv"
            ]
            
            team_df = None
            for path in possible_paths:
                if os.path.exists(path):
                    team_df = pd.read_csv(path)
                    break
            
            if team_df is None:
                raise FileNotFoundError("Team stats file not found in any expected location")
            
            team_df['SEASON_START_YEAR'] = team_df['SEASON'].apply(lambda x: int(x[:4]))
            filtered_df = team_df[team_df['SEASON_START_YEAR'].between(int(start_szn[:4]), int(end_szn[:4]))]
            
            # Handle missing PLUS_MINUS column
            if 'PLUS_MINUS' not in team_df.columns:
                if 'NET_RATING' in team_df.columns:
                    filtered_df = filtered_df.copy()
                    filtered_df['PLUS_MINUS'] = filtered_df['NET_RATING']
                else:
                    filtered_df = filtered_df.copy()
                    filtered_df['PLUS_MINUS'] = 0
            
            return filtered_df[['TEAM_NAME', 'W', 'L', 'PLUS_MINUS', 'SEASON', 'SEASON_START_YEAR']]
        except Exception as e:
            raise ValueError(f"Error loading team stats: {str(e)}")
    
    def get_player_gamelog(self, season='2024-25'):
        """Get player game log for specified season"""
        try:
            gamelog = playergamelog.PlayerGameLog(player_id=self.player_id, season=season)
            return gamelog.get_data_frames()[0]
        except Exception as e:
            raise ValueError(f"Error getting game log for player {self.player_id}: {str(e)}")
    
    def create_matchup_columns(self, df):
        """Create all possible matchup columns for one-hot encoding"""
        existing_columns = set(df.columns)
        
        # Generate the full set of required columns based on team_name_map and player's team
        required_columns = set(
            [f"MATCHUP_{self.player_team_abbr} @ {abbr}" for abbr in team_name_map.keys()] +
            [f"MATCHUP_{self.player_team_abbr} vs. {abbr}" for abbr in team_name_map.keys()]
        )
        
        # Add missing columns
        missing_columns = required_columns - existing_columns
        for col in missing_columns:
            df[col] = False
        
        print(f"Added {len(missing_columns)} missing matchup columns for team {self.player_team_abbr}")
        return df
    
    def process_common_features(self, df):
        """Process common features used across all pipelines"""
        # Convert game date and sort
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%b %d, %Y')
        df = df.sort_values(by="GAME_DATE", ascending=True)
        
        # Fix home game logic - @ means away game, not home
        df['isHomeGame'] = ~df['MATCHUP'].str.contains('@')
        df['isHomeGame'] = df['isHomeGame'].astype(int)
        
        # Store original matchup before encoding
        df['ORIGINAL_MATCHUP'] = df['MATCHUP']
        
        # One-hot encode matchups
        df = pd.get_dummies(df, columns=['MATCHUP'], drop_first=True)
        
        # Add missing matchup columns
        df = self.create_matchup_columns(df)
        
        # Extract month
        df['Month'] = df['GAME_DATE'].dt.month
        
        # Calculate rest days
        df['Rest_Days_for_Previous_Game'] = df['GAME_DATE'].shift(1).sub(df['GAME_DATE'].shift(2)).dt.days
        df['Rest_Days'] = df['GAME_DATE'].diff().dt.days
        df['DaysUntilNextGame'] = df['GAME_DATE'].shift(-1).sub(df['GAME_DATE']).dt.days
        
        # Minutes from last game
        df['Minutes_LastGame'] = df['MIN'].shift(1)
        
        # Opponent team processing
        df['versus_team_abbr'] = df['ORIGINAL_MATCHUP'].str[-3:]
        df['versus_team'] = df['versus_team_abbr'].map(team_name_map)
        
        # Create team ID mapping
        team_id_map = {team: idx for idx, team in enumerate(team_name_map.values())}
        df['versus_team_id'] = df['versus_team'].map(team_id_map)
        
        return df
    
    def add_lag_features(self, df, stats=['PTS', 'FGM', 'FGA', 'AST', 'REB', 'MIN'], num_lags=20):
        """Add lag features for specified stats"""
        for stat in stats:
            for lag in range(1, num_lags + 1):
                df[f'{stat}_Lag_{lag}'] = df[stat].shift(lag)
        return df
    
    def add_rolling_features(self, df, stats=['PTS', 'FGM', 'FGA', 'AST', 'REB', 'MIN'], window=5):
        """Add rolling average features"""
        for stat in stats:
            df[f'{stat}_Rolling_Avg_{window}'] = df[stat].shift(1).rolling(window=window).mean()
        return df
    
    def merge_team_stats(self, df, start_szn, end_szn):
        """Merge team statistics with player data"""
        team_stats = self.get_team_stats(start_szn, end_szn)
        
        # Add season information
        df['SEASON'] = df['GAME_DATE'].apply(self.assign_season)
        df['previous_season'] = df['SEASON'].apply(
            lambda x: f"{int(x.split('-')[0]) - 1}-{str(int(x.split('-')[0])).zfill(2)}"
        )
        
        # Fix season format for merging
        team_stats.rename(columns={'TEAM_NAME': 'Team_Name'}, inplace=True)
        df['previous_season'] = df['previous_season'].apply(lambda x: f"{x[:4]}-{x[7:]}")
        
        # Merge with team stats
        merged_df = pd.merge(
            df,
            team_stats[['Team_Name','SEASON', 'W', 'L', 'PLUS_MINUS']],
            left_on=['versus_team', 'previous_season'],
            right_on=['Team_Name', 'SEASON'],
            suffixes=('', '_opponent'),
            how='left'
        )
        
        # Clean up and rename
        merged_df = merged_df.drop(columns=['Team_Name', 'previous_season'])
        merged_df.rename(columns={
            'W': 'opponent_wins',
            'L': 'opponent_losses',
            'plus_minus': 'opponent_plus_minus'
        }, inplace=True)
        
        return merged_df

def get_feature_columns(num_lags=20):
    """Get standard feature columns used across all models"""
    return ([f'PTS_Lag_{lag}' for lag in range(1, num_lags + 1)] +
            [f'FGM_Lag_{lag}' for lag in range(1, num_lags + 1)] +
            [f'FGA_Lag_{lag}' for lag in range(1, num_lags + 1)] +
            [f'AST_Lag_{lag}' for lag in range(1, num_lags + 1)] +
            [f'REB_Lag_{lag}' for lag in range(1, num_lags + 1)] +
            ['Month', 'isHomeGame', 'Projected_Line', 'opponent_wins', 'opponent_losses', 'PLUS_MINUS_opponent',
             'versus_team_id', 'Rest_Days_for_Previous_Game', 'Rest_Days', 'DaysUntilNextGame',
             'Minutes_LastGame'])