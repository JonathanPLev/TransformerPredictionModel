import pandas as pd
from nba_api.stats.endpoints import playergamelog, leagueleaders
from nba_api.stats.library.parameters import SeasonAll
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.shared_utils import PlayerUtils, DataProcessor
import time

class StandardizedTrainingPipeline:
    """Standardized training pipeline that works with any NBA player"""
    
    def __init__(self, player_name, seasons_range=('2014-15', '2022-23')):
        """
        Initialize training pipeline for a specific player
        
        Args:
            player_name (str): Full name of the NBA player
            seasons_range (tuple): Start and end seasons for team stats
        """
        self.player_name = player_name
        self.seasons_range = seasons_range
        
        # Get player information
        try:
            self.player_id = PlayerUtils.get_player_id(player_name)
            self.player_team_abbr = PlayerUtils.get_player_team(self.player_id)
            print(f"Initialized training for {player_name} (ID: {self.player_id}, Team: {self.player_team_abbr})")
        except ValueError as e:
            raise ValueError(f"Failed to initialize player: {str(e)}")
        
        # Initialize data processor
        self.data_processor = DataProcessor(self.player_id, self.player_team_abbr)
        self.training_data = None
        self.model = None
    
    def get_career_stats(self):
        """Get career statistics for the player"""
        try:
            career = playergamelog.PlayerGameLog(player_id=self.player_id, season=SeasonAll.all)
            return career.get_data_frames()[0]
        except Exception as e:
            raise ValueError(f"Error getting career stats: {str(e)}")
    
    def adjusted_projected_line(self, row, stats_df):
        """Calculate adjusted projected line based on recent performance"""
        try:
            last_10_pts = stats_df['PTS'].shift(1).loc[max(0, row.name-10):row.name-1]
            under = sum(pts < row['Projected_Line'] for pts in last_10_pts if pd.notna(pts))
            if under < 6:
                return row['Projected_Line'] + 0.5
            else:
                return row['Projected_Line'] - 0.5
        except Exception:
            return row['Projected_Line']
    
    def date_to_features(self, date_val):
        date_obj = pd.to_datetime(date_val)
        return pd.Series({
            'year': date_obj.year,
            'month': date_obj.month,
            'day': date_obj.day,
            'dayofweek': date_obj.weekday(),
            'dayofyear': date_obj.timetuple().tm_yday,
        })


    def prepare_training_data(self):
        """Prepare and process training data"""
        print("Loading career statistics...")
        stats = self.get_career_stats()
        
        # Process common features
        stats = self.data_processor.process_common_features(stats)
        
        # Create projected line based on rolling average
        stats['Projected_Line'] = stats['PTS'].shift(1).rolling(window=10).mean()
        stats['Projected_Line'] = stats.apply(lambda row: self.adjusted_projected_line(row, stats), axis=1)
        stats['Beats_Projected_Line'] = (stats['PTS'] > stats['Projected_Line']).astype(int)
        stats['WL'] = (stats['WL'] == 'W').astype(int)

        
        # Add lag features
        stats = self.data_processor.add_lag_features(stats)
        
        # Drop rows with NaN values
        stats = stats.dropna()
        
        # Merge with team statistics
        stats = self.data_processor.merge_team_stats(stats, self.seasons_range[0], self.seasons_range[1])
        
        # Add rolling features
        stats = self.data_processor.add_rolling_features(stats, window=20)
        
        date_features = stats["GAME_DATE"].apply(self.date_to_features)
        
        stats = pd.concat([stats, date_features], axis=1)
        self.training_data = stats
        print(f"Training data prepared: {len(stats)} games")
        return stats


class MultiPlayerPipeline:
    """Pipeline to create training data for all NBA players in a season"""
    
    def __init__(self, season='2024-25', seasons_range=('2014-15', '2022-23'), min_games=10):
        """
        Initialize multi-player pipeline
        
        Args:
            season (str): Target season to get players from
            seasons_range (tuple): Range for team stats
            min_games (int): Minimum games played to include player
        """
        self.season = season
        self.seasons_range = seasons_range
        self.min_games = min_games
        self.all_training_data = None
        self.player_mapping = {}  # player_name -> player_id mapping
    
    def get_season_players(self):
        """Get all players who played in the specified season"""
        try:
            print(f"Getting all players from {self.season} season...")
            
            # Use league leaders to get all players with stats for the season
            # This gives us players sorted by points, but includes all qualifying players
            leaders = leagueleaders.LeagueLeaders(
                season=self.season, 
                season_type_all_star='Regular Season'
            )
            leaders_df = leaders.get_data_frames()[0]
            
            print(f"Available columns: {list(leaders_df.columns)}")
            print(f"Found {len(leaders_df)} players in league leaders")
            
            # Filter players with minimum games (GP column = Games Played)
            if 'GP' in leaders_df.columns:
                qualified_players = leaders_df[leaders_df['GP'] >= self.min_games].copy()
                qualified_players = qualified_players.rename(columns={'GP': 'GAMES_PLAYED'})
            else:
                print("GP column not found, using all players")
                qualified_players = leaders_df.copy()
                qualified_players['GAMES_PLAYED'] = self.min_games  # Assume they qualify
            
            print(f"Found {len(qualified_players)} players with {self.min_games}+ games")
            
            # Check if PLAYER column exists, if not use PLAYER_NAME
            if 'PLAYER' in qualified_players.columns:
                return qualified_players[['PLAYER_ID', 'PLAYER', 'GAMES_PLAYED']].rename(columns={'PLAYER': 'PLAYER_NAME'})
            else:
                return qualified_players[['PLAYER_ID', 'PLAYER_NAME', 'GAMES_PLAYED']]
            
        except Exception as e:
            print(f"Error getting season players: {str(e)}")
            return pd.DataFrame()
    
    def process_player_data(self, player_id, player_name, player_team_abbr):
        try:
            print(f"Processing {player_name} (ID: {player_id})...")
            
            # Create data processor for this player
            data_processor = DataProcessor(player_id, player_team_abbr)
            
            # Get player's career stats
            career = playergamelog.PlayerGameLog(player_id=player_id, season=SeasonAll.all)
            stats = career.get_data_frames()[0]
            
            if len(stats) == 0:
                print(f"No data found for {player_name}")
                return None
            
            # Process common features
            stats = data_processor.process_common_features(stats)
            
            # Create projected line based on rolling average
            stats['Projected_Line'] = stats['PTS'].shift(1).rolling(window=10).mean()
            stats['Beats_Projected_Line'] = (stats['PTS'] > stats['Projected_Line']).astype(int)
            stats['WL'] = (stats['WL'] == 'W').astype(int)
            
            # Add player info
            stats['Player_Name'] = player_name
            
            # Add lag features
            stats = data_processor.add_lag_features(stats)
            
            # Drop rows with NaN values
            stats = stats.dropna()
            
            if len(stats) == 0:
                print(f"No valid data after processing for {player_name}")
                return None
            
            # Merge with team statistics
            stats = data_processor.merge_team_stats(stats, self.seasons_range[0], self.seasons_range[1])
            
            # Add rolling features
            stats = data_processor.add_rolling_features(stats, window=20)
            
            # Add date features
            date_features = stats["GAME_DATE"].apply(self.date_to_features)
            stats = pd.concat([stats, date_features], axis=1)
            
            print(f"Processed {len(stats)} games for {player_name}")
            return stats
            
        except Exception as e:
            print(f"Error processing {player_name}: {str(e)}")
            return None
    
    def date_to_features(self, date_val):
        """Convert date to features"""
        date_obj = pd.to_datetime(date_val)
        return pd.Series({
            'year': date_obj.year,
            'month': date_obj.month,
            'day': date_obj.day,
            'dayofweek': date_obj.weekday(),
            'dayofyear': date_obj.timetuple().tm_yday,
        })
    
    def create_multi_player_dataset(self, max_players=50):
        """Create dataset with multiple players"""
        # Get qualified players
        qualified_players = self.get_season_players()
        
        if len(qualified_players) == 0:
            print("No qualified players found")
            return None
        
        # Limit number of players to process (for testing)
        if max_players:
            qualified_players = qualified_players.head(max_players)
        
        all_player_data = []
        
        for idx, row in qualified_players.iterrows():
            player_id = row['PLAYER_ID']
            player_name = row['PLAYER_NAME']
            
            # Store player mapping
            self.player_mapping[player_name] = player_id
            
            try:
                # Get player's current team
                player_team_abbr = PlayerUtils.get_player_team(player_id)
                
                # Process player data
                player_data = self.process_player_data(player_id, player_name, player_team_abbr)
                
                if player_data is not None and len(player_data) > 0:
                    all_player_data.append(player_data)
                
                # Add delay to avoid rate limiting
                time.sleep(0.6)  # NBA API rate limit
                
            except Exception as e:
                print(f"Skipping {player_name}: {str(e)}")
                continue
        
        if all_player_data:
            # Combine all player data
            self.all_training_data = pd.concat(all_player_data, ignore_index=True)
            print(f"Created dataset with {len(self.all_training_data)} total games from {len(all_player_data)} players")
            return self.all_training_data
        else:
            print("No player data was successfully processed")
            return None
    
    def save_dataset(self, filename="NBA_Multi_Player_Training_Data.csv"):
        """Save the multi-player dataset"""
        if self.all_training_data is not None:
            self.all_training_data.to_csv(filename, index=False)
            print(f"Dataset saved to {filename}")
            
            # Save player mapping
            mapping_df = pd.DataFrame(list(self.player_mapping.items()), 
                                    columns=['Player_Name', 'Player_ID'])
            mapping_df.to_csv("player_id_mapping.csv", index=False)
            print("Player ID mapping saved to player_id_mapping.csv")
        else:
            print("No data to save")


if __name__ == "__main__":
    # Create multi-player dataset
    multi_pipeline = MultiPlayerPipeline(season='2024-25', min_games=20)
    dataset = multi_pipeline.create_multi_player_dataset(max_players=20)  # Start with 20 players for testing
    
    if dataset is not None:
        multi_pipeline.save_dataset()
    
    # Single-player pipeline
    # pipeline = StandardizedTrainingPipeline(player_name="Franz Wagner", seasons_range=('2014-15', '2022-23'))
    # pipeline.prepare_training_data()
    # pipeline.training_data.to_csv("Franz_Wagner_training_data.csv", index=False)
