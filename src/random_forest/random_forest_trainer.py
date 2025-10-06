import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.library.parameters import SeasonAll
from src.utils.shared_utils import PlayerUtils, DataProcessor

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
        
        # Add lag features
        stats = self.data_processor.add_lag_features(stats)
        
        # Drop rows with NaN values
        stats = stats.dropna()
        
        # Merge with team statistics
        stats = self.data_processor.merge_team_stats(stats, self.seasons_range[0], self.seasons_range[1])
        
        # Add rolling features
        stats = self.data_processor.add_rolling_features(stats, window=20)
        
        self.training_data = stats
        print(f"Training data prepared: {len(stats)} games")
        return stats
