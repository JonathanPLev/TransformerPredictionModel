import os
import sys
import time
import pandas as pd

from nba_api.stats.endpoints import playergamelog, leagueleaders
from nba_api.stats.library.parameters import SeasonAll

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.shared_utils import PlayerUtils, DataProcessor


# ============================================================
#   SINGLE PLAYER TRAINING PIPELINE
# ============================================================
class StandardizedTrainingPipeline:
    """Standardized training pipeline for any NBA player."""

    def __init__(self, player_name, seasons_range=('2014-15', '2022-23')):
        self.player_name = player_name
        self.seasons_range = seasons_range

        # Retrieve player ID + team
        try:
            self.player_id = PlayerUtils.get_player_id(player_name)
            self.player_team_abbr = PlayerUtils.get_player_team(self.player_id)
            print(f"Initialized pipeline → {player_name} "
                  f"(ID {self.player_id}, Team {self.player_team_abbr})")

        except ValueError as e:
            raise ValueError(f"Failed to initialize player: {e}")

        self.data_processor = DataProcessor(self.player_id, self.player_team_abbr)
        self.training_data = None
        self.model = None

    # ------------------------------------------------------------
    def get_career_stats(self):
        """Fetch full career game logs."""
        try:
            logs = playergamelog.PlayerGameLog(
                player_id=self.player_id,
                season=SeasonAll.all
            )
            return logs.get_data_frames()[0]

        except Exception as e:
            raise ValueError(f"Error fetching logs: {e}")

    # ------------------------------------------------------------
    @staticmethod
    def date_to_features(date_val):
        """Convert a date into multiple time-based features."""
        date_obj = pd.to_datetime(date_val)
        return pd.Series({
            'year': date_obj.year,
            'month': date_obj.month,
            'day': date_obj.day,
            'dayofweek': date_obj.weekday(),
            'dayofyear': date_obj.timetuple().tm_yday,
        })

    # ------------------------------------------------------------
    def adjusted_projected_line(self, row, stats_df):
        """Shift projected line based on last 10 performance."""
        try:
            idx = row.name
            last10 = stats_df['PTS'].shift(1).loc[max(0, idx - 10):idx - 1]

            under_hits = sum(
                (pts < row['Projected_Line'])
                for pts in last10
                if pd.notna(pts)
            )

            return row['Projected_Line'] + 0.5 if under_hits < 6 else row['Projected_Line'] - 0.5

        except Exception:
            return row.get('Projected_Line', None)

    # ------------------------------------------------------------
    def prepare_training_data(self):
        """Full training dataset preparation."""
        print("Loading career statistics…")
        stats = self.get_career_stats()

        # Ensure data sorted
        stats['GAME_DATE'] = pd.to_datetime(stats['GAME_DATE'])
        stats = stats.sort_values("GAME_DATE")

        # Core features
        stats = self.data_processor.process_common_features(stats)

        # Projected line (rolling)
        stats['Projected_Line'] = stats['PTS'].shift(1).rolling(10).mean()
        stats['Projected_Line'] = stats.apply(
            lambda row: self.adjusted_projected_line(row, stats), axis=1
        )

        stats['Beats_Projected_Line'] = (stats['PTS'] > stats['Projected_Line']).astype(int)
        stats['WL'] = (stats['WL'] == 'W').astype(int)

        # Lag features
        stats = self.data_processor.add_lag_features(stats)

        # Remove "bad" early rows
        stats = stats.dropna()

        # Merge with historical team stats
        stats = self.data_processor.merge_team_stats(
            stats, self.seasons_range[0], self.seasons_range[1]
        )

        # Rolling features
        stats = self.data_processor.add_rolling_features(stats, window=20)

        # Date features
        date_features = stats["GAME_DATE"].apply(self.date_to_features)
        stats = pd.concat([stats, date_features], axis=1)

        self.training_data = stats
        print(f"Training data prepared: {len(stats)} games.")
        return stats


# ============================================================
#   MULTI-PLAYER PIPELINE
# ============================================================
class MultiPlayerPipeline:
    """Pipeline to create training data for all NBA players in a season."""

    def __init__(self, season='2024-25', seasons_range=('2014-15', '2022-23'), min_games=10):
        self.season = season
        self.seasons_range = seasons_range
        self.min_games = min_games

        self.all_training_data = None
        self.player_mapping = {}

    # ------------------------------------------------------------
    def get_season_players(self):
        """Pull all players from the given season with min games."""
        print(f"Fetching season players → {self.season}…")

        try:
            leaders = leagueleaders.LeagueLeaders(
                season=self.season,
                season_type_all_star='Regular Season'
            )
            df = leaders.get_data_frames()[0]

        except Exception as e:
            print(f"Failed to get season players: {e}")
            return pd.DataFrame()

        print(f"Found {len(df)} players in leaders list.")

        # Ensure GP field exists
        if 'GP' in df.columns:
            df = df[df['GP'] >= self.min_games].rename(columns={'GP': 'GAMES_PLAYED'})
        else:
            df['GAMES_PLAYED'] = self.min_games  # fallback

        print(f"{len(df)} players meet minimum game requirement.")

        # Normalize naming
        if "PLAYER" in df.columns:
            return df[['PLAYER_ID', 'PLAYER']].rename(columns={'PLAYER': 'PLAYER_NAME'}).assign(
                GAMES_PLAYED=df['GAMES_PLAYED']
            )
        return df[['PLAYER_ID', 'PLAYER_NAME', 'GAMES_PLAYED']]

    # ------------------------------------------------------------
    def date_to_features(self, date_val):
        date_obj = pd.to_datetime(date_val)
        return pd.Series({
            'year': date_obj.year,
            'month': date_obj.month,
            'day': date_obj.day,
            'dayofweek': date_obj.weekday(),
            'dayofyear': date_obj.timetuple().tm_yday,
        })

    # ------------------------------------------------------------
    def process_player_data(self, player_id, player_name, team_abbr):
        """Run full transformation pipeline for one player."""
        print(f"Processing → {player_name} (ID {player_id})…")

        try:
            logs = playergamelog.PlayerGameLog(player_id=player_id, season=SeasonAll.all)
            stats = logs.get_data_frames()[0]

        except Exception as e:
            print(f"Error fetching data for {player_name}: {e}")
            return None

        if stats.empty:
            print(f"No stats for {player_name}.")
            return None

        # Ensure sorted
        stats['GAME_DATE'] = pd.to_datetime(stats['GAME_DATE'])
        stats = stats.sort_values("GAME_DATE")

        processor = DataProcessor(player_id, team_abbr)
        stats = processor.process_common_features(stats)

        # Projected line
        stats['Projected_Line'] = stats['PTS'].shift(1).rolling(10).mean()
        stats['Beats_Projected_Line'] = (stats['PTS'] > stats['Projected_Line']).astype(int)
        stats['WL'] = (stats['WL'] == 'W').astype(int)
        stats['Player_Name'] = player_name

        # Lag features
        stats = processor.add_lag_features(stats)
        stats = stats.dropna()

        if stats.empty:
            print(f"No valid rows for {player_name} after cleaning.")
            return None

        # Merge team stats
        stats = processor.merge_team_stats(
            stats, self.seasons_range[0], self.seasons_range[1]
        )

        # Rolling features
        stats = processor.add_rolling_features(stats, 20)

        # Date features
        date_features = stats["GAME_DATE"].apply(self.date_to_features)
        stats = pd.concat([stats, date_features], axis=1)

        print(f"{len(stats)} processed games for {player_name}.")
        return stats

    # ------------------------------------------------------------
    def create_multi_player_dataset(self, max_players=50):
        """Build a giant dataset across multiple players."""
        players = self.get_season_players()

        if players.empty:
            print("No season players found.")
            return None

        players = players.head(max_players)

        all_data = []

        for _, row in players.iterrows():
            pid = row['PLAYER_ID']
            name = row['PLAYER_NAME']

            try:
                team = PlayerUtils.get_player_team(pid)
                self.player_mapping[name] = pid

                pdata = self.process_player_data(pid, name, team)
                if pdata is not None and not pdata.empty:
                    all_data.append(pdata)

                time.sleep(0.6)  # API rate limit

            except Exception as e:
                print(f"Skipping {name}: {e}")

        if not all_data:
            print("No data processed.")
            return None

        self.all_training_data = pd.concat(all_data, ignore_index=True)
        print(f"Created dataset: {len(self.all_training_data)} total rows.")
        return self.all_training_data

    # ------------------------------------------------------------
    def save_dataset(self, filename="NBA_Multi_Player_Training_Data.csv"):
        if self.all_training_data is None:
            print("No data to save.")
            return

        self.all_training_data.to_csv(filename, index=False)
        print(f"Saved dataset → {filename}")

        pd.DataFrame(
            list(self.player_mapping.items()),
            columns=['Player_Name', 'Player_ID']
        ).to_csv("player_id_mapping.csv", index=False)

        print("Saved player mapping → player_id_mapping.csv")


# ============================================================
#   MAIN EXECUTION
# ============================================================
if __name__ == "__main__":

    multi = MultiPlayerPipeline(season='2024-25', min_games=20)
    dataset = multi.create_multi_player_dataset(max_players=20)

    if dataset is not None:
        multi.save_dataset()
