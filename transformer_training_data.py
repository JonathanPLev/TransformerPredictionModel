#!/usr/bin/env python3
"""
Generate training data for transformer-based NBA player performance prediction.
Creates sequences with lag features, seasonal averages, and contextual data.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import sqlalchemy as sqla
from src.server.db_local import engine
import torch
from torch.utils.data import Dataset, DataLoader

PLAYER_STAT_COLS = [
    "personid",
    "playerteamname",
    "gameid",
    "clean_game_datetime",
    "hometeamid",
    "awayteamid",
    "points",
    "reboundstotal",
    "reboundsoffensive",
    "assists",
    "numminutes",
    "fieldgoalsmade",
    "fieldgoalsattempted",
    "threepointersmade",
    "threepointersattempted",
    "freethrowsmade",
    "freethrowsattempted",
    "steals",
    "blocks",
    "turnovers",
    "plusminuspoints",
    "foulspersonal",
    "home",
    "opponentteamname",
    "gametype",
]

ROLLING_PLAYER_STAT_COLS = [
    "points",
    "reboundstotal",
    "reboundsoffensive",
    "assists",
    "numminutes",
    "fieldgoalsmade",
    "fieldgoalsattempted",
    "threepointersmade",
    "threepointersattempted",
    "freethrowsmade",
    "freethrowsattempted",
    "steals",
    "blocks",
    "turnovers",
    "plusminuspoints",
    "foulspersonal",
]


PLAYER_FEATURE_COLS = [
    f"blended_std_{col}" for col in ROLLING_PLAYER_STAT_COLS
] + [
    "std_gmsc_baseline",
    "std_tsa_share_baseline",
    "std_expected_minutes",
]

# The individual player baseline stats you want to break out per teammate
TEAMMATE_STAT_COLS = [
    "std_gmsc_baseline",  # From your Game Score calculation
    "std_tsa_share_baseline",  # From your True Shooting Attempt Share calculation
]

TEAMMATE_FEATURE_COLS = [
    f"tm{teammate_num}_{stat}"
    for teammate_num in range(1, 5)
    for stat in TEAMMATE_STAT_COLS
]

TEAM_FEATURE_COLS = [
    "blended_std_ortg",
    "blended_std_drtg",
    "blended_std_win_pct",
    "blended_rolling_3_defensive_delta",
    "blended_rolling_10_defensive_delta",
    "is_b2b",
    "days_since_last_game",
]

OPPONENT_FEATURE_COLS = [
    "opp_rolling_3_ortg",
    "opp_rolling_10_ortg",
    "opp_blended_std_ortg",
    "opp_blended_std_drtg",
    "opp_blended_std_win_pct",
    "opp_blended_rolling_3_defensive_delta",
    "opp_blended_rolling_10_defensive_delta",
    "opp_is_b2b",
    "opp_days_since_last_game",
]

FEATURE_COLS = (
    PLAYER_FEATURE_COLS
    + TEAMMATE_FEATURE_COLS
    + TEAM_FEATURE_COLS
    + OPPONENT_FEATURE_COLS
)

TARGET_COLS = [
    "points",
    "assists", 
    "reboundstotal", 
    "numminutes"
]



class DataPreparer(Dataset):
    def __init__(self, feature_cols=FEATURE_COLS, target_cols=TARGET_COLS,window_size=10):
        self.engine = engine
        self.df = pd.DataFrame
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.samples = []
        self.indices = []

        self.get_all_data()
        self.setup_training_data


    def setup_training_data(self) -> None:
        self.df = self.df.sort_values(['personid','season','clean_game_datetime'])

        self.features = self.df[self.feature_cols].to_numpy(dtype=np.float32)
        self.targets = self.df[self.target_cols].to_numpy(dtype=np.float32)

        for _, group in self.df.groupby(['personid', 'season']):
            n_rows = len(group)
            
            if n_rows > self.window_size:
                start_offset = group.index[0]

                for i in range(n_rows - self.window_size):
                    feat_start = start_offset + i
                    feat_end = feat_start + self.window_size
                    target_idx = feat_end
                    self.indices.append((feat_start, feat_end, target_idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        feat_start, feat_end, target_idx = self.indices[idx]

        x = self.features[feat_start:feat_end]
        y = self.features[target_idx]

        return torch.from_numpy(x), torch.from_numpy(y)


    def get_all_data(self) -> None:
        query = """
        SELECT *
        FROM transformer_training_rows
        WHERE transformer_training_rows.feature_version = 'v6'
        """

        self.df = pd.read_sql(query,self.engine)


