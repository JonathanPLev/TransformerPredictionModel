import pandas as pd
import numpy as np
import sqlalchemy as sqla
from src.server.db_local import engine
from typing import Dict, Optional
from uuid import uuid4


FEATURE_VERSION = "v9"
TRAINING_TABLE_NAME = "transformer_training_rows"
DROP_PLAYER_FIRST_SEASONS = True


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

IDENTIFIER_COLS = [
    "feature_version",
    "personid",
    "gameid",
    "clean_game_datetime",
    "season",
    "player_game_num_in_season",
    "playerteamname",
    "opponentteamname",
    "home",
    "player_team_idx",
    "opp_team_idx"
]


PLAYER_FEATURE_COLS = [
    f"blended_std_{col}" for col in ROLLING_PLAYER_STAT_COLS
] + [
    "std_gmsc_baseline",
    "std_tsa_share_baseline",
    "std_expected_minutes",
    "std_blended_usage_pct",
    "rolling_5_usage_pct"
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
    "blended_std_pace",
    "rolling_5_pace",
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
    "opp_blended_std_pace",
    "opp_rolling_5_pace",
    "opp_blended_rolling_3_defensive_delta",
    "opp_blended_rolling_10_defensive_delta",
    "opp_is_b2b",
    "opp_days_since_last_game",
]

TRAINING_COLS = (
    IDENTIFIER_COLS
    + ROLLING_PLAYER_STAT_COLS
    + PLAYER_FEATURE_COLS
    + TEAMMATE_FEATURE_COLS
    + TEAM_FEATURE_COLS
    + OPPONENT_FEATURE_COLS
)

TRAINING_UNIQUE_KEY_COLS = ["feature_version", "personid", "gameid"]

# potential teammate stats to include, as well as maybe assist to turnover ratio?
"""
    "std_points",
    "std_reboundstotal",
    "std_assists",
    "std_numminutes",
    "std_fieldgoalsmade",
    "std_fieldgoalsattempted",
    "std_threepointersmade",
    "std_threepointersattempted",
    "std_freethrowsmade",
    "std_freethrowsattempted",
    "std_steals",
    "std_blocks",
    "std_turnovers",
    """


class TransformerDataGenerator:
    """Generate sequential training data with per-game opponent defensive stats."""

    def __init__(self):
        self.engine = engine
        self.all_teams = set()
        self.team_to_idx = {}

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        """Quote a trusted SQL identifier for generated PostgreSQL statements."""
        escaped_identifier = identifier.replace('"', '""')
        return f'"{escaped_identifier}"'

    @staticmethod
    def _postgres_type_for_series(series: pd.Series) -> str:
        """Infer a PostgreSQL column type for newly added training columns."""
        if pd.api.types.is_integer_dtype(series):
            return "BIGINT"
        if pd.api.types.is_float_dtype(series):
            return "DOUBLE PRECISION"
        if pd.api.types.is_bool_dtype(series):
            return "BOOLEAN"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "TIMESTAMP"
        return "TEXT"

    def build_training_frame(
        self, df: pd.DataFrame, target_season: Optional[int] = None
    ) -> pd.DataFrame:
        """Select only the columns persisted for model training."""
        training_df = df.copy()
        training_df["feature_version"] = FEATURE_VERSION

        if DROP_PLAYER_FIRST_SEASONS:
            prior_player_seasons = (
                training_df[["personid", "season"]]
                .drop_duplicates()
                .assign(
                    season=lambda player_seasons: player_seasons["season"] + 1
                )
            )
            training_df = training_df.merge(
                prior_player_seasons, on=["personid", "season"], how="inner"
            ).copy()

        if target_season is not None:
            training_df = training_df[training_df["season"] == target_season].copy()

        missing_cols = [col for col in TRAINING_COLS if col not in training_df.columns]
        if missing_cols:
            raise KeyError(f"Missing training columns: {missing_cols}")

        return training_df[TRAINING_COLS].copy()

    def ensure_training_table(self, training_df: pd.DataFrame) -> None:
        """Create the training table and unique key if they do not exist."""
        training_df.head(0).to_sql(
            TRAINING_TABLE_NAME,
            self.engine,
            schema="public",
            if_exists="append",
            index=False,
        )

        with self.engine.begin() as connection:
            existing_cols = set(
                connection.execute(
                    sqla.text(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                        AND table_name = :table_name
                        """
                    ),
                    {"table_name": TRAINING_TABLE_NAME},
                ).scalars()
            )

            for col in training_df.columns:
                if col in existing_cols:
                    continue

                connection.execute(
                    sqla.text(
                        f"""
                        ALTER TABLE public.{self._quote_identifier(TRAINING_TABLE_NAME)}
                        ADD COLUMN IF NOT EXISTS {self._quote_identifier(col)}
                        {self._postgres_type_for_series(training_df[col])}
                        """
                    )
                )

            connection.execute(
                sqla.text(
                    f"""
                    CREATE UNIQUE INDEX IF NOT EXISTS
                    {self._quote_identifier(f"{TRAINING_TABLE_NAME}_unique_key")}
                    ON public.{self._quote_identifier(TRAINING_TABLE_NAME)}
                    (
                        {self._quote_identifier("feature_version")},
                        {self._quote_identifier("personid")},
                        {self._quote_identifier("gameid")}
                    )
                    """
                )
            )

    def save_training_rows(self, training_df: pd.DataFrame) -> None:
        """Upsert training rows into Postgres using the training unique key."""
        if training_df.empty:
            print("No training rows to save.")
            return

        self.ensure_training_table(training_df)

        staging_table = f"tt_rows_stg_{uuid4().hex[:12]}"
        training_df.to_sql(
            staging_table,
            self.engine,
            schema="public",
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=5000,
        )

        quoted_cols = [self._quote_identifier(col) for col in training_df.columns]
        insert_cols = ", ".join(quoted_cols)
        update_cols = [
            col for col in training_df.columns if col not in TRAINING_UNIQUE_KEY_COLS
        ]
        update_assignments = ", ".join(
            f"{self._quote_identifier(col)} = EXCLUDED.{self._quote_identifier(col)}"
            for col in update_cols
        )
        conflict_cols = ", ".join(
            self._quote_identifier(col) for col in TRAINING_UNIQUE_KEY_COLS
        )

        try:
            with self.engine.begin() as connection:
                connection.execute(
                    sqla.text(
                        f"""
                        INSERT INTO public.{self._quote_identifier(TRAINING_TABLE_NAME)}
                        ({insert_cols})
                        SELECT {insert_cols}
                        FROM public.{self._quote_identifier(staging_table)}
                        ON CONFLICT ({conflict_cols})
                        DO UPDATE SET {update_assignments}
                        """
                    )
                )
        finally:
            with self.engine.begin() as connection:
                connection.execute(
                    sqla.text(
                        f"DROP TABLE IF EXISTS public.{self._quote_identifier(staging_table)}"
                    )
                )

    def get_all_teams(self) -> Dict[str, int]:
        """Get all teams for one-hot encoding."""
        if not self.team_to_idx:
            query = """
            SELECT DISTINCT opponent_team_name 
            FROM player_statistics 
            WHERE opponent_team_name IS NOT NULL
            UNION
            SELECT DISTINCT player_team_name 
            FROM player_statistics 
            WHERE player_team_name IS NOT NULL
            """
            df = pd.read_sql(query, self.engine)
            teams = sorted(df["opponent_team_name"].dropna().unique())
            self.team_to_idx = {team: idx for idx, team in enumerate(teams)}
        return self.team_to_idx

    def get_game_info(self, year1: int = 2023, year2: int = 2024):
        """get game and player information per game"""

        self.get_all_teams()

        query = f"""
            SELECT 
                psr.personid,
                psr.playerteamname, 
                psr.gameid,
                to_timestamp(gr.gamedatetimeest, 'YYYY-MM-DD HH24:MI:SS') AS clean_game_datetime,
                gr.hometeamid,
                gr.awayteamid,
                psr.points,
                psr.reboundstotal,
                psr.reboundsoffensive,
                psr.assists,
                psr.numminutes,
                psr.fieldgoalsmade,
                psr.fieldgoalsattempted,
                psr.threepointersmade,
                psr.threepointersattempted,
                psr.freethrowsmade,
                psr.freethrowsattempted,
                psr.steals,
                psr.blocks,
                psr.turnovers,
                psr.plusminuspoints,
                psr.foulspersonal, 
                psr.home,
                psr.opponentteamname,
                psr.gametype
            FROM public.player_statistics_raw psr
            INNER JOIN public.games_raw gr ON psr.gameid = gr.gameid
            WHERE gr.gametype = 'Regular Season'
            AND (
                EXTRACT(YEAR FROM to_timestamp(gr.gamedatetimeest, 'YYYY-MM-DD HH24:MI:SS'))
                - CASE
                    WHEN EXTRACT(MONTH FROM to_timestamp(gr.gamedatetimeest, 'YYYY-MM-DD HH24:MI:SS')) <= 9
                    THEN 1
                    ELSE 0
                  END
            ) IN ({year1}, {year2})
            ORDER BY clean_game_datetime DESC, psr.playerteamname, psr.points DESC;
        """

        df = pd.read_sql(query, self.engine)
        if df.empty:
            return {}

        # Re-sort to stack each player's chronological career games together
        df = df.sort_values(by=["personid", "clean_game_datetime"]).reset_index(
            drop=True
        )

        df["clean_game_datetime"] = pd.to_datetime(df["clean_game_datetime"])

        df["season"] = df["clean_game_datetime"].dt.year.where(
            df["clean_game_datetime"].dt.month > 9,
            df["clean_game_datetime"].dt.year - 1,
        )

        df["reboundstotal"] = pd.to_numeric(df["reboundstotal"], errors="coerce")
        df["reboundsoffensive"] = pd.to_numeric(df["reboundsoffensive"], errors="coerce")

        df["drebs"] = df["reboundstotal"] - df["reboundsoffensive"]


        print(
            f"Loaded seasons {sorted(df['season'].unique())}: "
            f"{len(df)} player-game rows, {df['personid'].nunique()} players"
        )

        df["player_game_num_in_season"] = df.groupby(["personid", "season"]).cumcount()

        # team stats
        # 1. Aggregate to the Team-Game level
        team_game_stats = (
            df.groupby(["gameid", "playerteamname", "opponentteamname"])
            .agg(
                clean_game_datetime=("clean_game_datetime", "first"),
                season=("season", "first"),
                team_fga=("fieldgoalsattempted", "sum"),
                team_fta=("freethrowsattempted", "sum"),
                team_fgm=("fieldgoalsmade", "sum"),
                team_oreb=(
                    "reboundsoffensive",
                    "sum",
                ),  # Ensure this column matches your raw data name
                team_tov=("turnovers", "sum"),
                team_points_scored=("points", "sum"),
                team_minutes_played=("numminutes", "sum"),
                team_dreb=("drebs", "sum")
            )
            .reset_index()
        )



        opp_stats = team_game_stats[
    [
        "gameid", 
        "playerteamname", 
        "team_points_scored", 
        "team_fga", 
        "team_fta", 
        "team_oreb", 
        "team_dreb", 
        "team_fgm", 
        "team_tov"
    ]
    ].rename(
        columns={
            "playerteamname": "opponentteamname",
            "team_points_scored": "team_points_allowed",
            "team_fga": "opp_fga",
            "team_fta": "opp_fta",
            "team_oreb": "opp_oreb",
            "team_dreb": "opp_dreb",
            "team_fgm": "opp_fgm", 
            "team_tov": "opp_tov"
        }
    )



        team_profile = team_game_stats.merge(
            opp_stats, on=["gameid", "opponentteamname"], how="left"
        )

        team_profile = team_profile.sort_values(
            ["playerteamname", "clean_game_datetime"]
        ).reset_index(drop=True)

        cols_to_convert = [
    "team_fga", "team_fta", "team_oreb", "opp_dreb", "team_fgm", "team_tov",
    "opp_fga", "opp_fta", "opp_oreb", "team_dreb", "opp_fgm", "opp_tov"
]
        # 2. Force convert all of them to numeric types in one go
        team_profile[cols_to_convert] = team_profile[cols_to_convert].apply(pd.to_numeric, errors="coerce")

        # 3. Optional: Fill any missing or bad data with 0 so the math doesn't result in NaN
        team_profile[cols_to_convert] = team_profile[cols_to_convert].fillna(0)

        # compute possessions
        team_profile["team_possessions"] = 0.5 * (
            # Team Calculation
            (
                team_profile["team_fga"]
                + (0.44 * team_profile["team_fta"])
                - (
                    1.07 
                    * (team_profile["team_oreb"] / (team_profile["team_oreb"] + team_profile["opp_dreb"])) 
                    * (team_profile["team_fga"] - team_profile["team_fgm"])
                )
                + team_profile["team_tov"]
            )
            + 
            # Opponent Calculation
            (
                team_profile["opp_fga"]
                + (0.44 * team_profile["opp_fta"])
                - (
                    1.07 
                    * (team_profile["opp_oreb"] / (team_profile["opp_oreb"] + team_profile["team_dreb"])) 
                    * (team_profile["opp_fga"] - team_profile["opp_fgm"])
                )
                + team_profile["opp_tov"]
            )
        )

        # Fixed: mapping opponent_possessions to itself, not team_possessions
        team_profile["team_possessions"] = pd.to_numeric(team_profile["team_possessions"], errors="coerce")
        team_profile["opponent_possessions"] = pd.to_numeric(team_profile["team_possessions"], errors="coerce")
        team_profile["team_minutes_played"] = pd.to_numeric(team_profile["team_minutes_played"], errors="coerce")
        team_profile["team_points_scored"] = pd.to_numeric(team_profile["team_points_scored"], errors="coerce")
        # The pace calculation is correct assuming team_minutes_played is the sum of all 5 players' minutes (e.g., 240 for a regulation game)
        team_profile["team_pace"] = 48 * (
        team_profile["team_possessions"] / (team_profile["team_minutes_played"] / 5)
        )

        team_profile["game_ortg"] = 100 * (
            team_profile["team_points_scored"] / team_profile["team_possessions"]
        )

        team_profile["opponent_possessions"] = pd.to_numeric(team_profile["opponent_possessions"], errors="coerce")

        team_profile["team_points_allowed"] = pd.to_numeric(team_profile["team_points_allowed"], errors="coerce")

        team_profile["game_drtg"] = 100 * (
            team_profile["team_points_allowed"] / team_profile["opponent_possessions"]
        )

        team_profile["game_w"] = (
            team_profile["team_points_scored"] > team_profile["team_points_allowed"]
        ).astype(int)

        team_profile = team_profile.sort_values(
            ["playerteamname", "clean_game_datetime"]
        ).reset_index(drop=True)

        team_profile["rolling_3_ortg"] = team_profile.groupby(["playerteamname"])[
            "game_ortg"
        ].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        team_profile["rolling_10_ortg"] = team_profile.groupby(["playerteamname"])[
            "game_ortg"
        ].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        team_profile["rolling_5_pace"] = team_profile.groupby(["playerteamname"])[
            "team_pace"
        ].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

        team_profile["std_ortg_raw"] = team_profile.groupby(
            ["playerteamname", "season"]
        )["game_ortg"].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        team_profile["std_drtg_raw"] = team_profile.groupby(
            ["playerteamname", "season"]
        )["game_drtg"].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        team_profile["std_win_pct_raw"] = team_profile.groupby(
            ["playerteamname", "season"]
        )["game_w"].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())

        team_profile["std_pace_raw"] = team_profile.groupby(
            ["playerteamname", "season"]
        )["team_pace"].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())

        # 2. Get the Game Counter (N) for the blend
        team_profile["team_game_num"] = team_profile.groupby(
            ["playerteamname", "season"]
        ).cumcount()

        past_season_totals = (
            team_profile.groupby(["playerteamname", "season"])
            .agg(
                prior_ortg=("game_ortg", "mean"),
                prior_drtg=("game_drtg", "mean"),
                prior_win_pct=("game_w", "mean"),
                prior_pace=("team_pace", "mean")
            )
            .reset_index()
        )

        # Step the season forward so it aligns as the baseline for the next year
        past_season_totals["season"] = past_season_totals["season"] + 1

        # Merge back into your running profiles
        team_profile = team_profile.merge(
            past_season_totals, on=["playerteamname", "season"], how="left"
        )

        # 3. Apply the Linear Blend Vector Matrix
        N = team_profile["team_game_num"]
        weight = (N / 15).clip(upper=1.0)

        # Execute the blend formula: (1 - weight) * Prior + (weight) * Current_Raw
        team_profile["blended_std_ortg"] = np.where(
            team_profile["prior_ortg"].notna(),
            (1 - weight) * team_profile["prior_ortg"]
            + (weight)
            * team_profile["std_ortg_raw"].fillna(team_profile["prior_ortg"]),
            team_profile["std_ortg_raw"].fillna(0),
        )
        team_profile["blended_std_drtg"] = np.where(
            team_profile["prior_drtg"].notna(),
            (1 - weight) * team_profile["prior_drtg"]
            + (weight)
            * team_profile["std_drtg_raw"].fillna(team_profile["prior_drtg"]),
            team_profile["std_drtg_raw"].fillna(0),
        )
        team_profile["blended_std_win_pct"] = np.where(
            team_profile["prior_win_pct"].notna(),
            (1 - weight) * team_profile["prior_win_pct"]
            + (weight)
            * team_profile["std_win_pct_raw"].fillna(team_profile["prior_win_pct"]),
            team_profile["std_win_pct_raw"].fillna(0),
        )
        team_profile["blended_std_pace"] = np.where(
                    team_profile["prior_pace"].notna(),
                    (1 - weight) * team_profile["prior_pace"]
                    + (weight)
                    * team_profile["std_pace_raw"].fillna(team_profile["prior_pace"]),
                    team_profile["std_pace_raw"].fillna(0),
                )

        opp_ortg_lookup = team_profile[
            ["gameid", "playerteamname", "blended_std_ortg"]
        ].rename(
            columns={
                "playerteamname": "opponentteamname",
                "blended_std_ortg": "opp_season_avg_ortg",
            }
        )

        team_profile = team_profile.merge(
            opp_ortg_lookup, on=["gameid", "opponentteamname"], how="left"
        )

        team_profile["single_game_defensive_delta"] = (
            team_profile["game_drtg"] - team_profile["opp_season_avg_ortg"]
        )
        team_profile = team_profile.sort_values(
            ["playerteamname", "clean_game_datetime"]
        ).reset_index(drop=True)

        team_profile["rolling_3_defensive_delta"] = team_profile.groupby(
            "playerteamname"
        )["single_game_defensive_delta"].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
        days_since_last_game = (
            team_profile["clean_game_datetime"].dt.normalize()
            .groupby(team_profile["playerteamname"])
            .diff()
            .dt.days
        )

        team_profile["rolling_10_defensive_delta"] = team_profile.groupby(
            "playerteamname"
        )["single_game_defensive_delta"].transform(
            lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
        )

        past_season_delta = (
            team_profile.groupby(["playerteamname", "season"])[
                "single_game_defensive_delta"
            ]
            .mean()
            .reset_index()
            .rename(
                columns={"single_game_defensive_delta": "prior_final_defensive_delta"}
            )
        )
        past_season_delta["season"] = past_season_delta["season"] + 1

        # Merge and apply your weight curve matrix
        team_profile = team_profile.merge(
            past_season_delta, on=["playerteamname", "season"], how="left"
        )

        N = team_profile["team_game_num"]
        weight = (N / 15).clip(upper=1.0)

        # Apply the blend to your 10-game baseline to stabilize long-term defensive expectations
        team_profile["blended_rolling_10_defensive_delta"] = np.where(
            team_profile["prior_final_defensive_delta"].notna(),
            ((1 - weight) * team_profile["prior_final_defensive_delta"])
            + (
                weight
                * team_profile["rolling_10_defensive_delta"].fillna(
                    team_profile["prior_final_defensive_delta"]
                )
            ),
            team_profile["rolling_10_defensive_delta"].fillna(0),
        )

        team_profile["blended_rolling_3_defensive_delta"] = np.where(
            team_profile["prior_final_defensive_delta"].notna(),
            ((1 - weight) * team_profile["prior_final_defensive_delta"])
            + (
                weight
                * team_profile["rolling_3_defensive_delta"].fillna(
                    team_profile["prior_final_defensive_delta"]
                )
            ),
            team_profile["rolling_3_defensive_delta"].fillna(0),
        )

        team_profile["days_since_last_game"] = days_since_last_game.fillna(10)

        team_profile["is_b2b"] = (team_profile["days_since_last_game"] == 1).astype(int)

        lookup = team_profile[
            [
                "gameid",
                "playerteamname",
                "rolling_3_ortg",
                "rolling_10_ortg",
                "blended_std_ortg",
                "blended_std_drtg",
                "blended_std_win_pct",
                "blended_rolling_3_defensive_delta",
                "blended_rolling_10_defensive_delta",
                "is_b2b",
                "days_since_last_game",
                "blended_std_pace",
                "rolling_5_pace",
            ]
        ]

        # 2. Merge Opponent Stats to your main player dataframe (Note the column swap)
        opp_lookup = lookup.rename(
            columns={
                "playerteamname": "opponentteamname",
                "rolling_3_ortg": "opp_rolling_3_ortg",
                "rolling_10_ortg": "opp_rolling_10_ortg",
                "blended_std_ortg": "opp_blended_std_ortg",
                "blended_std_drtg": "opp_blended_std_drtg",
                "blended_std_win_pct": "opp_blended_std_win_pct",
                "blended_rolling_3_defensive_delta": "opp_blended_rolling_3_defensive_delta",
                "blended_rolling_10_defensive_delta": "opp_blended_rolling_10_defensive_delta",
                "is_b2b": "opp_is_b2b",
                "days_since_last_game": "opp_days_since_last_game",
                "blended_std_pace": "opp_blended_std_pace",
                "rolling_5_pace": "opp_rolling_5_pace",
            }
        )


        df["numminutes"] = pd.to_numeric(df["numminutes"], errors="coerce")

        df = df[df.groupby(["personid", "season"])["numminutes"].transform("mean") >= 10]
        df = df[df["numminutes"] > 5]

        for col in ROLLING_PLAYER_STAT_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0)
            df[f"{col}_rolling_5"] = df.groupby("personid")[col].transform(
                lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
            )
            df[f"{col}_rolling_10"] = df.groupby("personid")[col].transform(
                lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
            )
            # season to date
            df[f"std_{col}"] = df.groupby(["personid", "season"])[col].transform(
                lambda x: x.shift(1).expanding(min_periods=1).mean()
            )

            prior_season_avg = (
                df.groupby(["personid", "season"])[col]
                .mean()
                .reset_index()
                .rename(columns={col: f"prior_final_avg_{col}"})
            )
            prior_season_avg["season"] = prior_season_avg["season"] + 1
            df = df.merge(prior_season_avg, on=["personid", "season"], how="left")

            N = df["player_game_num_in_season"]
            weight = (N / 15).clip(upper=1.0)
            current_std = df[f"std_{col}"]
            prior_avg = df[f"prior_final_avg_{col}"]
            df[f"blended_std_{col}"] = np.where(
                prior_avg.notna(),
                ((1 - weight) * prior_avg) + (weight * current_std.fillna(prior_avg)),
                current_std.fillna(0),
            )

        # calculate rest of stats for player
        df["gmsc"] = (
            df["points"]
            + (0.4 * df["fieldgoalsmade"])
            - (0.7 * df["fieldgoalsattempted"])
            - (0.4 * (df["freethrowsattempted"] - df["freethrowsmade"]))
            + (0.7 * df["reboundstotal"])
            + df["steals"]
            + (0.7 * df["assists"])
            + (0.7 * df["blocks"])
            - (0.4 * df["foulspersonal"])
            - df["turnovers"]
        )

        df["true_shooting_percentage"] = df["points"] / (2 * df["fieldgoalsattempted"] + .44 * df["freethrowsattempted"])

        # 3. Add individual True Shooting Attempts per row

        df["player_tsa"] = df["fieldgoalsattempted"] + (
            0.44 * df["freethrowsattempted"]
        )

        df["team_fga"] = df.groupby(["gameid", "playerteamname"])[
            "fieldgoalsattempted"
        ].transform("sum")
        df["team_fta"]= df.groupby(["gameid", "playerteamname"])[
            "freethrowsattempted"
        ].transform("sum")
        df["team_numminutes"] = df.groupby(["gameid", "playerteamname"])[
            "numminutes"
        ].transform("sum")
        df["team_tov"] = df.groupby(["gameid", "playerteamname"])[
            "turnovers"
        ].transform("sum")

        df["usage_pct"] = 100 * (
            (
                (df["fieldgoalsattempted"] + 0.44 * df["freethrowsattempted"] + df["turnovers"]) 
                * df["team_numminutes"]
            ) 
            / 
            (
                (df["team_fga"] + 0.44 * df["team_fta"] + df["team_tov"]) 
                * 5 * df["numminutes"]
            )
        )

        df["std_usage_pct_raw"] = df.groupby(["personid", "season"])["usage_pct"].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

        prior_usage_pct = (
                    df.groupby(["personid", "season"])
                    .agg(
                        prior_usage_pct=("usage_pct", "mean"),
                    )
                    .reset_index()
                )
        prior_usage_pct["season"] = prior_usage_pct["season"] + 1
        df = df.merge(prior_usage_pct, on=["personid", "season"], how="left")

        games_played = df["player_game_num_in_season"]
        weight = (games_played / 15).clip(upper=1.0)
        df["std_blended_usage_pct"] = np.where(
            df["prior_usage_pct"].notna(),
            (1 - weight) * df["prior_usage_pct"]
            + (weight) * df["std_usage_pct_raw"].fillna(df["prior_usage_pct"]),
            df["std_usage_pct_raw"].fillna(0),
        )

        df["rolling_5_usage_pct"] = df.groupby(["personid", "season"])["usage_pct"].transform(
                    lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
                )


        # 5. Compute lagged player baselines, blended with prior season for early-season stability.
        df["std_gmsc_raw"] = df.groupby(["personid", "season"])["gmsc"].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

        prior_gmsc = (
            df.groupby(["personid", "season"])
            .agg(
                prior_gmsc=("gmsc", "mean"),
            )
            .reset_index()
        )
        prior_gmsc["season"] = prior_gmsc["season"] + 1
        df = df.merge(prior_gmsc, on=["personid", "season"], how="left")

        games_played = df["player_game_num_in_season"]
        weight = (games_played / 15).clip(upper=1.0)
        df["std_gmsc_baseline"] = np.where(
            df["prior_gmsc"].notna(),
            (1 - weight) * df["prior_gmsc"]
            + (weight) * df["std_gmsc_raw"].fillna(df["prior_gmsc"]),
            df["std_gmsc_raw"].fillna(0),
        )


        df["tsa_share"] = pd.to_numeric(df["player_tsa"]) / (pd.to_numeric(df["team_fga"]) + 0.44 * pd.to_numeric(df["team_fta"]))

        df["std_tsa_share_raw"] = df.groupby(["personid", "season"])[
            "tsa_share"
        ].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())

        prior_tsa_share = (
            df.groupby(["personid", "season"])
            .agg(
                prior_tsa_share=("tsa_share", "mean"),
            )
            .reset_index()
        )
        prior_tsa_share["season"] = prior_tsa_share["season"] + 1
        df = df.merge(prior_tsa_share, on=["personid", "season"], how="left")

        df["std_tsa_share_baseline"] = np.where(
            df["prior_tsa_share"].notna(),
            (1 - weight) * df["prior_tsa_share"]
            + (weight) * df["std_tsa_share_raw"].fillna(df["prior_tsa_share"]),
            df["std_tsa_share_raw"].fillna(0),
        )

        df["std_expected_minutes_raw"] = df.groupby(["personid", "season"])[
            "numminutes"
        ].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())

        prior_expected_minutes = (
            df.groupby(["personid", "season"])
            .agg(
                prior_expected_minutes=("numminutes", "mean"),
            )
            .reset_index()
        )
        prior_expected_minutes["season"] = prior_expected_minutes["season"] + 1
        df = df.merge(prior_expected_minutes, on=["personid", "season"], how="left")

        df["std_expected_minutes"] = np.where(
            df["prior_expected_minutes"].notna(),
            (1 - weight) * df["prior_expected_minutes"]
            + (weight)
            * df["std_expected_minutes_raw"].fillna(df["prior_expected_minutes"]),
            df["std_expected_minutes_raw"].fillna(0),
        )



        df_teammates = df[
            ["gameid", "playerteamname", "personid", "std_expected_minutes"]
            + TEAMMATE_STAT_COLS
        ].copy()

        df_teammates["expected_role_rank"] = df_teammates.groupby(
            ["gameid", "playerteamname"]
        )["std_expected_minutes"].rank(ascending=False, method="first")

        top_4_expected_tms = df_teammates[
            df_teammates["expected_role_rank"] <= 4
        ].copy()

        teammate_matrix = top_4_expected_tms.pivot(
            index=["gameid", "playerteamname"],
            columns="expected_role_rank",
            values=TEAMMATE_STAT_COLS,
        )

        teammate_matrix.columns = [
            f"tm{int(c[1])}_{c[0]}" for c in teammate_matrix.columns
        ]
        teammate_matrix = teammate_matrix.reset_index()

        backup_tm = df_teammates[df_teammates["expected_role_rank"] == 5].copy()
        backup_pivot = backup_tm.pivot(
            index=["gameid", "playerteamname"],
            columns="expected_role_rank",
            values=TEAMMATE_STAT_COLS,
        )
        backup_pivot.columns = [f"tm{int(c[1])}_{c[0]}" for c in backup_pivot.columns]
        backup_pivot = backup_pivot.reset_index()

        df = df.merge(teammate_matrix, on=["gameid", "playerteamname"], how="left")
        df = df.merge(backup_pivot, on=["gameid", "playerteamname"], how="left")

        df["target_expected_rank"] = df.groupby(["gameid", "playerteamname"])[
            "std_expected_minutes"
        ].rank(ascending=False, method="first")

        for i in range(1, 5):  # Loop through slots 1, 2, 3, 4
            # Condition: Is this specific teammate column actually the target player?
            is_self = df["target_expected_rank"] == i

            for stat in TEAMMATE_STAT_COLS:
                # If it is the player themselves, replace that slot's stat with the 5th man's stat
                df[f"tm{i}_{stat}"] = np.where(
                    is_self, df[f"tm5_{stat}"], df[f"tm{i}_{stat}"]
                )

        # 4. Drop the temporary tm5 columns to keep your database clean
        df = df.drop(columns=[col for col in df.columns if col.startswith("tm5_")])

        df = df.merge(lookup, on=["gameid", "playerteamname"], how="left")

        df = df.merge(opp_lookup, on=["gameid", "opponentteamname"], how="left")
        target_season = int(df["season"].max())


        df["player_team_idx"] = df["playerteamname"].map(self.team_to_idx)
        df["opp_team_idx"] = df["opponentteamname"].map(self.team_to_idx)
        missing_team_names = sorted(
            set(df.loc[df["player_team_idx"].isna(), "playerteamname"].dropna())
            | set(df.loc[df["opp_team_idx"].isna(), "opponentteamname"].dropna())
        )
        if missing_team_names:
            raise ValueError(f"Missing team index mappings: {missing_team_names}")

        df["player_team_idx"] = df["player_team_idx"].astype("int64")
        df["opp_team_idx"] = df["opp_team_idx"].astype("int64")

        training_df = self.build_training_frame(df, target_season=target_season)
        print(
            f"Saving season {target_season}: "
            f"{len(training_df)} rows, {training_df['personid'].nunique()} players"
        )
        self.save_training_rows(training_df)
        print("done")


year_combos = [(y, y + 1) for y in range(2000, 2024)]
data_gen = TransformerDataGenerator()
for combo in year_combos:
    data_gen.get_game_info(combo[0], combo[1])