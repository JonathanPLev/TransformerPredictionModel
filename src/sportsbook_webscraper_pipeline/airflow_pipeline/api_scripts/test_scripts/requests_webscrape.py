import requests
import pandas as pd
from datetime import datetime, UTC

def fetch_prizepicks_nba_baseline_points():
    url = "https://api.prizepicks.com/projections"
    params = {"per_page": 500, "single_stat": "true"}
    headers = {
        "Accept": "application/json; charset=UTF-8",
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://app.prizepicks.com/"
    }

    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    projections = pd.json_normalize(data["data"])
    included = pd.json_normalize(data["included"])

    players = included[included["type"] == "new_player"]
    stat_types = included[included["type"] == "stat_type"]

    players = players[
        ["id", "attributes.name", "attributes.league", "attributes.team_name"]
    ].rename(columns={
        "id": "player_id",
        "attributes.name": "player_name",
        "attributes.league": "league",
        "attributes.team_name": "team"
    })

    stat_types = stat_types[["id", "attributes.name"]].rename(
        columns={"id": "stat_type_id", "attributes.name": "stat_type"}
    )

    projections["player_id"] = projections["relationships.new_player.data.id"]
    projections["stat_type_id"] = projections["relationships.stat_type.data.id"]

    merged = (
        projections.merge(players, on="player_id", how="left")
        .merge(stat_types, on="stat_type_id", how="left")
    )

    # --- Filter NBA + Points only ---
    merged = merged[
        (merged["league"].str.upper() == "NBA")
        & (merged["stat_type"].str.lower() == "points")
    ]

    # --- Fill missing indicator columns with False ---
    for col in [
        "attributes.is_promo",
        "attributes.is_flex",
        "attributes.is_alternate",
        "attributes.odds_type",
    ]:
        if col not in merged.columns:
            merged[col] = False

    # --- Keep baseline (standard) projections only ---
    baseline_mask = (
        (merged["attributes.is_promo"] == False)
        & (merged["attributes.is_flex"] == False)
        & (merged["attributes.is_alternate"] == False)
        & (merged["attributes.odds_type"].isin(["standard", None, False]))
    )
    merged = merged[baseline_mask]

    # --- Deduplicate: one projection per player/stat type ---
    merged = merged.sort_values("id").drop_duplicates(subset=["player_name", "stat_type"], keep="first")

    df = merged[[
        "player_name", "team", "league",
        "attributes.line_score", "stat_type",
        "attributes.start_time", "attributes.updated_at"
    ]].rename(columns={
        "attributes.line_score": "line_score",
        "attributes.start_time": "start_time",
        "attributes.updated_at": "updated_at"
    })

    df["scrape_time"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    filename = f"nba_points_baseline_{datetime.now(UTC).strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(df)} NBA baseline 'Points' projections to {filename}")
    return df

if __name__ == "__main__":
    df = fetch_prizepicks_nba_baseline_points()
    print(df.head(20))
