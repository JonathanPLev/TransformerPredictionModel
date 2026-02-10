import sqlalchemy as sqla
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True, precision=4)

# uncomment to connect to the sportsbook database, make sure to ssh tunnel if connecting locally and not from server
#engine1 = sqla.create_engine("postgresql+psycopg2://line_dancer:sportsbook_data@localhost:5433/nba_deeplearning")

# to connect straight from server 
engine2 = sqla.create_engine("postgresql+psycopg2://nba:nba@172.24.196.46:5432/nba")

# to connect from local machine - make sure to ssh tunnel to 55432 (or change connection string gitif you're using a different port)
#engine2 = sqla.create_engine("postgresql+psycopg2://nba:nba@127.0.0.1:55432/nba")

def get_training_data():
    with engine2.connect() as conn: 
        result = conn.execute(sqla.text("SELECT * FROM games;"))
        return result.fetchall()

#print(len(get_training_data()))


def get_static_features(player_id, current_season):
    start_date = f"{current_season - 5}-10-01"
    end_date = f"{current_season}-06-30"  
    
    query = f"""
    SELECT game_date, points, num_minutes, field_goals_attempted
    FROM player_statistics
    WHERE person_id = {player_id} 
      AND game_date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY game_date ASC
    """
    
    try:
        with engine2.connect() as conn:
            df = pd.read_sql(query, conn)
    except Exception as e:
        print(f"Error fetching static history: {e}")
        return np.zeros(11, dtype=np.float32)
    
    if df.empty:
        return np.zeros(11, dtype=np.float32)

    df['game_date'] = pd.to_datetime(df['game_date'])
    
    df['season'] = np.where(
        df['game_date'].dt.month >= 10,
        df['game_date'].dt.year,
        df['game_date'].dt.year - 1
    )

    avg_5yr = df['points'].mean()
    
    df_2yr = df[df['season'] == (current_season - 2)]
    avg_2yr = df_2yr['points'].mean() if not df_2yr.empty else 0.0
    
    df_last = df[df['season'] == (current_season - 1)]
    
    if df_last.empty:
        ls_metrics = [0.0] * 7
    else:
        pts = df_last['points'].values
        mins = df_last['num_minutes'].values
        fga = df_last['field_goals_attempted'].values
        
        ls_mean = pts.mean()
        ls_std = pts.std()
        
        half_idx = len(pts) // 2
        early_pts = pts[:half_idx].mean() if half_idx > 0 else 0
        late_pts = pts[half_idx:].mean() if half_idx > 0 else 0
        ls_split = late_pts - early_pts 
        
        def get_trend(y_values):
            if len(y_values) < 2: return 0.0
            x = np.arange(len(y_values))
            return np.polyfit(x, y_values, 1)[0]
            
        ls_mins_trend = get_trend(mins)
        ls_usage_trend = get_trend(fga)
        
        ls_last_10 = pts[-10:].mean() if len(pts) >= 10 else ls_mean
        ls_last_5 = pts[-5:].mean() if len(pts) >= 5 else ls_mean
        
        ls_metrics = [ls_mean, ls_std, ls_split, ls_mins_trend, ls_usage_trend, ls_last_10, ls_last_5]

    age_placeholder = 25.0 
    season_feat = float(current_season)
    
    print(np.array([
        avg_5yr,
        avg_2yr,
        *ls_metrics,
        age_placeholder,
        season_feat
    ], dtype=np.float32))


def main(): 
    get_static_features(2544, 2020)


if __name__ == "__main__":
    main()

