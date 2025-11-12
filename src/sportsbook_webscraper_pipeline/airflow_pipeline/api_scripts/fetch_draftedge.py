import requests
import json
import pandas as pd
from datetime import datetime
import sys

DRAFTEDGE_API = "https://draftedge.com/draftedge-data/nbaprops1.json"

def fetch_draftedge_data():
    """Fetch NBA props from DraftEdge API"""
    try:
        response = requests.get(DRAFTEDGE_API, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching DraftEdge data: {e}", file=sys.stderr)
        return None

def parse_draftedge_props(data):
    """
    Parse DraftEdge API response and extract standardized prop data.
    Extracts all prop types (points, rebounds, assists, combos, etc.).
    
    Returns:
        list: List of dictionaries with standardized schema
    """
    if not data:
        return []
    
    parsed_props = []
    time_scraped = datetime.now().isoformat()
    
    for player_id, player_data in data.items():
        try:
            # Get basic player info
            player_name = player_data.get('Player')
            team = player_data.get('DFS_Team')
            opponent_team = player_data.get('Opp')
            game_time = player_data.get('GameTime')
            
            # Skip if missing essential info
            if not player_name or not team:
                continue
            
            # Convert game time to ISO format if possible
            game_start = None
            if game_time:
                try:
                    # Normalize spaces and remove any timezone abbreviation
                    clean_time = game_time.replace("EDT", "").replace("EST", "").strip()
                    # Expect format like "11-03 19:00"
                    current_year = datetime.now().year
                    parsed_dt = datetime.strptime(f"{current_year}-{clean_time}", "%Y-%m-%d %H:%M")
                    game_start = parsed_dt.isoformat()
                except Exception:
                    # fallback: keep original string so we can inspect later
                    game_start = game_time
            
            # Get prop data
            prop_data = player_data.get('PropData', {})
            
            if not prop_data:
                continue
            
            # Define sportsbooks that typically provide main lines only
            # These books usually have 1-2 lines max and don't clutter with alternates
            MAIN_LINE_BOOKS = {
                'fanduel', 'draftkings', 'betmgm', 'fanatics', 
                'williamhill_us', 'betonlineag'
            }
            
            # Map DraftEdge prop types to standardized line_type names
            PROP_TYPE_MAPPING = {
                'PropPts': 'points',
                'PropReb': 'rebounds',
                'PropAst': 'assists',
                'PropFG3PM': '3-pointers',
                'PropPtsReb': 'points+rebounds',
                'PropPtsAst': 'points+assists',
                'PropPtsRebAst': 'points+rebounds+assists'
            }
            
            # Process each prop type (points, rebounds, assists, etc.)
            for prop_type_key, line_type in PROP_TYPE_MAPPING.items():
                props_list = prop_data.get(prop_type_key, [])
                
                if not props_list:
                    continue
                
                # Track unique lines per bookmaker for this stat type
                bookmaker_lines = {}
                
                # Process each sportsbook's line for this stat
                for prop in props_list:
                    bet_type = prop.get('BetType')
                    
                    # Only process "Over" lines (the line score is the same for Over/Under)
                    if bet_type != 'Over':
                        continue
                    
                    line_score = prop.get('Line')
                    bookmaker = prop.get('Bookmaker')
                    
                    # Skip if missing essential prop info
                    if line_score is None or not bookmaker:
                        continue
                    
                    # FILTER: Only include main line sportsbooks
                    if bookmaker.lower() not in MAIN_LINE_BOOKS:
                        continue
                    
                    # Track lines by bookmaker (use dict to avoid duplicates)
                    if bookmaker not in bookmaker_lines:
                        bookmaker_lines[bookmaker] = line_score
                
                # Create one row per bookmaker with their line for this stat type
                for bookmaker, line_score in bookmaker_lines.items():
                    # Format bookmaker name (capitalize first letter)
                    sportsbook = bookmaker.replace('_', ' ').title()
                    
                    parsed_prop = {
                        'player_name': player_name,
                        'team': team,
                        'sportsbook': sportsbook,
                        'line_score': float(line_score),
                        'game_start': game_start,
                        'time_scraped': time_scraped,
                        'opponent_team': opponent_team,
                        'line_type': line_type
                    }
                    
                    parsed_props.append(parsed_prop)
        
        except Exception as e:
            print(f"Error processing player {player_id}: {e}", file=sys.stderr)
            continue
    
    return parsed_props

def get_draftedge_df():
    """
    Fetch DraftEdge data and return as DataFrame with standardized schema.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['player_name', 'team', 'sportsbook', 
                      'line_score', 'game_start', 'time_scraped', 'opponent_team', 'line_type']
    """
    try:
        data = fetch_draftedge_data()
        
        if not data:
            print("Warning: No data retrieved from DraftEdge")
            return pd.DataFrame(columns=['player_name', 'team', 'sportsbook', 
                                         'line_score', 'game_start', 'time_scraped', 'opponent_team', 'line_type'])
        
        parsed_props = parse_draftedge_props(data)
        
        if not parsed_props:
            print("Warning: No props found after filtering")
            return pd.DataFrame(columns=['player_name', 'team', 'sportsbook', 
                                         'line_score', 'game_start', 'time_scraped', 'opponent_team', 'line_type'])
        
        # Convert to DataFrame with explicit column order
        df = pd.DataFrame(parsed_props)
        df = df[['player_name', 'team', 'sportsbook', 'line_score', 'game_start', 'time_scraped', 'opponent_team', 'line_type']]
        
        print(f"DraftEdge: Successfully retrieved {len(df)} records from {df['sportsbook'].nunique()} sportsbooks")
        return df
    
    except Exception as e:
        print(f"Error fetching DraftEdge data: {e}", file=sys.stderr)
        return pd.DataFrame(columns=['player_name', 'team', 'sportsbook', 
                                     'line_score', 'game_start', 'time_scraped', 'opponent_team', 'line_type'])

def main():
    """
    Main execution function
    """
    print("Fetching NBA player props from DraftEdge...")
    
    # Fetch and parse data
    df = get_draftedge_df()
    
    print(df.head(20))
    
    if df.empty:
        print("No props found in response")
        sys.exit(1)

if __name__ == "__main__":
    main()