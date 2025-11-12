import requests
import pandas as pd
from datetime import datetime 
import sys

PRIZEPICKS_API = "https://api.prizepicks.com/projections"

NBA_TEAMS = {
    'ATL', 'BOS', 'BKN', 'BRK', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
    'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
    'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
}

def get_prizepicks_data():
    """Fetch data from PrizePicks API"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        'Accept': 'application/json',
        'Referer': 'https://app.prizepicks.com/',
        'Origin': 'https://app.prizepicks.com',
        'Host': 'api.prizepicks.com',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site',
    }
    
    try:
        response = requests.get(PRIZEPICKS_API, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        return None


def filter_projections(data):    
    """Filter and standardize PrizePicks projections"""
    if not data:
        return []
    
    projections = data.get('data', [])
    included = data.get('included', [])
    
    players = {p['id']: p for p in included if p['type'] == 'new_player'}
    leagues = {l['id']: l for l in included if l['type'] == 'league'}
    stat_types = {s['id']: s for s in included if s['type'] == 'stat_type'}
    events = {e['id']: e for e in included if e['type'] == 'event'}
    teams = {t['id']: t for t in included if t['type'] == 'team'}
    
    time_scraped = datetime.now().isoformat()
    results = []
    
    for proj in projections:
        try:
            attrs = proj.get('attributes', {})
            rels = proj.get('relationships', {})
            
            # Check NBA league
            league_id = rels.get('league', {}).get('data', {}).get('id')
            if league_id:
                league = leagues.get(league_id, {})
                if 'NBA' not in league.get('attributes', {}).get('name', '').upper():
                    continue
            
            # Filter out special modes and non-base lines
            if any([
                attrs.get('projection_type', '').lower() in ['goblin', 'demon', 'special', 'multiplier'],
                any(k in attrs.get('board_name', '').lower() for k in ['goblin', 'demon', 'multiplier', 'combo', 'parlay']),
                attrs.get('odds_type', '').lower() != 'standard',
                attrs.get('is_promo', False),
                attrs.get('is_boosted', False),
                attrs.get('is_alternate', False)
            ]):
                continue
            
            # Get player info
            player_id = rels.get('new_player', {}).get('data', {}).get('id')
            if not player_id:
                continue
            
            player_attrs = players.get(player_id, {}).get('attributes', {})
            player_name = player_attrs.get('name') or player_attrs.get('display_name')
            team = player_attrs.get('team')
            
            # Validate single NBA player
            if not player_name or team not in NBA_TEAMS:
                continue
            
            if any(sep in player_name for sep in [' & ', ' and ', ' + ', ',']):
                continue
            
            # Extract opponent
            description = attrs.get('description', '')
            
            results.append({
                'player_name': player_name,
                'team': team,
                'sportsbook': 'PrizePicks',
                'line_score': attrs.get('line_score'),
                'game_start': attrs.get('start_time'),
                'time_scraped': time_scraped,
                'opponent_team': attrs.get('description'),
                'line_type': attrs.get('stat_type')
            })
        
        except Exception as e:
            continue
    
    return results

def get_prizepicks_df():
    """
    Fetch PrizePicks NBA projections and return as DataFrame.
    
    Returns:
        pd.DataFrame: Columns ['player_name', 'team', 'sportsbook', 'line_score', 
                      'line_type', 'game_start', 'time_scraped', 'opponent_team']
    """
    columns = ['player_name', 'team', 'sportsbook', 'line_score', 
               'game_start', 'time_scraped', 'opponent_team', 'line_type']
    
    try:
        data = get_prizepicks_data()
        if not data:
            return pd.DataFrame(columns=columns)
        
        filtered = filter_projections(data)
        if not filtered:
            print("Warning: No NBA projections found")
            return pd.DataFrame(columns=columns)
        
        df = pd.DataFrame(filtered)[columns]
        print(f"PrizePicks: Retrieved {len(df)} NBA projections")
        return df
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return pd.DataFrame(columns=columns)

def main():
    """Main execution"""
    df = get_prizepicks_df()
    if df.empty:
        sys.exit(1)

if __name__ == "__main__":
    main()
