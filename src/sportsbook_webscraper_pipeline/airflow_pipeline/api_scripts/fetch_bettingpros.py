import requests
import json
import pandas as pd
from datetime import datetime
import sys

def fetch_nba_props():
    """Fetch NBA props from BettingPros API"""
    url = "https://api.bettingpros.com/v3/props"
    
    params = {
        'limit': 2000,
        'sport': 'NBA',
        'market_id': '',
        'event_id': '',
        'location': 'ALL',
        'sort': 'bet_rating',
        'include_events': 'true',
        'include_selections': 'false',
        'include_markets': 'true',
        'include_books': 'true'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        return None

def parse_props(data):
    """
    Parse BettingPros API response and extract standardized prop data.
    Only includes base lines for single-player stats.
    """
    if not data or 'props' not in data:
        return []
    
    parsed_props = []
    time_scraped = datetime.now().isoformat()
    
    # Create lookup dictionaries for events and markets
    events_dict = {event['id']: event for event in data.get('events', [])}
    markets_dict = {market['id']: market for market in data.get('markets', [])}
    books_dict = {book['id']: book for book in data.get('books', [])}
    
    
    for prop in data['props']:
        try:
            prop_type = prop.get('type', '')
            if any(keyword in prop_type.lower() for keyword in ['combo', 'parlay', 'sgp', 'same game']):
                continue
            
            market_id = prop.get('market_id')
            market = markets_dict.get(market_id)
            line_type = market.get('meta', {}).get('short_label')
            line_type = line_type.replace(' + ', '+')

            participants = prop.get('participants', [])
            if isinstance(participants, list) and len(participants) > 1:
                continue
            
            # FILTER 3: Skip alternate lines (only keep main/standard lines)
            is_alternate = prop.get('is_alternate', False)
            is_main = prop.get('is_main', True)
            
            if is_alternate or not is_main:
                continue
            
            # FILTER 4: Skip boosted or promotional lines
            is_boosted = prop.get('is_boosted', False)
            is_promo = prop.get('is_promotional', False)
            
            if is_boosted or is_promo:
                continue
            
            # Get player info from participant
            participant = prop.get('participant', {})
            player_info = participant.get('player', {})
            player_name = participant.get('name', '')
            
            # FILTER 5: Ensure we have a single player name
            if not player_name:
                continue
            
            # Check if player_name contains multiple players
            if any(separator in player_name for separator in [' & ', ' and ', ' + ', ',']):
                continue
            
            # Get team info
            team = player_info.get('team', '')
            
            # Get event info
            event_id = prop.get('event_id')
            event = events_dict.get(event_id)
            game_start = None
            opponent_team = None
            
            if event:
                game_start = event.get('scheduled', '')
                
                # Get home and away teams
                visitor_id = event.get('visitor')
                home_id = event.get('home')
                participants_list = event.get('participants', [])
                
                visitor = next((p for p in participants_list if p['id'] == visitor_id), None)
                home = next((p for p in participants_list if p['id'] == home_id), None)
                
                if visitor and home:
                    visitor_team = visitor.get('team', {}).get('abbreviation', '')
                    home_team = home.get('team', {}).get('abbreviation', '')
                    
                    # Determine opponent based on player's team
                    if team and visitor_team and home_team:
                        opponent_team = home_team if team == visitor_team else visitor_team
            
            # Get line info from consensus (base line)
            over_info = prop.get('over', {})
            line_score = over_info.get('consensus_line')
            
            # Get book-specific lines if available
            book_lines = prop.get('books', [])
            
            if book_lines:
                # Process each sportsbook separately
                for book_data in book_lines:
                    # FILTER 6: Skip books with promotional or alternate lines
                    book_is_alternate = book_data.get('is_alternate', False)
                    book_is_promo = book_data.get('is_promo', False)
                    
                    if book_is_alternate or book_is_promo:
                        continue
                    
                    book_id = book_data.get('book_id')
                    book_info = books_dict.get(book_id, {})
                    sportsbook = book_info.get('name', 'Unknown')
                    
                    book_line = book_data.get('line')
                    
                    parsed_prop = {
                        'player_name': player_name,
                        'team': team,
                        'sportsbook': sportsbook,
                        'line_score': float(book_line) if book_line is not None else None,
                        'game_start': game_start,
                        'time_scraped': time_scraped,
                        'opponent_team': opponent_team,
                        'line_type': line_type,
                    }
                    
                    parsed_props.append(parsed_prop)
            else:
                # Use consensus line if no book-specific lines
                parsed_prop = {
                    'player_name': player_name,
                    'team': team,
                    'sportsbook': 'BettingPros',
                    'line_score': float(line_score) if line_score is not None else None,
                    'game_start': game_start,
                    'time_scraped': time_scraped,
                    'opponent_team': opponent_team,
                    'line_type': line_type
                }
                
                parsed_props.append(parsed_prop)
        
        except Exception as e:
            print(f"Error processing prop: {e}", file=sys.stderr)
            continue
    
    return parsed_props

def get_bettingpros_df():
    """
    Fetch BettingPros data and return as DataFrame with standardized schema.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['player_name', 'team', 'sportsbook', 
                      'line_score', 'game_start', 'time_scraped', 'opponent_team']
    """
    try:
        data = fetch_nba_props()
        
        if not data:
            print("Warning: No data retrieved from BettingPros")
            return pd.DataFrame(columns=['player_name', 'team', 'sportsbook', 
                                         'line_score', 'game_start', 'time_scraped', 'opponent_team', 'line_type'])
        
        parsed_props = parse_props(data)
        
        if not parsed_props:
            print("Warning: No props found after filtering")
            return pd.DataFrame(columns=['player_name', 'team', 'sportsbook', 
                                         'line_score', 'game_start', 'time_scraped', 'opponent_team', 'line_type'])
        
        # Convert to DataFrame with explicit column order
        df = pd.DataFrame(parsed_props)
        df = df[['player_name', 'team', 'sportsbook', 'line_score', 'game_start', 'time_scraped', 'opponent_team', 'line_type']]
        
        print(f"BettingPros: Successfully retrieved {len(df)} records")
        return df
    
    except Exception as e:
        print(f"Error fetching BettingPros data: {e}", file=sys.stderr)
        return pd.DataFrame(columns=['player_name', 'team', 'sportsbook', 
                                     'line_score', 'game_start', 'time_scraped', 'opponent_team','line_type'])



def main():
    """
    Main execution function
    """
    print("Fetching NBA player props from BettingPros...")
    
    # Fetch and parse data
    df = get_bettingpros_df()

    print(df) 
    if df.empty:
        print("No props found in response")
        sys.exit(1)
    

if __name__ == "__main__":
    main()
