import requests
import json
import pandas as pd
from datetime import datetime
import sys

def fetch_nba_props():
    """
    Fetch NBA player props from BettingPros API
    
    Returns:
        dict: JSON response from API
    """
    url = "https://api.bettingpros.com/v3/props"
    
    params = {
        'limit': 250,
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
    Parse props data into a structured format
    
    Args:
        data (dict): Raw API response
        
    Returns:
        list: List of dictionaries containing parsed prop data
    """
    if not data or 'props' not in data:
        return []
    
    parsed_props = []
    
    # Create lookup dictionaries for events and markets
    events_dict = {event['id']: event for event in data.get('events', [])}
    markets_dict = {market['id']: market for market in data.get('markets', [])}
    
    for prop in data['props']:
        # Get player info from participant
        participant = prop.get('participant', {})
        player_info = participant.get('player', {})
        
        # Get event info
        event = events_dict.get(prop.get('event_id'))
        event_name = None
        if event:
            visitor = next((p for p in event.get('participants', []) if p['id'] == event['visitor']), {})
            home = next((p for p in event.get('participants', []) if p['id'] == event['home']), {})
            event_name = f"{visitor.get('team', {}).get('city', '')} @ {home.get('team', {}).get('city', '')}"
        
        # Get market info
        market = markets_dict.get(prop.get('market_id'))
        market_name = market.get('meta', {}).get('short_label', 'Unknown') if market else 'Unknown'
        
        # Get over/under info
        line_info = prop.get('over', {})
        projection = prop.get('projection', {})
        
        parsed_prop = {
            'timestamp': datetime.now().isoformat(),
            'player_name': participant.get('name', 'Unknown'),
            'player_team': player_info.get('team', ''),
            'player_position': player_info.get('position', ''),
            'market_name': market_name,
            'event_name': event_name,
            'event_start_time': event.get('scheduled', '') if event else '',
            'line': line_info.get('consensus_line'),
        }
        
        parsed_props.append(parsed_prop)
    
    return parsed_props

def save_to_csv(props, filename=None):
    """
    Save props data to CSV file
    
    Args:
        props (list): List of parsed prop dictionaries
        filename (str): Output filename (default: nba_props_YYYYMMDD_HHMMSS.csv)
    """
    if not props:
        print("No props to save", file=sys.stderr)
        return
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'nba_props_{timestamp}.csv'
    
    df = pd.DataFrame(props)
    df.to_csv(filename, index=False)
    print(f"Saved {len(props)} props to {filename}")

def save_to_json(props, filename=None):
    """
    Save props data to JSON file
    
    Args:
        props (list): List of parsed prop dictionaries
        filename (str): Output filename (default: nba_props_YYYYMMDD_HHMMSS.json)
    """
    if not props:
        print("No props to save", file=sys.stderr)
        return
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'nba_props_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(props, f, indent=2)
    
    print(f"Saved {len(props)} props to {filename}")

def print_summary(props):
    """
    Print a summary of the scraped props
    
    Args:
        props (list): List of parsed prop dictionaries
    """
    if not props:
        print("No props found")
        return
    
    df = pd.DataFrame(props)
    
    print("\n" + "="*80)
    print(f"NBA PROPS SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"\nTotal Props: {len(props)}")
    print(f"\nUnique Players: {df['player_name'].nunique()}")
    print(f"Unique Markets: {df['market_name'].nunique()}")
    print(f"Unique Games: {df['event_name'].nunique()}")
    
    print("\nTop 5 Markets:")
    print(df['market_name'].value_counts().head())
    

    print("="*80 + "\n")

def main():
    """
    Main execution function
    """
    print("Fetching NBA player props...")
    
    # Fetch data
    data = fetch_nba_props()
    
    if not data:
        sys.exit(1)
    
    # Parse props
    props = parse_props(data)
    
    if not props:
        print("No props found in response")
        sys.exit(1)
    
    # Print summary
    print_summary(props)
    
    # Save to files
    save_to_csv(props)
    save_to_json(props)
    
    print("Scraping completed successfully!")

if __name__ == "__main__":
    main()