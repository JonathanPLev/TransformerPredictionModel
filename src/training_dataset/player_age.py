import requests
import pandas as pd

url = "https://stats.nba.com/stats/leaguedashplayerbiostats"
params = {
    "College": "", "Conference": "", "Country": "", "DateFrom": "", "DateTo": "",
    "Division": "", "DraftPick": "", "DraftYear": "", "GameScope": "", "GameSegment": "",
    "Height": "", "ISTRound": "", "LastNGames": 0, "LeagueID": "00", "Location": "",
    "Month": 0, "OpponentTeamID": 0, "Outcome": "", "PORound": 0, "PerMode": "PerGame",
    "Period": 0, "PlayerExperience": "", "PlayerPosition": "", "Season": "2024-25",
    "SeasonSegment": "", "SeasonType": "Regular Season", "ShotClockRange": "",
    "StarterBench": "", "TeamID": 0, "VsConference": "", "VsDivision": "", "Weight": ""
}

headers = {
    # these matter a LOT for nba stats
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

session = requests.Session()
resp = session.get(url, params=params, headers=headers)
resp.raise_for_status()

data = resp.json()
rs = data["resultSets"][0]
df = pd.DataFrame(rs["rowSet"], columns=rs["headers"])

bio = df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE", "PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT"]]
print(bio.head())