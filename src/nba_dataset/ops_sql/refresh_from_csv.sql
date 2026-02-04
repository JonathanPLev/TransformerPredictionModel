BEGIN;

-- Clear cleaned tables (daily refresh strategy)
TRUNCATE games, player_statistics, injury_data RESTART IDENTITY;

-- Drop raw tables if they exist
DROP TABLE IF EXISTS games_raw;
DROP TABLE IF EXISTS player_statistics_raw;

-- Raw games table (matches Kaggle CSV)
CREATE TABLE games_raw (
  gameId TEXT,
  gameDateTimeEst TEXT,
  hometeamCity TEXT,
  hometeamName TEXT,
  hometeamId TEXT,
  awayteamCity TEXT,
  awayteamName TEXT,
  awayteamId TEXT,
  homeScore TEXT,
  awayScore TEXT,
  winner TEXT,
  gameType TEXT,
  gameSubtype TEXT,
  gameLabel TEXT,
  gameSubLabel TEXT,
  seriesGameNumber TEXT,
  attendance TEXT,
  arenaId TEXT,
  arenaName TEXT,
  arenaCity TEXT,
  arenaState TEXT,
  officials TEXT
);

-- Raw player statistics table (matches Kaggle CSV)
CREATE TABLE player_statistics_raw (
  firstName TEXT,
  lastName TEXT,
  personId TEXT,
  gameId TEXT,
  gameDateTimeEst TEXT,
  playerteamCity TEXT,
  playerteamName TEXT,
  opponentteamCity TEXT,
  opponentteamName TEXT,
  gameType TEXT,
  gameLabel TEXT,
  gameSubLabel TEXT,
  seriesGameNumber TEXT,
  win TEXT,
  home TEXT,
  numMinutes TEXT,
  points TEXT,
  assists TEXT,
  blocks TEXT,
  steals TEXT,
  fieldGoalsAttempted TEXT,
  fieldGoalsMade TEXT,
  fieldGoalsPercentage TEXT,
  threePointersAttempted TEXT,
  threePointersMade TEXT,
  threePointersPercentage TEXT,
  freeThrowsAttempted TEXT,
  freeThrowsMade TEXT,
  freeThrowsPercentage TEXT,
  reboundsDefensive TEXT,
  reboundsOffensive TEXT,
  reboundsTotal TEXT,
  foulsPersonal TEXT,
  turnovers TEXT,
  plusMinusPoints TEXT
);

-- Load CSVs into raw tables
COPY games_raw
FROM '/data/Games.csv'
WITH (FORMAT csv, HEADER true, NULL '', QUOTE '"', ESCAPE '"');

COPY player_statistics_raw
FROM '/data/PlayerStatistics.csv'
WITH (FORMAT csv, HEADER true, NULL '', QUOTE '"', ESCAPE '"');

-- Injury data 
COPY injury_data(date, relinquished, spacer, notes)
FROM '/data/InjuryData.csv'
WITH (FORMAT csv, HEADER true, NULL '', QUOTE '"', ESCAPE '"');

-- Insert cleaned games 
INSERT INTO games (
  game_id,
  home_team_city,
  home_team_name,
  away_team_city,
  away_team_name
)
SELECT DISTINCT ON (game_id)
  game_id,
  home_team_city,
  home_team_name,
  away_team_city,
  away_team_name
FROM (
  SELECT
    NULLIF(gameId,'')::REAL::INT            AS game_id,
    hometeamCity                           AS home_team_city,
    hometeamName                           AS home_team_name,
    awayteamCity                           AS away_team_city,
    awayteamName                           AS away_team_name,
    NULLIF(gameDateTimeEst,'')::TIMESTAMP  AS game_ts
  FROM games_raw
  WHERE NULLIF(gameId,'') IS NOT NULL
) g
ORDER BY game_id, game_ts DESC NULLS LAST;

-- Insert cleaned player statistics 
INSERT INTO player_statistics (
  first_name,
  last_name,
  person_id,
  game_id,
  game_date,
  player_team_city,
  player_team_name,
  opponent_team_city,
  opponent_team_name,
  game_type,
  game_label,
  game_sublabel,
  series_game_number,
  win,
  home,
  num_minutes,
  points,
  assists,
  blocks,
  steals,
  field_goals_attempted,
  field_goals_made,
  field_goals_percentage,
  three_pointers_attempted,
  three_pointers_made,
  three_pointers_percentage,
  free_throws_attempted,
  free_throws_made,
  free_throws_percentage,
  rebounds_defensive,
  rebounds_offensive,
  rebounds_total,
  fouls_personal,
  turnovers,
  plus_minus_points
)
SELECT
  firstName,
  lastName,
  NULLIF(personId,'')::REAL::INT,
  NULLIF(gameId,'')::REAL::INT,
  NULLIF(gameDateTimeEst,'')::TIMESTAMP::DATE,
  playerteamCity,
  playerteamName,
  opponentteamCity,
  opponentteamName,
  gameType,
  gameLabel,
  gameSubLabel,
  NULLIF(seriesGameNumber,'')::REAL::INT,
  NULLIF(win,'')::BOOLEAN,
  NULLIF(home,'')::BOOLEAN,
  NULLIF(numMinutes,'')::REAL,
  NULLIF(points,'')::REAL::INT,
  NULLIF(assists,'')::REAL::INT,
  NULLIF(blocks,'')::REAL::INT,
  NULLIF(steals,'')::REAL::INT,
  NULLIF(fieldGoalsAttempted,'')::REAL::INT,
  NULLIF(fieldGoalsMade,'')::REAL::INT,
  NULLIF(fieldGoalsPercentage,'')::REAL,
  NULLIF(threePointersAttempted,'')::REAL::INT,
  NULLIF(threePointersMade,'')::REAL::INT,
  NULLIF(threePointersPercentage,'')::REAL,
  NULLIF(freeThrowsAttempted,'')::REAL::INT,
  NULLIF(freeThrowsMade,'')::REAL::INT,
  NULLIF(freeThrowsPercentage,'')::REAL,
  NULLIF(reboundsDefensive,'')::REAL::INT,
  NULLIF(reboundsOffensive,'')::REAL::INT,
  NULLIF(reboundsTotal,'')::REAL::INT,
  NULLIF(foulsPersonal,'')::REAL::INT,
  NULLIF(turnovers,'')::REAL::INT,
  NULLIF(plusMinusPoints,'')::REAL::INT
FROM player_statistics_raw;

COMMIT;