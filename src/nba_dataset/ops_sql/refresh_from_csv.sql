BEGIN;

-- Clear raw staging tables before reloading CSVs
TRUNCATE games_raw;
TRUNCATE player_statistics_raw;


-- Load CSVs into raw tables
COPY games_raw
FROM '/data/Games.csv'
WITH (FORMAT csv, HEADER true, NULL '', QUOTE '"', ESCAPE '"');

COPY player_statistics_raw
FROM '/data/PlayerStatistics.csv'
WITH (FORMAT csv, HEADER true, NULL '', QUOTE '"', ESCAPE '"');

-- -- Injury data 
-- COPY injury_data(date, relinquished, spacer, notes)
-- FROM '/data/InjuryData.csv'
-- WITH (FORMAT csv, HEADER true, NULL '', QUOTE '"', ESCAPE '"');

-- Upsert cleaned games 
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
ORDER BY game_id, game_ts DESC NULLS LAST
ON CONFLICT (game_id) DO UPDATE
SET
  home_team_city = EXCLUDED.home_team_city,
  home_team_name = EXCLUDED.home_team_name,
  away_team_city = EXCLUDED.away_team_city,
  away_team_name = EXCLUDED.away_team_name;


-- Upsert cleaned player statistics (dedupe per person_id + game_id)
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
SELECT DISTINCT ON (person_id, game_id)
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
FROM (
  SELECT
    firstName                                  AS first_name,
    lastName                                   AS last_name,
    NULLIF(personId,'')::REAL::INT             AS person_id,
    NULLIF(gameId,'')::REAL::INT               AS game_id,
    NULLIF(gameDateTimeEst,'')::TIMESTAMP::DATE AS game_date,
    playerteamCity                             AS player_team_city,
    playerteamName                             AS player_team_name,
    opponentteamCity                           AS opponent_team_city,
    opponentteamName                           AS opponent_team_name,
    gameType                                   AS game_type,
    gameLabel                                  AS game_label,
    gameSubLabel                               AS game_sublabel,
    NULLIF(seriesGameNumber,'')::REAL::INT     AS series_game_number,
    NULLIF(win,'')::BOOLEAN                    AS win,
    NULLIF(home,'')::BOOLEAN                   AS home,
    NULLIF(numMinutes,'')::REAL                AS num_minutes,
    NULLIF(points,'')::REAL::INT               AS points,
    NULLIF(assists,'')::REAL::INT              AS assists,
    NULLIF(blocks,'')::REAL::INT               AS blocks,
    NULLIF(steals,'')::REAL::INT               AS steals,
    NULLIF(fieldGoalsAttempted,'')::REAL::INT  AS field_goals_attempted,
    NULLIF(fieldGoalsMade,'')::REAL::INT       AS field_goals_made,
    NULLIF(fieldGoalsPercentage,'')::REAL      AS field_goals_percentage,
    NULLIF(threePointersAttempted,'')::REAL::INT AS three_pointers_attempted,
    NULLIF(threePointersMade,'')::REAL::INT    AS three_pointers_made,
    NULLIF(threePointersPercentage,'')::REAL   AS three_pointers_percentage,
    NULLIF(freeThrowsAttempted,'')::REAL::INT  AS free_throws_attempted,
    NULLIF(freeThrowsMade,'')::REAL::INT       AS free_throws_made,
    NULLIF(freeThrowsPercentage,'')::REAL      AS free_throws_percentage,
    NULLIF(reboundsDefensive,'')::REAL::INT    AS rebounds_defensive,
    NULLIF(reboundsOffensive,'')::REAL::INT    AS rebounds_offensive,
    NULLIF(reboundsTotal,'')::REAL::INT        AS rebounds_total,
    NULLIF(foulsPersonal,'')::REAL::INT        AS fouls_personal,
    NULLIF(turnovers,'')::REAL::INT            AS turnovers,
    NULLIF(plusMinusPoints,'')::REAL::INT      AS plus_minus_points,
    NULLIF(gameDateTimeEst,'')::TIMESTAMP      AS game_ts
  FROM player_statistics_raw
  WHERE NULLIF(personId,'') IS NOT NULL
    AND NULLIF(gameId,'') IS NOT NULL
) s
-- keep the most recent row if duplicates exist
ORDER BY person_id, game_id, game_ts DESC NULLS LAST
ON CONFLICT (person_id, game_id) DO UPDATE
SET
  first_name = EXCLUDED.first_name,
  last_name = EXCLUDED.last_name,
  game_date = EXCLUDED.game_date,
  player_team_city = EXCLUDED.player_team_city,
  player_team_name = EXCLUDED.player_team_name,
  opponent_team_city = EXCLUDED.opponent_team_city,
  opponent_team_name = EXCLUDED.opponent_team_name,
  game_type = EXCLUDED.game_type,
  game_label = EXCLUDED.game_label,
  game_sublabel = EXCLUDED.game_sublabel,
  series_game_number = EXCLUDED.series_game_number,
  win = EXCLUDED.win,
  home = EXCLUDED.home,
  num_minutes = EXCLUDED.num_minutes,
  points = EXCLUDED.points,
  assists = EXCLUDED.assists,
  blocks = EXCLUDED.blocks,
  steals = EXCLUDED.steals,
  field_goals_attempted = EXCLUDED.field_goals_attempted,
  field_goals_made = EXCLUDED.field_goals_made,
  field_goals_percentage = EXCLUDED.field_goals_percentage,
  three_pointers_attempted = EXCLUDED.three_pointers_attempted,
  three_pointers_made = EXCLUDED.three_pointers_made,
  three_pointers_percentage = EXCLUDED.three_pointers_percentage,
  free_throws_attempted = EXCLUDED.free_throws_attempted,
  free_throws_made = EXCLUDED.free_throws_made,
  free_throws_percentage = EXCLUDED.free_throws_percentage,
  rebounds_defensive = EXCLUDED.rebounds_defensive,
  rebounds_offensive = EXCLUDED.rebounds_offensive,
  rebounds_total = EXCLUDED.rebounds_total,
  fouls_personal = EXCLUDED.fouls_personal,
  turnovers = EXCLUDED.turnovers,
  plus_minus_points = EXCLUDED.plus_minus_points;

COMMIT;