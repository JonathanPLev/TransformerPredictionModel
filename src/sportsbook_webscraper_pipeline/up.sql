CREATE TABLE IF NOT EXISTS player_lines (
  id BIGSERIAL PRIMARY KEY,

  player_name TEXT NOT NULL,
  team TEXT,
  sportsbook TEXT NOT NULL,

  line_score DOUBLE PRECISION,         
  game_start TIMESTAMPTZ,              
  time_scraped TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  opponent_team TEXT,
  line_type TEXT NOT NULL               
);
