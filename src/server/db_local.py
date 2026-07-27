import os

from sqlalchemy import create_engine


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://nba:nba@localhost:5432/nba",
)

engine = create_engine(DATABASE_URL)
