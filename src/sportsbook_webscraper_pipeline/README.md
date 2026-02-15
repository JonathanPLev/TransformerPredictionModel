# webscrape
Dockerized Airflow pipeline that scrapes NBA player prop lines using scripts under airflow_pipeline/api_scripts, normalizes and appends the results into Postgres on docker
itself. Airflow scheduled to run the nba_props_dag daily.

---

## Prerequisites
- Docker Desktop (or Docker Engine) with Compose v2
- Optional: Python 3.10+ if you want to run the scraper scripts locally

---

## Quick start (Docker)

1. **Configure**
   ```bash
   git clone <repo>
   cd webscrape
   ```
   The Airflow container reads DB credentials from the environment (`DB_USERNAME`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`). Defaults target the Postgres service defined in `docker-compose.yml` (`line_dancer / sportsbook_data @ postgres:5432 / nba_deeplearning`). Update either `.env` or the compose file if you need different values.

2. **Build and launch**
   ```bash
   docker compose down     # don't dompose down -v: will wipe old containers/volumes, as 
                           # well as the postgres database
   docker compose up -d    # no need to build

   docker ps               # check container status
   
   ```
   This starts:
   - `postgres`: stores the scraped data (volume `postgres_data` keeps rows between runs, exposed on `localhost:5433`)
   - `airflow`: runs Airflow 3 with SequentialExecutor + SQLite metadata, but connects to the Postgres service for ETL output

3. **Access Airflow**
   - UI: http://localhost:8080  
   - Credentials: `admin` / `M7fhanqeB2mUPSpe` (set in `docker-compose.yml`; rotate before sharing externally)

4. **Trigger the DAG**
   - In the UI, unpause `nba_sportsbook_pipeline` and click “Trigger DAG”
   - Or via CLI:
     ```bash
     docker compose exec airflow airflow dags test nba_sportsbook_pipeline $(date +%Y-%m-%d)
     ```
   The DAG imports `migrate_to_postgres.py`, which runs the BettingPros, PrizePicks, and DraftEdge scrapers, validates the schema, and appends rows into the `player_lines` table inside Postgres.

5. **Inspect the database**
   ```bash
   psql -h localhost -p 5433 -U line_dancer -d nba_deeplearning
   \dt
   SELECT COUNT(*) FROM player_lines;


   docker exec -it webscrape-postgres-1 psql -U line_dancer -d nba_deeplearning;
   ```

6. **Stop / clean up**
   ```bash
   docker compose down          # keep scraped data
   docker compose down -v       # remove containers + Postgres volume // DON'T DO THIS
   ```

---

## Local development (optional)
- Install dependencies: `pip install -r requirements.txt`
- Export the same `DB_*` vars as the container and run `python airflow/dags/migrate_to_postgres.py` to append data without Airflow.

---

## Notes
- Airflow metadata stays on SQLite (default) so Postgres holds only the scraper output.
- Update secrets (`DB_PASSWORD`, Airflow admin password) before committing or sharing the project.
