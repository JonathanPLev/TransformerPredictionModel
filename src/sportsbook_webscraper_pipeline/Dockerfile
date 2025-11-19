FROM apache/airflow:3.0.0-python3.10

USER root
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

USER airflow
WORKDIR /opt/airflow

COPY pyproject.toml .
COPY uv.lock .
RUN pip install uv && uv pip install --system .

COPY airflow/dags /opt/airflow/dags
COPY airflow_pipeline /opt/airflow/airflow_pipeline

ENV PYTHONPATH="/opt/airflow/airflow_pipeline:${PYTHONPATH}"
ENV AIRFLOW_HOME=/opt/airflow

CMD ["airflow", "standalone"]
