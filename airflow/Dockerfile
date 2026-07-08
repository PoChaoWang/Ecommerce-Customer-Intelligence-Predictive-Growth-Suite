FROM apache/airflow:2.10.0-python3.12

# We don't need build-essential or root access, which simplifies the build
# and avoids apt-get cache layer issues on some Docker environments.
USER airflow

# Create a virtual environment for dbt to isolate it from Airflow dependencies
RUN python -m venv /opt/airflow/venv_dbt && \
    /opt/airflow/venv_dbt/bin/pip install --no-cache-dir \
    dbt-core>=1.7.0 \
    dbt-bigquery>=1.7.0
