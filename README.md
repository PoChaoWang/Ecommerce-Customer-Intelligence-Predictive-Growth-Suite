# E-Commerce Business Decision System

[English](README.md) | [繁體中文](README.zh-TW.md)

A full-stack data project integrating **Data Engineering**, **Data Analytics**, and **AI/ML** to transform raw e-commerce data into actionable business decisions.

> 💡 **Core Value**: The system does not just output static reports; it actively provides a "Daily Action Plan" for CRM, marketing, and product teams — directly answering: **"Who should we contact today? What offers should we give them? Who is at risk of churn?"**

---

## 🚀 Key Highlights & Engineering Metrics

* **Business-Value Driven**: Integrates RFM modeling with BigQuery ML predicted Customer Lifetime Value (LTV) to automatically generate Customer 360 profiles (C360) and high-priority CRM action recommendations.
* **High-Throughput Real-time Ingestion**: Supports real-time data streaming at **500 ~ 2,000+ EPS** (Events Per Second), processing approximately **43M ~ 170M+ records daily**, demonstrating the pipeline's stability and horizontal scalability under high-concurrency scenarios.
* **Enterprise CI/CD & Slim CI**: Employs GitHub Actions and GCP Workload Identity Federation (keyless authentication) combined with Slim CI to run tests only on modified dbt models, **reducing BigQuery node compute costs by over 90%**.
* **Zero-Query-Cost Monitoring**: Apache Airflow schedules daily runs and parses local `run_results.json` to generate execution reports and instant Slack/Email alerts, **without executing any extra queries on BigQuery**.
* **Security & Privacy Compliance (PII Protection)**: Sensitive email addresses are completely stripped in the Staging layer and standard SHA-256 hashed for irreversible masking, with fine-grained IAM dataset-level (Raw/Staging/Marts) access control managed via Terraform.
* **Production-Grade Container Orchestration (GKE & Terraform)**: Uses Terraform to spin up VPC and auto-scaling GKE clusters, deploying Kafka streaming, PostgreSQL PVC, and Airflow KubernetesExecutor (integrated with git-sync).

---

## 🛠️ Tech Stack

| Layer | Technologies & Tools |
|---|---|
| **Data Warehouse** | Google BigQuery |
| **Data Transformation** | dbt (SQL), Python |
| **Streaming Pipeline** | Apache Kafka, Apache Spark (Structured Streaming) |
| **Machine Learning** | BigQuery ML (XGBoost/Regression), Scikit-learn, Pandas |
| **Natural Language Processing (NLP)** | VADER (Sentiment Analysis) |
| **Orchestration** | Apache Airflow |
| **Infrastructure as Code (IaC)** | Terraform |
| **Container Orchestration** | Google Kubernetes Engine (GKE), Docker, Kubernetes |
| **BI & Visualization** | LightDash (Looker-compatible) |

---

## 📐 System Architecture

```
[Raw CSV History Data]         [Real-time Simulation (Faker + Python)]
       │                                     │
       │                                     ▼ (Real-time Write)
       │                              Apache Kafka (Message Queue)
       │                                     │
       │ (Batch Ingest)                      ▼ (Real-time Stream Consume)
       │                              Apache Spark (Structured Streaming)
       │                                     │
       └───────────────────┬─────────────────┘
                           ▼
                 Google BigQuery (raw_ecommerce dataset)
                           │
                           ▼
                       dbt Staging
                           │
                           ▼
                     dbt Intermediate (Feature Layer)
                           │
             ┌─────────────┴─────────────┐
             ▼                           │
    BigQuery ML / Python (ML Prediction) │
             │                           │
             ▼                           ▼
     prediction tables ──────────►   dbt Marts (Business Layer)
                                         │ (Integrate LTV for Priority Score)
                                         ▼
                             C360 / CRM Recommendation List
                                         │
                                         ▼
                             LightDash Dashboard + CRM Action
```

---

## 💾 Data & Modeling Design

### 1. dbt Layer Design
The project follows a strict three-layer dbt architecture, each with defined responsibilities:
* **Staging (`stg_`)** — Lightweight cleaning and standardization of raw tables (e.g., [stg_users.sql](ecommerce_dbt/models/staging/stg_users.sql)). No business logic here; just consistent naming and types, along with PII masking.
* **Intermediate (`int_`)** — Where the core business logic resides. This layer computes RFM scores (`int_rfm_scores`), cohort retention features (`int_cohort_base`), and LTV training features (`int_ltv_training_features`), separating logic from presentation for maximum reusability.
* **Marts (`mart_`)** — Business-facing tables built for direct business use: `mart_c360_table` (customer 360 profile) and `mart_business_recommendations`. These are what the CRM and marketing teams actually query.

### 2. Machine Learning & AI Applications
* **LTV Forecast Model (BigQuery ML)**: Using customer RFM features and historical purchase behavior, a regression model predicts each customer's expected revenue over the next 90 days. This feeds directly into budget allocation logic — high-LTV customers receive high-touch retention treatment; low-LTV customers are routed to low-cost automated communication flows.
* **NLP Sentiment Analysis (Python / VADER)**: Processes customer reviews to extract sentiment labels and product pain points. The system flags high-value customers with negative sentiment as priority cases for customer service, catching churn signals before they materialize.

---

## 🔒 Enterprise Engineering Practices

### 1. PII Security & Dataset Access Control (Data Privacy & IAM)
* **PII De-identification**: [stg_users.sql](ecommerce_dbt/models/staging/stg_users.sql) completely strips raw email fields. The dedicated [stg_users_hashed.sql](ecommerce_dbt/models/staging/stg_users_hashed.sql) standardizes emails and stores them as **SHA-256** hashes (`to_hex(sha256(...))`) for secure third-party ad network (Lookalike) matching.
* **Fine-Grained Dataset IAM (GCP BigQuery)**:
  - **`raw_ecommerce` (Raw Layer)**: Restricted to data engineers and dbt as read-only (`dataViewer`) to prevent accidental raw data modifications.
  - **`dbt_ecommerce_staging` / `intermediate` (Semi-processed Layer)**: Strictly limited to the dbt execution service account (`dataEditor`) and **completely locked** for general analysts and marketing, preventing query errors and securing PII fields from bypass reading.
  - **`dbt_ecommerce_marts` (Marts Layer)**: Granted read-only (`dataViewer`) to data analysts and marketing teams, serving as the Single Source of Truth.

### 2. Modern CI/CD & Slim CI
* **Workload Identity Federation (WIF)**: Connects GitHub Actions with GCP using dynamic, short-lived OIDC tokens for **keyless authentication**, eliminating the risk of long-lived Service Account JSON key leaks. Infrastructure is defined in [cicd.tf](terraform/cicd.tf).
* **dbt Slim CI (State-Aware Testing)**: Upon PR submission, the CI pipeline ([dbt_ci.yml](.github/workflows/dbt_ci.yml)) downloads the production `manifest.json` and runs `dbt build --select state:modified+` **only for modified models and their downstream dependencies** inside a temporary dataset (`ci_pr_<pr_num>`), minimizing BigQuery compute costs.
* **Static Code Analysis (Linting)**: Integrates `ruff` (Python) and `sqlfluff` (SQL) for static style and syntax checking.

### 3. Production-Grade Airflow Scheduling & Zero-Cost Monitoring (Airflow Orchestration)
* **Scheduling Strategy**: Runs daily at 01:00 AM (Taipei Time) as defined in [ecommerce_dbt_dag.py](airflow/dags/ecommerce_dbt_dag.py). Configured with 3 automatic retries and custom task failure alerts (`on_failure_callback`) that email formatted HTML diagnostics to the operations team.
* **Zero-Query-Cost Monitoring**: After the dbt build completes, Airflow parses the local `run_results.json` metadata artifact directly to compile execution metrics and send Email/Slack reports, **incurring zero BigQuery scan fees for daily monitoring**.

### 4. Cloud Production Deployment Blueprint (Kubernetes & IaC)
* **Terraform IaC**: Defined in [gke.tf](deploy/terraform/gke.tf), provisioning VPC networks and auto-scaling GKE clusters (1-5 nodes) on GCP.
* **Kubernetes Container Orchestration**:
  - **Airflow KubernetesExecutor**: Auto-scales task-level pods on-demand, integrated with `git-sync` to sync DAG code from GitHub every 60 seconds without rebuilding docker images.
  - **Strimzi Kafka Operator**: Spins up a 3-node ZooKeeper and Kafka cluster.
  - **PostgreSQL PVC**: Uses PersistentVolumeClaim to dynamically request GCP cloud disks for stateful storage persistence.

---

## 📊 Key Deliverables & Business Applications

| Deliverable | Business Problem Solved | Core Use Case |
|---|---|---|
| **C360 Table** | Who are my VVIPs vs. dormant customers? | Provides a complete customer profile and RFM segments for targeted campaigns. |
| **Cohort Analysis** | Are last quarter's new customers actually high-quality? | Evaluates retention curves over time to optimize user acquisition spend. |
| **LTV Forecast** | Which customers are worth investing in over the next 90 days? | Allocates marketing budget to high-LTV customers and sets up automation for low-LTV ones. |
| **Business Recommendation List** | Who should CRM contact *today*, and with what offer? | Filters the daily Top 100 attention-needed accounts based on priority score (`LTV × Churn Risk`). |
| **CRM Agent Report** | How to quickly summarize daily operations for executives? | Auto-generated daily Traditional Chinese summary report for business leaders. |

---

## 📂 Project Structure

```
/data               # Raw CSV datasets
/ecommerce_dbt      # All dbt models (staging / intermediate / marts)
/scripts            # Python ingestion and preprocessing scripts
/ipynb              # LTV modeling and sentiment analysis Jupyter Notebooks
/agents             # CRM AI agent logic
/deploy             # Cloud production blueprint (Terraform & Kubernetes Manifests)
/memo               # Architecture design docs, planning notes, and setup guides
/dashboard_preview  # Dashboard layout drafts (Executive Overview, Segmentation)
```

---

## ⚡ Local Quick Start

This project supports two local infrastructure setups:

### Option A: Using Docker Compose (Default & Lightweight)

1. **Start the Infrastructure** (Zookeeper, Kafka, Kafka UI):
   ```bash
   # Start container services
   docker compose up -d
   ```
   *Visit [http://localhost:8085](http://localhost:8085) to monitor Kafka topics and messages in real-time.*
2. **Data Initialization & Backfill** (Automatically backfills history gaps up to yesterday):
   ```bash
   python scripts/run_gap_filler.py --action gap-fill
   ```
3. **Run the Streaming Pipeline** (Open two terminal windows/tabs):
   * **Terminal 1 (Mock Data Producer)**:
     By default, it runs at the `INFO` log level, which prints status summaries every 5 seconds (optimized for high speed):
     ```bash
     python scripts/run_kafka_producer.py --delay 1.0
     ```
     If you want to view every single generated event (user signup, event, order, review) in real-time, run in `DEBUG` mode:
     ```bash
     python scripts/run_kafka_producer.py --log-level DEBUG --delay 1.0
     ```
   * **Terminal 2 (PySpark Streaming Consumer, writes to BigQuery)**:
     ```bash
     python scripts/spark_bigquery_consumer.py
     ```

### Option B: Using Kubernetes (Local K8s Test Drive)

1. **One-Click Deploy Postgres & Kafka Cluster**:
   ```bash
   # Deploy PostgreSQL and PVC storage
   kubectl apply -f deploy/k8s/postgres.yaml
   # Deploy 1-Node KRaft Kafka cluster and Topics (exposing NodePort 30094 & 30095)
   kubectl apply -f deploy/k8s/kafka.yaml
   ```
2. **Data Initialization & Backfill**:
   ```bash
   python scripts/run_gap_filler.py --action gap-fill
   ```
3. **Run the Streaming Pipeline (Specify K8s Kafka Port)** (Open two terminal windows/tabs):
   * **Terminal 1 (Mock Data Producer)**:
     ```bash
     KAFKA_BOOTSTRAP_SERVERS="localhost:30094" python scripts/run_kafka_producer.py --delay 1.0
     ```
   * **Terminal 2 (PySpark Streaming Consumer, writes to BigQuery)**:
     ```bash
     KAFKA_BOOTSTRAP_SERVERS="localhost:30094" python scripts/spark_bigquery_consumer.py
     ```

⚠️ **Note**: When switching between Option A and Option B, make sure to run `rm -rf spark_checkpoints` to delete the old Spark checkpoint directory. Otherwise, the Spark job will fail due to offset mismatches from the different Kafka clusters.

---

*Built as a portfolio project demonstrating end-to-end data engineering, Platform/DataOps, and applied machine learning in a commercial context.*