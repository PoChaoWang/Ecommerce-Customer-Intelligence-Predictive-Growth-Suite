# E-Commerce Business Decision System

[English](README.md) | [繁體中文](README.zh-TW.md)

A full-stack data project integrating **Data Engineering**, **Analytics**, and **AI** to transform raw e-commerce data into actionable business recommendations — built with **BigQuery**, **dbt**, **BigQuery ML**, and **LightDash (Looker-compatible)**.

> The goal isn't reporting. It's telling the CRM, marketing, and product teams *who to contact today, and what to offer them*.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Warehouse | Google BigQuery |
| Data Transformation | dbt (SQL), Python |
| Streaming Pipeline | Apache Kafka, Apache Spark (Structured Streaming) |
| Machine Learning | BigQuery ML, Scikit-learn, Pandas |
| AI Agent | LLM (automated CRM reporting) - **[WIP]** |
| BI / Visualization | LightDash (Looker-compatible) |

---

## Architecture Overview

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

## dbt Layer Design

The project follows a strict three-layer dbt architecture — each layer has a distinct responsibility:

- **Staging (`stg_`)** — Lightweight cleaning and standardization of raw tables (e.g., `stg_users`, `stg_orders`). No business logic here; just consistent naming and types.

- **Intermediate (`int_`)** — Where the business logic lives. This layer computes RFM scores (`int_rfm_scores`), cohort retention features (`int_cohort_base`), and LTV training features (`int_ltv_training_features`). Keeping this separate from marts makes models reusable and testable.

- **Marts (`mart_`)** — Consumer-facing tables built for direct business use: `mart_c360_table` (customer 360 profile), `mart_cart_abandonment_list`, and `mart_business_recommendations`. These are what the CRM and marketing teams actually query.

This separation ensures raw data changes don't cascade unpredictably into reports, and that each transformation step is independently auditable.

---

## Machine Learning Applications

**LTV Prediction (BigQuery ML)**
Using customer RFM features and historical purchase behavior, a regression model predicts each customer's expected revenue over the next 90 days. This feeds directly into budget allocation logic — high-LTV customers receive high-touch retention treatment; low-LTV customers are routed to low-cost automated flows.

**NLP Sentiment Analysis (Python / VADER)**
Customer reviews are processed via `sentiment_analysis_to_bq.ipynb` to extract sentiment labels and product pain points. The output flags high-value customers with negative sentiment as priority cases for customer service — catching churn signals before they materialize.

---

## Key Deliverables

| Output | What it answers |
|---|---|
| **C360 Table** (RFM segments) | Who are my VVIPs vs. dormant customers? |
| **Cohort Analysis** | Are last quarter's new customers actually good customers? |
| **LTV Forecast** | Which customers are worth investing in over the next 90 days? |
| **Business Recommendation List** | Who should CRM contact *today*, and with what offer? |
| **CRM Agent Report** | Auto-generated daily summary in Traditional Chinese for team leads |

The `mart_business_recommendations` table computes a `priority_score` (LTV × Churn Risk) to surface the daily Top 100 accounts that need attention — removing the guesswork from the CRM team's morning routine.

---

## Project Structure

```
/data               # Raw CSV datasets
/ecommerce_dbt      # All dbt models (staging / intermediate / marts)
/scripts            # Python ingestion and preprocessing scripts
/ipynb              # LTV modeling and sentiment analysis notebooks
/agents             # CRM AI agent logic
/memo               # Architecture design docs and planning notes
/dashboard_preview  # Dashboard layout drafts (Executive Overview, Segmentation)
```

---

## Development Phases

The project is structured in five maturity phases:

1. **Phase 1** — RFM segmentation + sentiment labeling → C360 Table
2. **Phase 2** — Cart abandonment + churn risk action lists
3. **Phase 3** — LTV prediction model + budget allocation logic
4. **Phase 4** — Cohort retention analysis + cross-sell affinity rules
5. **Phase 5** — Decision engine integrating all signals → Business Recommendation List

---

## Data Privacy & Security

When designing the data warehouse, this project follows the **Principle of Least Privilege** and **Privacy Compliance** standards. We enforce strict access controls on both sensitive user data (PII) and the different layers of the data warehouse.

### 1. PII Data Protection (User Sensitive Information)
* **Removing Raw Email**: To prevent sensitive PII (Personally Identifiable Information) from being exposed during data analysis and modeling, the plain-text `email` column was **completely removed** from the queries in the staging model [stg_users.sql](ecommerce_dbt/models/staging/stg_users.sql).
* **Secure Hashing**: Marketing and advertising platforms (e.g., Facebook Ads, Google Ads) typically require email addresses as seed lists for audience matching and lookalike modeling. To support this securely, we built a dedicated [stg_users_hashed.sql](ecommerce_dbt/models/staging/stg_users_hashed.sql) model:
  - Standardizes emails using `lower(trim(email))`.
  - Hashes the normalized email using the highly secure **SHA-256** algorithm, storing it as a hexadecimal string via `to_hex(sha256(...))`.
  - This hash is cryptographically one-way, meeting marketing integration requirements while fully protecting user privacy.
* **Final Marketing Output**: The marketing-specific model [mart_lookalike_hashed_email.sql](ecommerce_dbt/models/marts/marketing/mart_lookalike_hashed_email.sql) contains only the `hashed_email` and segment prediction labels, allowing the marketing team to safely export and upload lists without exposing raw credentials.

### 2. Dataset-Level Access Control (GCP BigQuery IAM)
We use **Terraform** to manage fine-grained IAM bindings for each BigQuery dataset to prevent data misuse and unauthorized access:

* **Raw Dataset (`raw_ecommerce`)**:
  * **Role**: `roles/bigquery.dataViewer` (Read-only)
  * **Scope**: Granted only to the dbt Service Account and data engineering teams. This prevents raw ingest data from being accidentally modified or deleted.
* **Staging and Intermediate Datasets (`dbt_ecommerce_staging`, `dbt_ecommerce_intermediate`)**:
  * **Role**: `roles/bigquery.dataEditor` (Read/Write)
  * **Scope**: **Strictly restricted to the dbt Service Account**. General data analysts and marketing teams are **completely blocked** from these layers.
  * **Purpose**: Staging and Intermediate layers contain views and semi-processed data. Blocking user access prevents analysts from querying raw intermediate views directly (avoiding high BigQuery scan fees), prevents downstream dependency breaks if intermediate schemas change, and secures PII fields from bypass reading.
* **Marts Dataset (`dbt_ecommerce_marts`)**:
  * **Role**: The dbt Service Account is granted `dataEditor` to write/update models, while analysts and marketing teams are granted `dataViewer` (Read-only).
  * **Purpose**: Users can only access fully cleaned, aggregated, and anonymized tables in the Marts layer, guaranteeing high-performance dashboard reporting and data consistency.

---

## Local Streaming Setup & Execution Guide

This project supports a full real-time data streaming pipeline. Below are the steps to deploy and run it locally:

### 1. Install Dependencies & Spark Runtime
The Spark streaming consumer requires **Java 17 or Java 11** installed on your machine.
Activate your virtual environment and install the required Python packages:
```bash
# Run this after activating your virtual environment
pip install -r requirements.txt
```

### 2. Start Docker Containers (Kafka & Kafka UI)
Ensure Docker Desktop is running locally, then execute:
```bash
# Start Zookeeper, Kafka Broker, and Kafka UI
docker compose up -d
```
*   **Kafka UI Access**: Open your browser and navigate to **[http://localhost:8085](http://localhost:8085)** to monitor topics and messages in real-time.

### 3. Initialize & Fill History Gap (Optional)
To reset the local dataset and automatically fill the gap between the last Kaggle update date and yesterday:
```bash
# Reset local CSV files back to clean Kaggle originals
python scripts/run_gap_filler.py --action reset

# Incrementally fill the historical gap (up to yesterday)
python scripts/run_gap_filler.py --action gap-fill
```

### 4. Run the Real-Time Pipeline
Open two separate terminal windows or tabs to run the Producer and Consumer concurrently:

*   **Terminal 1 - Start the Kafka Producer**:
    Streams simulated stateful e-commerce events and orders into Kafka at a specified interval (e.g., 1 message per second):
    ```bash
    python scripts/run_kafka_producer.py --delay 1.0
    ```
*   **Terminal 2 - Start the Spark Structured Streaming Consumer**:
    Subscribes to all Kafka topics and writes streaming batches directly to BigQuery raw tables using the Storage Write API (Direct Mode):
    ```bash
    python scripts/spark_bigquery_consumer.py
    ```

*(Press `Ctrl + C` in either terminal to stop execution.)*

---

*Built as a portfolio project demonstrating end-to-end data engineering and applied ML in a business context.*