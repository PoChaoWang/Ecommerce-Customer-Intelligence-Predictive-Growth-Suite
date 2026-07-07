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