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
| Machine Learning | BigQuery ML, Scikit-learn, Pandas |
| AI Agent | LLM (automated CRM reporting) - **[WIP]** |
| BI / Visualization | LightDash (Looker-compatible) |

---

## Architecture Overview

```
Raw CSVs (users, orders, products, reviews, events)
    │
    ▼  [Python ingest script]
Google BigQuery  (raw_ecommerce dataset)
    │
    ▼  [dbt]
Staging  →  Intermediate  →  Marts
    │
    ▼
BigQuery ML / Python  (LTV prediction, NLP sentiment)
    │
    ▼
LightDash Dashboard  +  Business Recommendation List
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

*Built as a portfolio project demonstrating end-to-end data engineering and applied ML in a business context.*