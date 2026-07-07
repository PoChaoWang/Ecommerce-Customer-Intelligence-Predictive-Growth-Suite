{{ config(
    materialized='view'
) }}

WITH latest_order AS (
    SELECT MAX(order_date) AS latest_date FROM {{ ref('stg_orders') }}
),

dates AS (
    SELECT
        latest_date,
        DATE_SUB(latest_date, INTERVAL 90 DAY) AS split_date
    FROM latest_order
),

observation_period AS (
    SELECT
        user_id,
        DATE_DIFF(d.split_date, MAX(order_date), DAY) AS recency,
        COUNT(DISTINCT order_id) AS frequency
    FROM {{ ref('stg_orders') }}, dates AS d
    WHERE order_date < d.split_date
    GROUP BY user_id, d.split_date
),

prediction_period AS (
    SELECT
        user_id,
        SUM({{ calculate_profit('total_amount') }}) AS actual_profit
    FROM {{ ref('stg_orders') }}, dates AS d
    WHERE order_date >= d.split_date
    GROUP BY user_id
),

events AS (
    SELECT * FROM {{ ref('int_event_aggregates') }}
),

sentiment AS (
    SELECT
        user_id,
        AVG(compound_score) AS compound_score
    FROM {{ source('external_intermediate', 'int_review_sentiment_user') }}
    GROUP BY user_id
)

SELECT
    o.user_id,
    o.recency,
    o.frequency,
    COALESCE(e.total_views, 0) AS total_views,
    COALESCE(e.total_cart_adds, 0) AS total_cart_adds,
    COALESCE(s.compound_score, 0) AS compound_score,
    COALESCE(p.actual_profit, 0) AS actual_profit
FROM observation_period AS o
LEFT JOIN prediction_period AS p ON o.user_id = p.user_id
LEFT JOIN events AS e ON o.user_id = e.user_id
LEFT JOIN sentiment AS s ON o.user_id = s.user_id
