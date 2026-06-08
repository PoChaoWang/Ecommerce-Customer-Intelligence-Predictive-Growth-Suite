{{ config(
    materialized='view'
) }}

WITH rfm AS (
    SELECT * FROM {{ ref('int_rfm_scores') }}
),

sentiment AS (
    SELECT * FROM {{ source('external_intermediate', 'int_review_sentiment_user') }}
),

users AS (
    SELECT * FROM {{ ref('stg_users') }}
),

joined AS (
    SELECT
        rfm.*,
        COALESCE(sentiment.compound_score, 0) AS compound_score,
        COALESCE(sentiment.risk_flag, 0) AS risk_flag,
        users.city,
        users.gender,
        users.signup_date
    FROM rfm
    LEFT JOIN sentiment ON rfm.user_id = sentiment.user_id
    LEFT JOIN users ON rfm.user_id = users.user_id
)

SELECT
    *,
    {{ classify_customer_segment('recency_score', 'frequency_score', 'monetary_score') }} AS segment
FROM joined
