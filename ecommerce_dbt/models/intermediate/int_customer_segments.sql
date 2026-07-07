{{ config(
    materialized='view'
) }}

WITH rfm AS (
    SELECT * FROM {{ ref('int_rfm_scores') }}
),

reviews AS (
    SELECT * FROM {{ ref('stg_reviews') }}
),

sentiment_detail AS (
    SELECT DISTINCT
        review_id,
        compound_score,
        risk_flag
    FROM {{ source('external_intermediate', 'int_review_sentiment_detail') }}
),

latest_sentiment AS (
    SELECT
        r.user_id,
        sd.compound_score,
        sd.risk_flag,
        ROW_NUMBER() OVER (PARTITION BY r.user_id ORDER BY r.review_date DESC) AS rn
    FROM reviews AS r
    INNER JOIN sentiment_detail AS sd ON r.review_id = sd.review_id
),

sentiment_user AS (
    SELECT
        user_id,
        compound_score,
        risk_flag
    FROM latest_sentiment
    WHERE rn = 1
),

users AS (
    SELECT * FROM {{ ref('stg_users') }}
),

joined AS (
    SELECT
        rfm.*,
        users.city,
        users.gender,
        users.signup_date,
        COALESCE(sentiment_user.compound_score, 0) AS compound_score,
        COALESCE(sentiment_user.risk_flag, 0) AS risk_flag
    FROM rfm
    LEFT JOIN sentiment_user ON rfm.user_id = sentiment_user.user_id
    LEFT JOIN users ON rfm.user_id = users.user_id
)

SELECT
    *,
    {{ classify_customer_segment('recency_score', 'frequency_score', 'monetary_score') }} AS segment
FROM joined
