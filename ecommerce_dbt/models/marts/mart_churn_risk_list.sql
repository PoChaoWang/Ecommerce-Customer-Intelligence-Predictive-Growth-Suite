WITH percentile AS (
    SELECT PERCENTILE_CONT(predicted_profit_90_days, 0.4) OVER () AS p40
    FROM {{ ref('mart_c360_table') }}
    LIMIT 1
)

SELECT
    c.user_id,
    c.segment,
    c.recency,
    c.recency_score,
    c.frequency_score,
    c.predicted_profit_90_days,
    c.risk_flag
FROM {{ ref('mart_c360_table') }} AS c, percentile
WHERE
    c.recency_score <= 2
    AND c.frequency_score >= 3
    AND c.predicted_profit_90_days <= percentile.p40
ORDER BY c.predicted_profit_90_days ASC
