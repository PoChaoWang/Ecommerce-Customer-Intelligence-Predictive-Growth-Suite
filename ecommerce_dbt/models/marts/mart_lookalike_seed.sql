WITH percentile AS (
    SELECT PERCENTILE_CONT(predicted_profit_90_days, 0.8) OVER () AS p80
    FROM {{ ref('mart_c360_table') }}
    LIMIT 1
)

SELECT
    c.user_id,
    c.segment,
    c.recency,
    c.frequency,
    c.monetary,
    c.predicted_profit_90_days
FROM {{ ref('mart_c360_table') }} AS c, percentile
WHERE
    c.segment = 'VVIP Loyal High-Value Customer'
    OR c.predicted_profit_90_days >= percentile.p80
