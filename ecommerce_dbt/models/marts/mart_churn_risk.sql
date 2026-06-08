WITH percentile AS (
  SELECT PERCENTILE_CONT(predicted_profit_90_days, 0.4) OVER () AS p40
  FROM {{ ref('mart_customer_360') }}
  LIMIT 1
)

SELECT
  c.user_id AS user_id,
  c.segment AS segment,
  c.recency AS recency,
  c.recency_score AS recency_score,
  c.frequency_score AS frequency_score,
  c.predicted_profit_90_days AS predicted_profit_90_days,
  c.risk_flag AS risk_flag
FROM {{ ref('mart_customer_360') }} c, percentile
WHERE c.recency_score <= 2
AND c.frequency_score >= 3
AND c.predicted_profit_90_days <= percentile.p40
ORDER BY c.predicted_profit_90_days ASC
