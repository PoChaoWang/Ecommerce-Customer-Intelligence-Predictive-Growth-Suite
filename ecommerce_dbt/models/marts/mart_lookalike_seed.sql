WITH percentile AS (
  SELECT PERCENTILE_CONT(predicted_profit_90_days, 0.8) OVER () AS p80
  FROM {{ ref('mart_customer_360') }}
  LIMIT 1
)

SELECT
  c.user_id AS user_id,
  c.segment AS segment,
  c.recency AS recency,
  c.frequency AS frequency,
  c.monetary AS monetary,
  c.predicted_profit_90_days AS predicted_profit_90_days
FROM {{ ref('mart_customer_360') }} c, percentile
WHERE c.segment = 'VVIP Loyal High-Value Customer'
OR c.predicted_profit_90_days >= percentile.p80
