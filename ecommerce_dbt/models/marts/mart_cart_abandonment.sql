SELECT
  c.user_id AS user_id,
  ab.abandoned_product_ids AS abandoned_product_ids,
  c.total_cart_adds AS total_cart_adds,
  c.total_views AS total_views,
  c.segment AS segment,
  c.predicted_profit_90_days AS predicted_profit_90_days,
  c.risk_flag AS risk_flag
FROM {{ ref('mart_customer_360') }} c
INNER JOIN {{ ref('int_cart_abandonment_base') }} ab USING (user_id)
ORDER BY c.total_cart_adds DESC
