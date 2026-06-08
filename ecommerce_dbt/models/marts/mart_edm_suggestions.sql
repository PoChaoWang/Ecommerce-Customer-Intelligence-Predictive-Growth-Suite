WITH dormant_users AS (
  SELECT user_id
  FROM {{ ref('int_customer_segments') }}
  WHERE segment = 'Dormant High-Value Customer'
)

SELECT
  p.category AS category,
  SUM(oi.quantity) AS total_quantity
FROM {{ ref('stg_order_items') }} oi
INNER JOIN dormant_users du USING (user_id)
INNER JOIN {{ ref('stg_products') }} p USING (product_id)
GROUP BY p.category
ORDER BY total_quantity DESC
LIMIT 10
