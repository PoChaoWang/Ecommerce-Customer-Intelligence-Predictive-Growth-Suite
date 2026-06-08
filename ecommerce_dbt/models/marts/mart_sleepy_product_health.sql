WITH dormant_users AS (
  SELECT user_id
  FROM {{ ref('int_customer_segments') }}
  WHERE segment = 'Dormant High-Value Customer'
),

dormant_products AS (
  SELECT DISTINCT oi.product_id
  FROM {{ ref('stg_order_items') }} oi
  INNER JOIN dormant_users du USING (user_id)
)

SELECT
  ph.product_id AS product_id,
  ph.product_name AS product_name,
  ph.category AS category,
  ph.brand AS brand,
  ph.price AS price,
  ph.avg_rating_overall AS avg_rating_overall,
  ph.rating_30d AS rating_30d,
  ph.total_risk_reviews AS total_risk_reviews,
  ph.top_pain_points AS top_pain_points,
  CASE
    WHEN ph.total_risk_reviews > 0 THEN 'Product Team Review'
    ELSE 'Consider Re-exposure or Recommendation to Dormant Customers'
  END AS action_required
FROM {{ ref('int_product_review_health') }} ph
INNER JOIN dormant_products dp USING (product_id)
