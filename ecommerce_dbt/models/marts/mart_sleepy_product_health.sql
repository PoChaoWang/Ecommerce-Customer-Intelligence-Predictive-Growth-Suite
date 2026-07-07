WITH dormant_users AS (
    SELECT user_id
    FROM {{ ref('int_customer_segments') }}
    WHERE segment = 'Dormant High-Value Customer'
),

dormant_products AS (
    SELECT DISTINCT oi.product_id
    FROM {{ ref('stg_order_items') }} AS oi
    INNER JOIN dormant_users ON oi.user_id = dormant_users.user_id
)

SELECT
ph.product_id,
ph.product_name,
ph.category,
ph.brand,
ph.price,
ph.avg_rating_overall,
ph.rating_30d,
ph.total_risk_reviews,
ph.top_pain_points,
CASE
    WHEN ph.total_risk_reviews > 0 THEN 'Product Team Review'
    ELSE 'Consider Re-exposure or Recommendation to Dormant Customers'
END AS action_required
FROM {{ ref('int_product_review_health') }} AS ph
INNER JOIN dormant_products ON ph.product_id = dormant_products.product_id
