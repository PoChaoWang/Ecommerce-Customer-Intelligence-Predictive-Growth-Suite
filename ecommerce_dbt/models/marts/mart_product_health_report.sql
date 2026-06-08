SELECT
  product_id,
  product_name,
  category,
  brand,
  price,
  avg_rating_overall,
  rating_30d,
  rating_45d,
  rating_60d,
  total_risk_reviews,
  top_pain_points
FROM {{ ref('int_product_review_health') }}
ORDER BY total_risk_reviews DESC
