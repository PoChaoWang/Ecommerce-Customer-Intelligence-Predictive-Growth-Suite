WITH review_details AS (
  SELECT
    r.product_id,
    r.rating,
    r.review_date,
    sd.risk_flag
  FROM {{ ref('stg_reviews') }} r
  LEFT JOIN {{ source('external_intermediate', 'int_review_sentiment_detail') }} sd
    USING (review_id)
),

product_info AS (
  SELECT
    product_id,
    product_name,
    category,
    brand,
    price
  FROM {{ ref('stg_products') }}
),

ref_date AS (
  SELECT MAX(review_date) AS max_date FROM review_details
),

aggregated AS (
  SELECT
    rd.product_id,
    AVG(rd.rating) AS avg_rating_overall,
    AVG(CASE WHEN rd.review_date >= DATE_SUB(ref_date.max_date, INTERVAL 30 DAY) THEN rd.rating END) AS rating_30d,
    AVG(CASE WHEN rd.review_date >= DATE_SUB(ref_date.max_date, INTERVAL 45 DAY) THEN rd.rating END) AS rating_45d,
    AVG(CASE WHEN rd.review_date >= DATE_SUB(ref_date.max_date, INTERVAL 60 DAY) THEN rd.rating END) AS rating_60d,
    SUM(rd.risk_flag) AS total_risk_reviews
  FROM review_details rd
  CROSS JOIN ref_date
  GROUP BY rd.product_id, ref_date.max_date
)

SELECT
  p.product_id,
  p.product_name,
  p.category,
  p.brand,
  p.price,
  a.avg_rating_overall,
  a.rating_30d,
  a.rating_45d,
  a.rating_60d,
  COALESCE(a.total_risk_reviews, 0) AS total_risk_reviews,
  COALESCE(pp.top_pain_points, 'N/A') AS top_pain_points
FROM product_info p
LEFT JOIN aggregated a USING (product_id)
LEFT JOIN {{ source('external_intermediate', 'int_product_pain_points') }} pp USING (product_id)
