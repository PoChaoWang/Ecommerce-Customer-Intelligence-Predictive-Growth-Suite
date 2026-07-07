WITH total_orders AS (
    SELECT COUNT(DISTINCT order_id) AS total FROM {{ ref('stg_order_items') }}
),

product_order_counts AS (
    SELECT
        product_id,
        COUNT(DISTINCT order_id) AS order_count
    FROM {{ ref('stg_order_items') }}
    GROUP BY product_id
),

pairs AS (
    SELECT
        a.product_id AS product_a_id,
        b.product_id AS product_b_id,
        COUNT(DISTINCT a.order_id) AS co_occurrence_count
    FROM {{ ref('stg_order_items') }} AS a
    INNER JOIN {{ ref('stg_order_items') }} AS b
        ON a.order_id = b.order_id AND a.product_id < b.product_id
    GROUP BY a.product_id, b.product_id
)

SELECT
    'product' AS affinity_level,
    p.product_a_id,
    pa.product_name AS product_a_name,
    pa.category AS product_a_category,
    p.product_b_id,
    pb.product_name AS product_b_name,
    pb.category AS product_b_category,
    p.co_occurrence_count,
    ROUND(SAFE_DIVIDE(p.co_occurrence_count, t.total), 4) AS support,
    ROUND(SAFE_DIVIDE(p.co_occurrence_count, poc.order_count), 4) AS confidence_a_to_b
FROM pairs AS p
CROSS JOIN total_orders AS t
INNER JOIN product_order_counts AS poc ON p.product_a_id = poc.product_id
INNER JOIN {{ ref('stg_products') }} AS pa ON p.product_a_id = pa.product_id
INNER JOIN {{ ref('stg_products') }} AS pb ON p.product_b_id = pb.product_id
WHERE
    SAFE_DIVIDE(p.co_occurrence_count, t.total) >= 0.0001
    AND p.co_occurrence_count >= 2
ORDER BY confidence_a_to_b DESC
