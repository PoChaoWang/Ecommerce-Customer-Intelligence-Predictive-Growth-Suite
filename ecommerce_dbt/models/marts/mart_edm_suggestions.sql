WITH dormant_users AS (
    SELECT user_id
    FROM {{ ref('int_customer_segments') }}
    WHERE segment = 'Dormant High-Value Customer'
)

SELECT
    p.category,
    SUM(oi.quantity) AS total_quantity
FROM {{ ref('stg_order_items') }} AS oi
INNER JOIN dormant_users ON oi.user_id = dormant_users.user_id
INNER JOIN {{ ref('stg_products') }} AS p ON oi.product_id = p.product_id
GROUP BY p.category
ORDER BY total_quantity DESC
LIMIT 10
