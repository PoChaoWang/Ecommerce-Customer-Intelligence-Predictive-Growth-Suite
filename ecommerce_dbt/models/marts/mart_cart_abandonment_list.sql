SELECT
    c.user_id,
    ab.abandoned_product_ids,
    c.total_cart_adds,
    c.total_views,
    c.segment,
    c.predicted_profit_90_days,
    c.risk_flag
FROM {{ ref('mart_c360_table') }} AS c
INNER JOIN {{ ref('int_cart_abandonment_base') }} AS ab ON c.user_id = ab.user_id
ORDER BY c.total_cart_adds DESC
