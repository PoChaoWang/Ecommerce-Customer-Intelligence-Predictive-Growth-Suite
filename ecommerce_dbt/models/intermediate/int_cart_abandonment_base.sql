{{ config(
    materialized='view'
) }}

WITH active_users AS (
    SELECT DISTINCT user_id
    FROM {{ ref('stg_orders') }}
    WHERE
        order_date >= DATE_SUB((SELECT MAX(order_date) FROM {{ ref('stg_orders') }}), INTERVAL 30 DAY)
        AND {{ is_valid_order() }}
)

SELECT
    user_id,
    STRING_AGG(DISTINCT CAST(product_id AS STRING) ORDER BY CAST(product_id AS STRING)) AS abandoned_product_ids
FROM {{ ref('stg_events') }}
WHERE
    event_type = 'cart'
    AND user_id NOT IN (SELECT user_id FROM active_users)
GROUP BY user_id
