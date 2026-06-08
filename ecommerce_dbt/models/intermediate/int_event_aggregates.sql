{{ config(
    materialized='view'
) }}

SELECT
    user_id,
    COUNTIF(event_type = 'view') AS total_views,
    COUNTIF(event_type = 'cart') AS total_cart_adds
FROM {{ ref('stg_events') }}
GROUP BY user_id
