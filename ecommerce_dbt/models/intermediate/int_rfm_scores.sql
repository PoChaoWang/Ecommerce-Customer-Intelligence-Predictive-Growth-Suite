{{ config(
    materialized='view'
) }}

WITH order_summary AS (
    SELECT
        user_id,
        MAX(order_date) AS max_order_date,
        COUNT(DISTINCT order_id) AS frequency,
        SUM(total_amount) AS monetary
    FROM {{ ref('stg_orders') }}
    WHERE {{ is_valid_order() }}
    GROUP BY 1
),

rfm_base AS (
    SELECT
        s.user_id,
        DATE_DIFF({{ get_reference_date(ref('stg_orders'), 'order_date') }}, s.max_order_date, DAY) AS recency,
        s.frequency,
        s.monetary
    FROM order_summary AS s
)

SELECT
    user_id,
    recency,
    frequency,
    monetary,
    {{ get_rfm_score('recency', 'recency') }} AS recency_score,
    {{ get_rfm_score('frequency', 'frequency') }} AS frequency_score,
    {{ get_rfm_score('monetary', 'monetary') }} AS monetary_score
FROM rfm_base
