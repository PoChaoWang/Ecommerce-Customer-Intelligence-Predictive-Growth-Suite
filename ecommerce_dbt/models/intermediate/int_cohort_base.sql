{{ config(
    materialized='view'
) }}

WITH valid_orders AS (
    SELECT
        user_id,
        order_id,
        order_date,
        total_amount
    FROM {{ ref('stg_orders') }}
    WHERE {{ is_valid_order() }}
),

user_cohort AS (
    SELECT
        user_id,
        DATE_TRUNC(MIN(order_date), MONTH) AS cohort_month
    FROM valid_orders
    GROUP BY 1
)

SELECT
    o.user_id,
    o.order_id,
    c.cohort_month,
    o.total_amount,
    DATE_TRUNC(o.order_date, MONTH) AS order_month,
    DATE_DIFF(DATE_TRUNC(o.order_date, MONTH), c.cohort_month, MONTH) AS period_number
FROM valid_orders AS o
INNER JOIN user_cohort AS c ON o.user_id = c.user_id
