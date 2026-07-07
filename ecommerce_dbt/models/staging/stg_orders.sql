WITH source AS (
    SELECT * FROM {{ source('raw_ecommerce', 'raw_orders') }}
),

renamed AS (
    SELECT
        order_id,
        user_id,
        cast(total_amount AS float64) AS total_amount,
        order_status,
        date(cast(order_date AS timestamp)) AS order_date
    FROM source
)

SELECT * FROM renamed
