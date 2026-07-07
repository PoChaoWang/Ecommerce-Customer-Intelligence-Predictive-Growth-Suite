WITH source AS (
    SELECT * FROM {{ source('raw_ecommerce', 'raw_order_items') }}
),

renamed AS (
    SELECT
        order_item_id,
        order_id,
        product_id,
        user_id,
        cast(quantity AS float64) AS quantity,
        cast(item_price AS float64) AS item_price,
        cast(item_total AS float64) AS item_total
    FROM source
)

SELECT * FROM renamed
