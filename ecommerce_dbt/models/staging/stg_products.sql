WITH source AS (
    SELECT * FROM {{ source('raw_ecommerce', 'raw_products') }}
),

renamed AS (
    SELECT
        product_id,
        product_name,
        category,
        brand,
        cast(price AS float64) AS price,
        cast(rating AS float64) AS rating
    FROM source
)

SELECT * FROM renamed
