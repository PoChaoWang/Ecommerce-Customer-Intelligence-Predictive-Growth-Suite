WITH source AS (
    SELECT * FROM {{ source('raw_ecommerce', 'raw_reviews') }}
),

renamed AS (
    SELECT
        review_id,
        order_id,
        user_id,
        product_id,
        cast(rating AS float64) AS rating,
        coalesce(review_text, '') AS review_text,
        date(cast(review_date AS timestamp)) AS review_date
    FROM source
)

SELECT * FROM renamed
