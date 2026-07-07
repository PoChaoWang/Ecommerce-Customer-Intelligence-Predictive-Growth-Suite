WITH source AS (
    SELECT * FROM {{ source('raw_ecommerce', 'raw_events') }}
),

renamed AS (
    SELECT
        event_id,
        user_id,
        event_type,
        product_id,
        cast(event_timestamp AS timestamp) AS event_timestamp
    FROM source
    WHERE user_id IS NOT null
)

SELECT * FROM renamed
