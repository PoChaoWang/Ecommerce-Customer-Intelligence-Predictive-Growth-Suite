WITH source AS (
    SELECT * FROM {{ source('raw_ecommerce', 'raw_users') }}
),

renamed AS (
    SELECT
        user_id,
        to_hex(sha256(lower(trim(email)))) AS hashed_email
    FROM source
)

SELECT * FROM renamed
