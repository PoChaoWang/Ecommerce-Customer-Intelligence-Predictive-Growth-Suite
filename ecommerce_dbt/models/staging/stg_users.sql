WITH source AS (
    SELECT * FROM {{ source('raw_ecommerce', 'raw_users') }}
),

renamed AS (
    SELECT
        user_id,
        name,
        gender,
        city,
        date(cast(signup_date AS timestamp)) AS signup_date
    FROM source
)

SELECT * FROM renamed
-- test
