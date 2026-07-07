with source as (
    select * from {{ source('raw_ecommerce', 'raw_users') }}
),
renamed as (
    select
        user_id,
        to_hex(sha256(lower(trim(email)))) as hashed_email
    from source
)
select * from renamed