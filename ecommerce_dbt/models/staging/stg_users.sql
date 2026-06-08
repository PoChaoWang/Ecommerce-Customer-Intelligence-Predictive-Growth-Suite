with source as (
    select * from {{ source('raw_ecommerce', 'raw_users') }}
),
renamed as (
    select
        user_id,
        name,
        email,
        gender,
        city,
        date(cast(signup_date as timestamp)) as signup_date 
    from source
)
select * from renamed