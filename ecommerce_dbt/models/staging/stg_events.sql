with source as (
    select * from {{ source('raw_ecommerce', 'raw_events') }}
),
renamed as (
    select
        event_id,
        user_id,
        event_type,
        product_id,
        cast(event_timestamp as timestamp) as event_timestamp
    from source
    where user_id is not null
)
select * from renamed