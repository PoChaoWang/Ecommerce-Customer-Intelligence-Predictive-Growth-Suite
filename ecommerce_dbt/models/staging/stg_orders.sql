with source as (
    select * from {{ source('raw_ecommerce', 'raw_orders') }}
),
renamed as (
    select
        order_id,
        user_id,
        date(cast(order_date as timestamp)) as order_date,
        cast(total_amount as float64) as total_amount,
        order_status
    from source
)
select * from renamed