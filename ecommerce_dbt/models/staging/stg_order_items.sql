with source as (
    select * from {{ source('raw_ecommerce', 'raw_order_items') }}
),
renamed as (
    select
        order_item_id,
        order_id,
        product_id,
        user_id,
        cast(quantity as float64) as quantity,
        cast(item_price as float64) as item_price,
        cast(item_total as float64) as item_total 
    from source
)
select * from renamed