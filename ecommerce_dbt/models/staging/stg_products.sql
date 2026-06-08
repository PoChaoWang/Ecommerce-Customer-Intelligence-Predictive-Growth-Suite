with source as (
    select * from {{ source('raw_ecommerce', 'raw_products') }}
),
renamed as (
    select
        product_id,
        product_name,
        category,
        brand,
        cast(price as float64) as price,
        cast(rating as float64) as rating
    from source
)
select * from renamed