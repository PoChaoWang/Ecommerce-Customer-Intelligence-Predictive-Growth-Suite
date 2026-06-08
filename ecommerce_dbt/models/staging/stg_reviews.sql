with source as (
    select * from {{ source('raw_ecommerce', 'raw_reviews') }}
),
renamed as (
    select
        review_id,
        order_id,
        user_id,
        product_id,
        cast(rating as float64) as rating,
        coalesce(review_text, '') as review_text,
        date(cast(review_date as timestamp)) as review_date
    from source
)
select * from renamed