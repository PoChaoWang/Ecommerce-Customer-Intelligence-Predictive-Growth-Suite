SELECT
    seed.user_id,
    seed.segment,
    seed.recency,
    seed.frequency,
    seed.monetary,
    seed.predicted_profit_90_days,
    u.hashed_email
FROM {{ ref('mart_lookalike_seed') }} AS seed
LEFT JOIN {{ ref('stg_users_hashed') }} AS u ON seed.user_id = u.user_id
