WITH predictions AS (
    SELECT
        user_id,
        predicted_profit_90_days,
        primary_driver,
        primary_barrier
    FROM (
        SELECT
            user_id,
            predicted_profit_90_days,
            primary_driver,
            primary_barrier,
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY predicted_profit_90_days DESC) AS rn
        FROM {{ source('external_intermediate', 'int_ltv_predictions') }}
    )
    WHERE rn = 1
)

SELECT
    cs.user_id,
    cs.recency,
    cs.frequency,
    cs.monetary,
    cs.recency_score,
    cs.frequency_score,
    cs.monetary_score,
    cs.segment,
    cs.compound_score,
    cs.risk_flag,
    cs.city,
    cs.gender,
    cs.signup_date,
    ea.total_views,
    ea.total_cart_adds,
    lf.actual_profit,
    lp.predicted_profit_90_days,

    -- MinMax scaling (approximation using window functions)
    {{ min_max_scaling('lp.predicted_profit_90_days') }} AS prob_minmax,

    -- Sigmoid scaling (Squashes predicted profit into a 0-1 score for easier interpretation)
    {{ sigmoid_scaling('lp.predicted_profit_90_days', 100.0) }} AS prob_sigmoid,

    lp.primary_driver,
    lp.primary_barrier,

    -- Automation trigger
    CASE
        WHEN cs.segment = 'VVIP Loyal High-Value Customer' AND cs.risk_flag = 1
            THEN 'Trigger: VIP Concierge Outreach'
        WHEN cs.segment = 'Recent New Customer' AND ea.total_cart_adds > 5
            THEN 'Trigger: 10% Welcome Discount'
        ELSE 'Keep Current Strategy'
    END AS automation_trigger

FROM {{ ref('int_customer_segments') }} AS cs
LEFT JOIN {{ ref('int_event_aggregates') }} AS ea ON cs.user_id = ea.user_id
LEFT JOIN {{ ref('int_ltv_training_features') }} AS lf ON cs.user_id = lf.user_id
LEFT JOIN predictions AS lp ON cs.user_id = lp.user_id
