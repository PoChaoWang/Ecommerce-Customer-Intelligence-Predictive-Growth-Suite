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
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY predicted_profit_90_days DESC) as rn
        FROM {{ source('external_intermediate', 'int_ltv_predictions') }}
    )
    WHERE rn = 1
)

SELECT
  cs.user_id AS user_id,
  cs.recency AS recency,
  cs.frequency AS frequency,
  cs.monetary AS monetary,
  cs.recency_score AS recency_score,
  cs.frequency_score AS frequency_score,
  cs.monetary_score AS monetary_score,
  cs.segment AS segment,
  cs.compound_score AS compound_score,
  cs.risk_flag AS risk_flag,
  cs.city AS city,
  cs.gender AS gender,
  cs.signup_date AS signup_date,
  ea.total_views AS total_views,
  ea.total_cart_adds AS total_cart_adds,
  lf.actual_profit AS actual_profit,
  lp.predicted_profit_90_days AS predicted_profit_90_days,

  -- MinMax scaling (approximation using window functions)
  {{ min_max_scaling('lp.predicted_profit_90_days') }} AS prob_minmax,
  
  -- Sigmoid scaling (Squashes predicted profit into a 0-1 score for easier interpretation)
  {{ sigmoid_scaling('lp.predicted_profit_90_days', 100.0) }} AS prob_sigmoid,
  
  lp.primary_driver AS primary_driver,
  lp.primary_barrier AS primary_barrier,

  -- Automation trigger
  CASE
    WHEN cs.segment = 'VVIP Loyal High-Value Customer' AND cs.risk_flag = 1
      THEN 'Trigger: VIP Concierge Outreach'
    WHEN cs.segment = 'Recent New Customer' AND ea.total_cart_adds > 5
      THEN 'Trigger: 10% Welcome Discount'
    ELSE 'Keep Current Strategy'
  END AS automation_trigger

FROM {{ ref('int_customer_segments') }} cs
LEFT JOIN {{ ref('int_event_aggregates') }} ea USING (user_id)
LEFT JOIN {{ ref('int_ltv_training_features') }} lf USING (user_id)
LEFT JOIN predictions lp USING (user_id)
