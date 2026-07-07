WITH thresholds AS (
    SELECT
        PERCENTILE_CONT(predicted_profit_90_days, 0.75) OVER () AS high_ltv,
        PERCENTILE_CONT(predicted_profit_90_days, 0.75) OVER () AS high_cart_profit,
        PERCENTILE_CONT(predicted_profit_90_days, 0.75) OVER () AS high_frequency
    FROM {{ ref('mart_c360_table') }}
    LIMIT 1
),

merged AS (
    SELECT
        c.user_id,
        c.segment,
        c.predicted_profit_90_days,
        c.compound_score AS sentiment_score,
        c.frequency,
        c.risk_flag,
        COALESCE(ca.user_id IS NOT NULL, FALSE) AS has_cart_abandonment,
        COALESCE(ca.predicted_profit_90_days, 0) AS cart_predicted_profit,
        COALESCE(ls.user_id IS NOT NULL, FALSE) AS is_vvip_seed,
        COALESCE(cr.predicted_profit_90_days, 0) AS churn_risk_score
    FROM {{ ref('mart_c360_table') }} AS c
    LEFT JOIN {{ ref('mart_cart_abandonment_list') }} AS ca ON c.user_id = ca.user_id
    LEFT JOIN {{ ref('mart_lookalike_seed') }} AS ls ON c.user_id = ls.user_id
    LEFT JOIN {{ ref('mart_churn_risk_list') }} AS cr ON c.user_id = cr.user_id
)

SELECT
m.user_id,
m.segment,
CASE
    WHEN m.churn_risk_score > 0 AND m.predicted_profit_90_days >= t.high_ltv
        THEN 'Retention Campaign (Discount + EDM)'
    WHEN m.sentiment_score < 0 AND m.frequency >= t.high_frequency
        THEN 'Customer Support Priority'
    WHEN m.has_cart_abandonment AND m.cart_predicted_profit >= t.high_cart_profit
        THEN 'Retargeting Campaign'
    WHEN m.is_vvip_seed = TRUE
        THEN 'Lookalike Expansion'
    ELSE 'Monitor'
END AS recommended_action,
ROUND(m.predicted_profit_90_days * m.churn_risk_score, 4) AS priority_score,
ROUND(m.predicted_profit_90_days * 0.1, 2) AS estimated_revenue,
CASE
    WHEN m.churn_risk_score > 0 AND m.predicted_profit_90_days >= t.high_ltv
        THEN 'High churn risk and high LTV customer'
    WHEN m.sentiment_score < 0 AND m.frequency >= t.high_frequency
        THEN 'Negative sentiment from a high-frequency customer'
    WHEN m.has_cart_abandonment AND m.cart_predicted_profit >= t.high_cart_profit
        THEN 'Cart abandoned with high predicted 90-day profit'
    WHEN m.is_vvip_seed = TRUE
        THEN 'VVIP seed user suitable for audience expansion'
    ELSE 'No high-priority business trigger matched'
END AS reason
FROM merged AS m, thresholds AS t
ORDER BY priority_score DESC, estimated_revenue DESC
