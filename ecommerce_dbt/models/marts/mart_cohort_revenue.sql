WITH cohort_agg AS (
    SELECT
        cohort_month,
        period_number,
        COUNT(DISTINCT user_id) AS n_customers,
        SUM(total_amount) AS total_revenue
    FROM {{ ref('int_cohort_base') }}
    GROUP BY cohort_month, period_number
)

SELECT
    cohort_month,
    MAX(CASE WHEN period_number = 0 THEN SAFE_DIVIDE(total_revenue, n_customers) END) AS period_0,
    MAX(CASE WHEN period_number = 1 THEN SAFE_DIVIDE(total_revenue, n_customers) END) AS period_1,
    MAX(CASE WHEN period_number = 2 THEN SAFE_DIVIDE(total_revenue, n_customers) END) AS period_2,
    MAX(CASE WHEN period_number = 3 THEN SAFE_DIVIDE(total_revenue, n_customers) END) AS period_3,
    MAX(CASE WHEN period_number = 4 THEN SAFE_DIVIDE(total_revenue, n_customers) END) AS period_4,
    MAX(CASE WHEN period_number = 5 THEN SAFE_DIVIDE(total_revenue, n_customers) END) AS period_5,
    MAX(CASE WHEN period_number = 6 THEN SAFE_DIVIDE(total_revenue, n_customers) END) AS period_6,
    MAX(CASE WHEN period_number = 7 THEN SAFE_DIVIDE(total_revenue, n_customers) END) AS period_7,
    MAX(CASE WHEN period_number = 8 THEN SAFE_DIVIDE(total_revenue, n_customers) END) AS period_8,
    MAX(CASE WHEN period_number = 9 THEN SAFE_DIVIDE(total_revenue, n_customers) END) AS period_9,
    MAX(CASE WHEN period_number = 10 THEN SAFE_DIVIDE(total_revenue, n_customers) END) AS period_10,
    MAX(CASE WHEN period_number = 11 THEN SAFE_DIVIDE(total_revenue, n_customers) END) AS period_11,
    MAX(CASE WHEN period_number = 12 THEN SAFE_DIVIDE(total_revenue, n_customers) END) AS period_12
FROM cohort_agg
GROUP BY cohort_month
ORDER BY cohort_month
