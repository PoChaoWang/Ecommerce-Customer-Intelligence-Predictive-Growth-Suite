WITH cohort_size AS (
  SELECT
    cohort_month,
    COUNT(DISTINCT user_id) AS cohort_size
  FROM {{ ref('int_cohort_base') }}
  WHERE period_number = 0
  GROUP BY cohort_month
),

cohort_agg AS (
  SELECT
    cb.cohort_month,
    cb.period_number,
    COUNT(DISTINCT cb.user_id) AS n_customers
  FROM {{ ref('int_cohort_base') }} cb
  GROUP BY cb.cohort_month, cb.period_number
)

SELECT
  ca.cohort_month AS cohort_month,
  cs.cohort_size AS cohort_size,
  MAX(CASE WHEN ca.period_number = 0  THEN SAFE_DIVIDE(ca.n_customers, cs.cohort_size) END) AS period_0,
  MAX(CASE WHEN ca.period_number = 1  THEN SAFE_DIVIDE(ca.n_customers, cs.cohort_size) END) AS period_1,
  MAX(CASE WHEN ca.period_number = 2  THEN SAFE_DIVIDE(ca.n_customers, cs.cohort_size) END) AS period_2,
  MAX(CASE WHEN ca.period_number = 3  THEN SAFE_DIVIDE(ca.n_customers, cs.cohort_size) END) AS period_3,
  MAX(CASE WHEN ca.period_number = 4  THEN SAFE_DIVIDE(ca.n_customers, cs.cohort_size) END) AS period_4,
  MAX(CASE WHEN ca.period_number = 5  THEN SAFE_DIVIDE(ca.n_customers, cs.cohort_size) END) AS period_5,
  MAX(CASE WHEN ca.period_number = 6  THEN SAFE_DIVIDE(ca.n_customers, cs.cohort_size) END) AS period_6,
  MAX(CASE WHEN ca.period_number = 7  THEN SAFE_DIVIDE(ca.n_customers, cs.cohort_size) END) AS period_7,
  MAX(CASE WHEN ca.period_number = 8  THEN SAFE_DIVIDE(ca.n_customers, cs.cohort_size) END) AS period_8,
  MAX(CASE WHEN ca.period_number = 9  THEN SAFE_DIVIDE(ca.n_customers, cs.cohort_size) END) AS period_9,
  MAX(CASE WHEN ca.period_number = 10 THEN SAFE_DIVIDE(ca.n_customers, cs.cohort_size) END) AS period_10,
  MAX(CASE WHEN ca.period_number = 11 THEN SAFE_DIVIDE(ca.n_customers, cs.cohort_size) END) AS period_11,
  MAX(CASE WHEN ca.period_number = 12 THEN SAFE_DIVIDE(ca.n_customers, cs.cohort_size) END) AS period_12
FROM cohort_agg ca
LEFT JOIN cohort_size cs USING (cohort_month)
GROUP BY ca.cohort_month, cs.cohort_size
ORDER BY ca.cohort_month
