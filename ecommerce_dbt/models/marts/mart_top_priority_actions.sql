SELECT *
FROM {{ ref('mart_business_recommendations') }}
LIMIT 100
