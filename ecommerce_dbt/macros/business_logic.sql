/* 
    Returns the filter condition for valid orders.
    Standardizes which orders are considered 'successful' across the project.
*/
{% macro is_valid_order(column_name='order_status') %}
    {{ column_name }} IN ('completed', 'shipped')
{% endmacro %}

/* 
    Calculates profit based on total amount.
    Centralizes the margin percentage and fixed cost assumptions.
*/
{% macro calculate_profit(total_amount_column, margin_rate=0.25, fixed_cost=20) %}
    ({{ total_amount_column }} * {{ margin_rate }}) - {{ fixed_cost }}
{% endmacro %}

/* 
    Assigns a customer segment based on R, F, M scores.
    Centralizes the mapping logic for customer personas.
*/
{% macro classify_customer_segment(r_score, f_score, m_score) %}
    CASE
        WHEN {{ r_score }} >= 4 AND {{ f_score }} >= 4 AND {{ m_score }} >= 4
            THEN 'VVIP Loyal High-Value Customer'
        WHEN {{ r_score }} <= 2 AND {{ m_score }} >= 4
            THEN 'Dormant High-Value Customer'
        WHEN {{ r_score }} >= 4 AND {{ f_score }} = 1
            THEN 'Recent New Customer'
        WHEN {{ r_score }} <= 3 AND {{ f_score }} >= 3
            THEN 'At-Risk Repeat Customer'
        ELSE 'General Potential Customer'
    END
{% endmacro %}

/* 
    Calculates RFM scores. 
    Standardizes NTILE vs Custom threshold logic.
*/
{% macro get_rfm_score(column, type) %}
    {% if type == 'recency' %}
        6 - NTILE(5) OVER (ORDER BY {{ column }} ASC)
    {% elif type == 'frequency' %}
        CASE
            WHEN {{ column }} <= 1 THEN 1
            WHEN {{ column }} <= 2 THEN 2
            WHEN {{ column }} <= 3 THEN 3
            WHEN {{ column }} <= 5 THEN 4
            ELSE 5
        END
    {% elif type == 'monetary' %}
        NTILE(5) OVER (ORDER BY {{ column }} ASC)
    {% endif %}
{% endmacro %}

/* 
    Gets the reference date for calculations (max date + 1 day).
*/
{% macro get_reference_date(table_ref, date_column) %}
    (SELECT DATE_ADD(MAX({{ date_column }}), INTERVAL 1 DAY) FROM {{ table_ref }})
{% endmacro %}

/* 
    Sigmoid scaling to squash values into 0-1 range.
*/
{% macro sigmoid_scaling(column, scale_factor=100.0) %}
    1 / (1 + EXP(-({{ column }} / {{ scale_factor }})))
{% endmacro %}

/* 
    Min-Max scaling using window functions.
*/
{% macro min_max_scaling(column) %}
    SAFE_DIVIDE(
      {{ column }} - MIN({{ column }}) OVER (),
      MAX({{ column }}) OVER () - MIN({{ column }}) OVER ()
    )
{% endmacro %}

/* 
    Standardizes health alert levels based on sentiment and risk.
*/
{% macro get_health_alert_level(avg_sentiment, risk_ratio) %}
    CASE
        WHEN {{ avg_sentiment }} < -0.2 AND {{ risk_ratio }} > 0.4 
            THEN 'High Risk: Immediate Intervention'
        WHEN {{ avg_sentiment }} < -0.1 AND {{ risk_ratio }} > 0.25 
            THEN 'Medium Risk: Monitor Closely'
        ELSE 'Healthy'
    END
{% endmacro %}
