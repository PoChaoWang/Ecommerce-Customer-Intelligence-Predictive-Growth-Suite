# E-Commerce Business Decision System

This project is a business decision system built for e-commerce growth. Its goal is not only to generate analytical reports or prediction models, but to transform distributed customer, transaction, browsing, cart, review, and product data into prioritized action lists and decision recommendations that marketing, CRM, and product teams can use directly.

It answers the operational questions teams face every day:

- Which customers should be prioritized for retention?
- Which cart abandonment customers are most likely to convert?
- Which high-value customers can be used as lookalike audience seeds?
- Which products or categories are creating negative customer experiences?
- Where should marketing resources be allocated first?

---

## 1. Project Overview (Business Framing)

E-commerce companies usually do not lack data. The real challenge is that data often does not turn into decisions quickly enough. When revenue drops, marketing teams may not know whether the issue comes from lower-quality new customers, churned existing customers, product experience problems, or funnel friction. CRM teams may have large member lists but no clear view of who should be contacted first. Product teams may receive negative reviews but not know which issues directly affect retention and revenue.

This project builds a data-to-action decision workflow that helps teams:

- Increase revenue by identifying high-value customers and high-conversion opportunities
- Reduce churn by detecting high-risk customers early and assigning retention actions
- Improve marketing efficiency by focusing budget on customers most likely to generate return
- Improve product experience by identifying problematic products through reviews and retention signals
- Establish action priority so teams know what to handle first

The final output is not a static analysis, but a repeatable business decision loop.

---

## 2. End-to-End Decision Loop

The system is designed as a complete closed loop:

**Data -> Insight -> Action -> Measurement -> Feedback -> Model Improvement**

### Data: Integrate Customer and Operational Data

The system integrates orders, browsing behavior, cart activity, reviews, products, and customer data into a single customer view. This allows teams to move beyond individual transactions and understand each customer's value, activity, sentiment, churn risk, and future potential.

### Insight: Turn Data Into Business Insights

The analysis answers concrete business questions, such as:

- Which customers are `VVIP Loyal High-Value Customer` or `Dormant High-Value Customer`?
- Which customers have high churn risk?
- Which cart abandonment events are worth retargeting?
- Which products are frequently purchased together?
- Which acquisition cohorts have weaker retention?

### Action: Turn Insights Into Executable Lists

The system outputs action-oriented CSV files such as churn recovery lists, cart abandonment lists, lookalike seed lists, and final recommendation lists. These lists can be handed directly to CRM, paid media, EDM, customer support, or product teams.

### Measurement: Track Action Effectiveness

Each action can be tracked through business metrics, such as:

- Did the Retention Campaign reduce churn?
- Did the Retargeting Campaign improve conversion?
- Did Lookalike Expansion improve ROAS?
- Did Customer Support Priority reduce negative reviews and refunds?

### Feedback: Feed Results Back Into Decision Rules

If a segment responds well to discounts, its priority can be increased. If a customer type responds poorly to EDM, the channel or message can be adjusted. The value of this system is its ability to continuously learn the relationship between marketing actions and customer response.

### Model Improvement: Continuously Improve Prediction and Prioritization

LTV, churn risk, customer segmentation, and recommendation rules can be updated as new data and campaign results become available. This makes the decision process operational, monitorable, and optimizable rather than a one-time analysis.

---

## 3. Action Effectiveness Review: What If the Recommendation Does Not Work?

In a real business environment, not every insight or recommended action will immediately produce results. That does not mean the analysis failed. It means the original business hypothesis needs to be revalidated.

Therefore, this system does not treat recommended actions as the end of the decision process. It treats each action as a business hypothesis that can be tracked, validated, and improved.

When an action does not achieve the expected result, the BA should diagnose it from four angles:

### 1. Insight Validation

Check whether the original insight identified the real problem.

For example, the system may identify a group of customers as high LTV and high churn risk, then recommend a retention campaign. However, if the real cause of churn is product quality, delivery experience, or after-sales service, a simple discount may not be enough to increase repurchase.

### 2. Action Fit

Check whether the recommended action actually addresses the customer's main barrier.

For example:

- If the customer's main issue is price sensitivity, a discount or coupon may work.
- If the customer's main issue is negative reviews or product experience, customer support intervention or product improvement may be more effective.
- If the customer simply abandoned a cart, retargeting ads or reminder EDM may be more suitable than a general retention campaign.

Recommended actions should therefore continue to be adjusted based on segment, predicted LTV, churn risk, sentiment signal, cart behavior, and primary barrier.

### 3. Execution Check

Check whether the action was executed correctly.

Even if the insight and strategy are reasonable, execution problems can still occur, such as:

- EDM sent to the wrong audience
- Coupon setup errors
- Incomplete ad audience upload
- Campaign message mismatch with customer pain points
- Send time misaligned with the customer purchase cycle
- Poor landing page or product page experience

When performance is weak, the team should not only review the analytical model; it must also check campaign execution.

### 4. Measurement Review

Check whether the KPI, observation window, and evaluation method are appropriate.

Different actions should use different success metrics:

| Recommended Action | Primary KPI | Diagnostic KPI |
|---|---|---|
| Retention Campaign | Repurchase Rate, Retention Rate | Open Rate, CTR, Coupon Usage |
| Retargeting Campaign | Conversion Rate, ROAS | CTR, CPA, Add-to-Cart Rate |
| Lookalike Expansion | New Customer LTV, CAC, ROAS | First Purchase Rate, Audience Quality |
| Customer Support Priority | Negative Review Reduction, Refund Rate Reduction | Contact Rate, Resolution Rate |
| Product Health Action | Rating Improvement, Return Rate Reduction | Pain Point Keyword Trend |

Without a control group, it is difficult to determine whether the improvement came from the action itself. In practice, teams should design treatment and control groups whenever possible to measure incremental impact.

### Feedback Into the Decision Loop

When an action does not work, the result should be fed into the next decision cycle:

- If a high-LTV customer group responds poorly to discounts, test a different offer or channel next time.
- If negative-sentiment customers respond better to customer support intervention, increase the priority of Customer Support Priority.
- If retargeting ROAS for cart abandonment customers is weak, recheck the product page, price, shipping cost, or audience quality.
- If a segment consistently performs poorly on EDM, lower the priority of that action to avoid wasting marketing resources.

This makes the project more than a one-time analysis. It establishes a business decision loop that can be continuously validated, learned from, and improved.

---

## 4. Phase-by-Phase Breakdown

### Phase 1: RFM Segmentation + Sentiment Analysis

**Business Objective**

Identify customer value and relationship status so teams know who is high value, who is dormant, and who needs to be reactivated.

**What Analysis Is Done**

The system builds customer segments and sentiment labels based on recency, purchase frequency, monetary value, and review sentiment.

**Output**

- `C360_Table.csv`
- `EDM_Suggestions.csv`

**Business Impact**

- CRM teams can design different communication strategies for `VVIP Loyal High-Value Customer`, `Recent New Customer`, and `Dormant High-Value Customer`
- Marketing teams can avoid applying the same discount or message to all members
- Customers with negative sentiment can be prioritized by customer support or member operations teams

### Phase 2: Behavioral Features + Marketing Action Lists

**Business Objective**

Turn browsing, cart, review, and product signals into executable marketing and product action lists.

**What Analysis Is Done**

The system integrates customer behavior data to identify cart abandonment, churn risk, high-value seed audiences, and product health issues.

**Output**

- `Cart_Abandonment_List.csv`
- `Churn_Risk_List.csv`
- `Lookalike_Seed_List.csv`
- `Product_Health_Report.csv`
- `Sleepy_Product_Health_Report.csv`
- `Brand_Health_Monitor.csv`

**Business Impact**

- Paid media teams can retarget cart abandonment customers
- CRM teams can run retention campaigns for churn-risk customers
- Growth teams can upload high-value customers to ad platforms for lookalike expansion
- Product teams can prioritize issues that create negative reviews or churn

### Phase 3: LTV Prediction Model

**Business Objective**

Estimate future customer value so teams can allocate limited budget to customers with the highest expected return.

**What Analysis Is Done**

The system predicts potential value over the next 90 days based on historical spend, interactions, segments, and behavior signals.

**Output**

- `C360_Table.csv`
- `Model_Explanations_Table.csv`
- `Model_Validation_Table.csv`
- `Model_Run_Metadata.json`

**Business Impact**

- Marketing budget can be allocated by expected value instead of evenly across all customers
- High-LTV customers can receive higher-cost retention or loyalty treatments
- Low-LTV customers can be managed through lower-cost automated communication, improving overall investment efficiency

**Model Validation**

`Model_Validation_Table.csv` is designed for the Model Validation dashboard and is used to evaluate whether the Phase 3 LTV prediction model is reliable. It answers questions such as:

- Is the model generally overpredicting or underpredicting?
- Does prediction error vary by segment?
- Do the customers with the highest predicted value actually generate higher profit?
- Are top predicted customers more valuable than the average customer?

`predicted_profit_90_days` is a model prediction and does not mean the value will definitely occur. Model validation should not only focus on overall MAE or RMSE, because the primary use case of this model is customer value ranking and marketing resource prioritization. Therefore, prediction decile validation and top predicted group performance should also be reviewed. If the actual profit of the Top 10% or Top 20% predicted customers is clearly higher than average, the model still has business decision value even if it does not perfectly predict each customer's exact amount.

`actual_profit` is filled with 0 when missing to avoid overestimating model performance by validating only customers who made purchases.

Each `Model_Validation_Table.csv` also records `validation_run_date`, `prediction_window_start`, and `prediction_window_end`. The dashboard can use these fields to show when the validation result was generated and which prediction period was used to calculate actual customer profit.

**Model Run Metadata**

`Model_Run_Metadata.json` stores model execution history, not only the latest result. Each time `main.py` runs, a new record is appended to the `runs` array, including:

- `run_id`
- `run_timestamp`
- `model_type`
- `performance_metrics`
- `feature_importance`
- `retrain_triggered`

This history is used to monitor whether model performance deteriorates over time and to support model monitoring and retraining decisions. The `retrain_triggered` logic is: if the current RMSE increases by more than 15% compared with the previous run, the current run is marked as `true`.

### Phase 4: Cohort Analysis + Product / Category Affinity Analysis

**Business Objective**

Evaluate customer lifecycle quality and identify product or category relationships to support bundles, cross-sell, add-on recommendations, and merchandising strategy.

**What Analysis Is Done**

The system observes retention and revenue performance for customers acquired in different months. It first identifies frequently co-purchased SKU pairs at the product level. If the SKU count is high and each product appears too infrequently, making product-level affinity unstable or empty, the system automatically falls back to category-level affinity to identify more stable cross-sell and bundle opportunities.

**Output**

- `Cohort_Retention.csv`
- `Cohort_Revenue.csv`
- `Product_Affinity.csv`

**Business Impact**

- Management can determine whether campaigns bring high-quality new customers or one-time low-price buyers
- Product teams can design bundles, add-on recommendations, and cross-sell strategies
- Marketing teams can use cohort performance to evaluate the long-term value of different months or campaigns
- Procurement and merchandising teams can plan campaigns, homepage placement, and recommendation modules based on category relationships

**BA Interpretation**

Product-level affinity is suitable for specific SKU-to-SKU recommendations. However, when there are many SKUs, low product frequency, and highly scattered combinations, the result may be empty or unstable. This does not necessarily indicate a data issue or the absence of pairing opportunities. It means SKU-level data is too sparse to form stable signals. Therefore, this project includes a category-level affinity fallback to identify more reliable cross-sell, bundle, and add-on recommendation directions at a higher category level.

### Phase 5: Business Decision Engine

**Business Objective**

Integrate all previous analysis results into a final business recommendation list so teams know what action to take for each customer and who should be handled first.

**What Analysis Is Done**

The system combines customer segment, LTV, churn risk, cart abandonment, sentiment, and lookalike signals, then applies business rules to generate recommended actions and priorities.

**Output**

- `Business_Recommendation_List.csv`
- `Top_Priority_Actions.csv`

**Business Impact**

- Teams do not need to interpret multiple reports manually
- Each customer has a clear recommended action
- Managers can start assigning work directly from the Top 100 priority list
- Marketing, CRM, customer support, and product teams can share the same decision basis

---

## 5. Decision Engine (Phase 5 Highlight)

Phase 5 is the core business decision layer of this project. It does not simply list analysis results; it translates analysis into "what to do next."

### How Recommendations Are Generated

The system generates recommended actions based on different customer signals:

- High churn risk + high LTV: `Retention Campaign (Discount + EDM)`
- Cart abandonment + high predicted profit: `Retargeting Campaign`
- VVIP customer: `Lookalike Expansion`
- Negative sentiment + high purchase frequency: `Customer Support Priority`

If a customer meets multiple conditions, the system selects the most important action based on business priority. For example, a high-value, high-churn-risk customer will be prioritized for retention instead of general ad retargeting.

### How Priority Is Calculated

The priority score measures how quickly a customer should be handled:

```text
priority_score = predicted_LTV x churn_risk_score
```

This has two business meanings:

- The higher the `predicted_LTV`, the greater the customer's future value
- The higher the `churn_risk_score`, the greater the risk of not taking action

Therefore, high-priority customers are usually those who are worth saving and need to be saved quickly.

### How Teams Decide What To Do First

A practical workflow can look like this:

1. The CRM Manager opens `Top_Priority_Actions.csv` every day
2. The team first handles the top 100 high-value, high-risk customers
3. Work is assigned to different teams based on `recommended_action`
4. The team uses `reason` to understand why the recommendation was made
5. After execution, conversion, repurchase, retention, and customer support outcomes are tracked

Example ownership:

- CRM: retention campaigns, EDM, coupons
- Performance Marketing: retargeting and lookalike audiences
- Customer Support: priority contact for high-value customers with negative sentiment
- Product: product pain points and categories with concentrated negative reviews

---

## 6. Dashboard Layer (Visualization & Monitoring)

The CSV outputs from this project can be connected to BI tools to create stakeholder-facing dashboards. The goal of the dashboard is not to show complex models, but to help different teams make operational decisions quickly.

### Key KPIs

Recommended dashboard KPIs include:

- Revenue: total revenue, weekly revenue, incremental revenue from campaigns
- Conversion Rate: browse-to-cart, cart-to-purchase, retargeting conversion rate
- Retention: cohort retention, repurchase rate, `Dormant High-Value Customer` reactivation rate
- ROAS: ad return and lookalike audience performance
- AOV: average order value and bundle sales performance
- Churn Risk: number of high-risk customers and high-value revenue at risk

### Dashboard Modules

**Funnel Analysis**

- Monitor drop-off points between browsing, cart addition, and purchase
- Determine whether product pages, cart flow, or retargeting strategy need optimization

**Customer Segmentation**

- Show the proportion of `VVIP Loyal High-Value Customer`, `Recent New Customer`, `Dormant High-Value Customer`, and `At-Risk Repeat Customer`
- Support CRM teams in designing segmented communication strategies

**Churn Risk Distribution**

- Monitor the number of high-risk customers and estimated revenue at risk
- Help managers allocate retention budget and customer support resources

**Top Priority Actions**

- Show customers who need immediate attention and their recommended actions
- Help teams start work each day from the same priority list

**Model Validation**

| Dashboard Section | Fields Used | Purpose |
| --- | --- | --- |
| Overall Model Performance | `absolute_error`, `squared_error`, `prediction_error`, `predicted_profit_90_days`, `actual_profit_filled` | Show MAE, RMSE, Avg Predicted Profit, Avg Actual Profit, and Bias |
| Segment-level Error | `segment`, `absolute_error`, `squared_error`, `prediction_error` | Compare model error across customer segments |
| Prediction Decile Validation | `prediction_decile_label`, `predicted_profit_90_days`, `actual_profit_filled` | Check whether higher predicted-profit deciles also have higher actual profit |
| Top Predicted Group Performance | `top_10_flag`, `top_20_flag`, `top_30_flag`, `actual_profit_filled` | Validate whether top predicted customers are more valuable than average customers |

The dashboard can also use `validation_run_date` to show when the validation result was generated, and `prediction_window_start` plus `prediction_window_end` to show the actual observation period used for actual profit.

**Product & Brand Health**

- Show products with concentrated negative reviews, category sentiment changes, and major pain points
- Help product and operations teams prioritize improvements

### Who Uses This Dashboard

**Marketing Team**

- Decide retargeting audiences
- Evaluate ROAS and conversion funnel
- Build lookalike audiences

**CRM Team**

- Execute high-value customer retention
- Arrange churn recovery campaigns
- Design segmented EDM and offer strategies

**Product Team**

- Identify product issues affecting retention and sentiment
- Adjust products, descriptions, quality, or suppliers based on pain points
- Design bundles and add-on recommendations

**Management Team**

- Track changes in revenue, retention, conversion, and high-risk customers
- Evaluate whether growth strategies create long-term value
- Decide budget allocation and cross-functional priorities

---

## 7. Stakeholder Story

### Scenario: How a Marketing Manager Uses the System After Conversion Rate Drops

On Monday morning, the Marketing Manager finds that total revenue dropped by 8% last week, while website traffic did not decline significantly. This suggests that the issue may not be insufficient traffic, but a change in conversion or customer quality.

She first opens the Funnel Analysis dashboard and finds that the cart-to-purchase conversion rate has declined and cart abandonment has increased. She then checks `Cart_Abandonment_List.csv` and finds that a group of customers has high `predicted_profit_90_days`. This means they are not low-value traffic; they have clear purchase intent and are potential high-value customers worth recovering.

She then checks Phase 5's `Top_Priority_Actions.csv` and finds that the system has already ranked the high-value, high-risk customers at the top and recommended:

- Retargeting Campaign
- Retention Campaign (Discount + EDM)
- Customer Support Priority

She splits the list into three actions:

- Run dynamic product retargeting for cart abandonment customers with high predicted profit
- Send limited-time discount EDM to customers with high LTV and high churn risk
- Assign customer support to contact frequent buyers who recently left negative reviews

One week later, she returns to the dashboard to track results:

- Conversion rate for the retargeting audience recovered
- Some of the Top 100 high-priority customers repurchased
- Negative review share decreased after customer support intervention
- Estimated churn-risk revenue declined

She then feeds the campaign results back to the team:

- High-LTV customers responded well to limited-time discounts, so the next cycle should increase priority for that group
- One category has a high abandonment rate, so the product team should review product pages and review pain points
- If lookalike audience ROAS is above average, the next cycle should increase ad budget

This workflow allows the team to move beyond simply seeing that "revenue dropped." It helps them quickly locate the problem, choose actions, measure results, and feed what they learned into the next decision cycle.

---

## 8. Business Impact Summary

The core problem this system solves is moving data from "visible" to "decision-ready, actionable, and measurable."

### Problems Solved

- Too many customer lists and no clear priority
- Marketing budget spread too broadly instead of focused on high-return opportunities
- Churn risk detected only after it happens, with no early warning
- Product negative reviews not connected to revenue impact
- Analysis stuck at the reporting layer instead of being converted into team actions

### Decision-Making Improvements

- From average marketing to segmented marketing
- From post-event review to early warning
- From single-KPI tracking to a complete decision loop
- From manual list judgment to systematic prioritization
- From department-specific interpretations to a shared decision layer

### Simulated Business Impact

If this system is introduced into daily operations, the following improvements can reasonably be expected:

- Higher recovery rate for high-value churn-risk customers
- Higher retargeting conversion rate for cart abandonment customers
- Improved ad ROAS due to better lookalike seed quality
- Higher efficiency for CRM list handling
- Earlier detection of product issues, reducing negative review and refund risk
- Faster management diagnosis of the reasons behind revenue changes

In short, this project connects data analysis, prediction models, marketing actions, and performance measurement into a complete business decision system. It helps e-commerce teams find problems faster, allocate resources more precisely, and continuously turn each action result into better decisions for the next cycle.

---

## Output Data Products

Main output files are located in `schema/`:

- `C360_Table.csv`: customer 360 view and segmentation
- `Model_Run_Metadata.json`: LTV model execution history for tracking each run's performance, feature importance, and retraining warning
- `Model_Validation_Table.csv`: dedicated table for the Model Validation dashboard, used to validate LTV prediction error, ranking ability, and actual value of high-prediction groups
- `Churn_Risk_List.csv`: churn-risk customer list
- `Cart_Abandonment_List.csv`: cart abandonment retargeting list
- `Lookalike_Seed_List.csv`: high-value lookalike audience seed list
- `Product_Health_Report.csv`: product health and pain points
- `Brand_Health_Monitor.csv`: category and brand sentiment monitoring
- `Cohort_Retention.csv`: cohort retention
- `Cohort_Revenue.csv`: cohort revenue
- `Product_Affinity.csv`: product or category affinity and bundle opportunities; if product-level data is too sparse, the system automatically uses category-level fallback
- `Business_Recommendation_List.csv`: full business recommendation list
- `Top_Priority_Actions.csv`: top 100 high-priority action list

### Model_Run_Metadata.json Schema

`Model_Run_Metadata.json` stores multiple execution records for the Phase 3 LTV model. The top-level JSON object contains a `runs` array. Each time `main.py` runs, a new run is appended instead of overwriting previous records.

| Field | Description |
| --- | --- |
| `run_id` | Model run ID in `YYYYMMDD_HHMMSS` format |
| `run_timestamp` | Model execution time |
| `model_type` | Model type used |
| `performance_metrics` | Model performance metrics for this run, such as RMSE and MAE |
| `feature_importance` | Feature importance for this run |
| `retrain_triggered` | Whether a retraining warning was triggered |

`retrain_triggered` supports model monitoring and retraining decisions. If the current RMSE increases by more than 15% compared with the previous run, the current run is marked as `true`, reminding the team to check for data drift, model degradation, or the need for retraining.

### Model_Validation_Table.csv Schema

`Model_Validation_Table.csv` supports the Model Validation dashboard and validates whether the Phase 3 LTV prediction model can provide reliable business ranking and resource allocation signals.

| Field | Description |
| --- | --- |
| `validation_run_date` | Date when this model validation table was generated |
| `prediction_window_start` | Start date of the actual validation window, used to calculate actual profit for the prediction period |
| `prediction_window_end` | End date of the actual validation window, used to calculate actual profit for the prediction period |
| `user_id` | Customer ID |
| `segment` | Customer segment |
| `predicted_profit_90_days` | Model-predicted profit for the next 90 days |
| `actual_profit` | Raw actual 90-day profit; may be empty if the customer did not purchase during the prediction period |
| `actual_profit_filled` | Actual profit after filling missing `actual_profit` with 0, used for model validation |
| `prediction_error` | Predicted profit minus actual profit; positive means overprediction, negative means underprediction |
| `absolute_error` | Absolute error, used to calculate MAE |
| `squared_error` | Squared error, used to calculate RMSE |
| `prediction_decile` | Customers split into 10 groups by predicted profit descending; 1 means Top 10% |
| `prediction_decile_label` | Readable decile label |
| `top_10_flag` | Whether the customer belongs to the predicted-profit Top 10% |
| `top_20_flag` | Whether the customer belongs to the predicted-profit Top 20% |
| `top_30_flag` | Whether the customer belongs to the predicted-profit Top 30% |

### Model Validation Dashboard Usage

| Dashboard Section | Fields Used | Purpose |
| --- | --- | --- |
| Overall Model Performance | `absolute_error`, `squared_error`, `prediction_error`, `predicted_profit_90_days`, `actual_profit_filled` | Show MAE, RMSE, Avg Predicted Profit, Avg Actual Profit, and Bias |
| Segment-level Error | `segment`, `absolute_error`, `squared_error`, `prediction_error` | Compare model error across customer segments |
| Prediction Decile Validation | `prediction_decile_label`, `predicted_profit_90_days`, `actual_profit_filled` | Check whether higher predicted-profit deciles also have higher actual profit |
| Top Predicted Group Performance | `top_10_flag`, `top_20_flag`, `top_30_flag`, `actual_profit_filled` | Validate whether top predicted customers are more valuable than average customers |

`validation_run_date` can show when the validation result was generated. `prediction_window_start` and `prediction_window_end` can show the actual profit observation period used for model validation, for example:

```text
This validation result is based on actual customer profit from prediction_window_start to prediction_window_end.
```

Currently, `Model_Validation_Table.csv` represents the validation result of the latest model run and is suitable for showing current model validation. If future work needs to track performance across multiple model runs, this can be extended into an append-style `Model_Validation_History.csv`.

### Output Value Labels

Customer segment and status values in the CSV files are standardized in English to support BI, CRM, ad platforms, and automation tools.

| Field | Possible Value |
| --- | --- |
| `segment` | `VVIP Loyal High-Value Customer` |
| `segment` | `Dormant High-Value Customer` |
| `segment` | `Recent New Customer` |
| `segment` | `At-Risk Repeat Customer` |
| `segment` | `General Potential Customer` |
| `alert_level` | `Healthy` |
| `alert_level` | `Medium Risk: Monitor Closely` |
| `alert_level` | `High Risk: Immediate Intervention` |
| `action_required` | `Product Team Review` |
| `action_required` | `Consider Re-exposure or Recommendation to Dormant Customers` |

### Product_Affinity.csv Schema

`Product_Affinity.csv` supports product recommendations, bundles, cross-sell, and merchandising strategy. It may contain product-level results or category-level fallback results, depending on whether SKU-level data has enough stable co-purchase signals.

| Field | Description |
| --- | --- |
| `affinity_level` | Affinity analysis level, either `product` or `category` |
| `product_a_id` | Product A ID; empty for category-level fallback |
| `product_a_name` | Product A name; empty for category-level fallback |
| `product_a_category` | Product A category or Category A |
| `product_b_id` | Product B ID; empty for category-level fallback |
| `product_b_name` | Product B name; empty for category-level fallback |
| `product_b_category` | Product B category or Category B |
| `co_occurrence_count` | Number of times A and B appear in the same order |
| `support` | Number of orders containing both A and B / total orders |
| `confidence_A_to_B` | Among orders containing A, the percentage that also contain B |

---

## How To Run

```bash
pip install -r requirements.txt
python main.py
```
