You can find the dashboard at [HERE](https://datastudio.google.com/s/uQC9wEcXzxg)

# Customer Segmentation

## Slide Purpose

The purpose of this slide is to turn customers into actionable segments based on RFM behavior and review sentiment, helping CRM, marketing, and customer support teams understand the size, value, activity level, and risk status of each segment.

This slide is not only meant to answer "which segment is the largest." It is designed to evaluate:

- Which segments have the largest customer base and are suitable for automated or low-cost engagement
- Which segments are smaller in size but have high average customer value and may justify higher-touch retention or reactivation efforts
- Which segments show negative review signals or potential customer support risk and should be monitored more closely
- How each segment should connect to later actions such as Top Priority Actions, Retention Campaigns, EDM, or Customer Support Priority

Based on the current data, `General Potential Customer` is the largest segment and also contributes the highest total historical revenue. `VVIP Loyal High-Value Customer` and `Dormant High-Value Customer` have the highest average customer value. This means the customer strategy should not only focus on customer volume, but also consider customer value and risk.

## Data Source

Primary data source:

```text
C360_Table.csv
```

## Chart Purpose, Usage, and Business Questions

### 1. Segment Distribution

**Chart type: Donut chart**

| Setting | Value |
| --- | --- |
| Dimension | `segment` |
| Metric | `COUNT_DISTINCT(user_id)` |
| Sort | Customers descending |

**Chart purpose**

Show the overall customer structure and help users quickly understand which segments customers are mainly concentrated in.

**Usage**

This chart helps determine whether CRM or marketing should use broad-based engagement or focus on smaller, high-value segments. If a segment has a large customer share, it is usually suitable for automated EDM, segmented messaging, or other low-cost engagement methods.

**Business question**

> What does the current customer structure look like? Which segment has the largest customer base?

Currently, `General Potential Customer` is the largest segment. This indicates that most customers are still in a general potential stage and should be managed through low-cost, scalable nurturing mechanisms to improve engagement and conversion.

### 2. Revenue by Segment

**Chart type: Horizontal bar chart**

| Setting | Value |
| --- | --- |
| Dimension | `segment` |
| Metric | `SUM(monetary)` |
| Sort | Revenue descending |

**Chart purpose**

Compare the total historical revenue contribution of each segment and help the team understand which customer groups are driving revenue.

**Usage**

This chart helps distinguish between "high revenue because the segment is large" and "high revenue because the customers are individually valuable." It should be interpreted together with Avg Monetary by Segment to avoid mistaking a large segment for the highest-value segment.

**Business question**

> Which customer segments contribute the most historical revenue?

Currently, `General Potential Customer` has the highest total historical revenue, mainly because it has the largest customer base. `Dormant High-Value Customer` and `VVIP Loyal High-Value Customer` have fewer customers but still contribute significant revenue, indicating strong business value.

### 3. Avg Monetary by Segment

**Chart type: Bar chart**

| Setting | Value |
| --- | --- |
| Dimension | `segment` |
| Metric | `AVG(monetary)` |
| Sort | Avg Monetary descending |

**Chart purpose**

Compare the average historical customer value across segments and identify which segments have the highest value per customer.

**Usage**

This chart helps decide which segments justify higher-cost engagement methods, such as exclusive offers, priority customer support, VIP retention programs, or dormant customer reactivation campaigns.

**Business question**

> Which customer segments have the highest average customer value?

Currently, `VVIP Loyal High-Value Customer` has the highest average monetary value, followed by `Dormant High-Value Customer`. This shows that high-value customers are not only found in active loyal segments; some have already become dormant and should be included in reactivation strategies.

### 4. Risk Customer Rate by Segment

**Chart type: Bar chart**

| Setting | Value |
| --- | --- |
| Dimension | `segment` |
| Metric | `SUM(risk_flag) / COUNT_DISTINCT(user_id)` |
| Format | Percent |
| Sort | Sentiment Risk Rate descending |

**Chart purpose**

Compare the percentage of customers with negative sentiment risk in each segment and identify which groups need priority attention for experience or support issues.

**Usage**

This chart helps customer support, CRM, and product teams decide which customer groups should be followed up more quickly. If a high-value segment also has a high sentiment risk rate, the potential business loss is higher, and the segment should be prioritized for Customer Support Priority or Retention Campaigns.

**Business question**

> Which customer segment has the highest share of customers with negative sentiment risk?

Currently, `VVIP Loyal High-Value Customer` and `At-Risk Repeat Customer` have relatively high sentiment risk rates. This means that some high-value or repeat customers are already showing negative experience signals, which may affect future retention and repeat purchases if not addressed.
