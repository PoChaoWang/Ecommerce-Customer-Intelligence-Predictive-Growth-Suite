import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter


def perform_cohort_analysis(orders):
    """
    Requirement 6: Perform Cohort Analysis (Retention and Revenue).
    
    Args:
        orders (pd.DataFrame): Raw orders with user_id, order_date, total_amount.
        
    Returns:
        cohort_retention (pd.DataFrame): Pivot table of retention rates.
        cohort_revenue (pd.DataFrame): Pivot table of avg revenue per user.
    """
    df = orders.copy()
    
    # 1. Define cohort month (first order month) and order month
    df["order_month"] = df["order_date"].dt.to_period("M")
    df["cohort_month"] = df.groupby("user_id")["order_date"].transform("min").dt.to_period("M")
    
    # 2. Calculate period_number (months since joining)
    df["period_number"] = (df["order_month"].astype(int) - df["cohort_month"].astype(int))
    
    # 3. Aggregate data
    cohort_group = df.groupby(["cohort_month", "period_number"]).agg(
        n_customers=("user_id", "nunique"),
        total_revenue=("total_amount", "sum")
    ).reset_index()
    
    # 4. Calculate Cohort Size (n_customers at period 0)
    cohort_size = cohort_group[cohort_group["period_number"] == 0][["cohort_month", "n_customers"]].rename(
        columns={"n_customers": "cohort_size"}
    )
    cohort_group = cohort_group.merge(cohort_size, on="cohort_month")
    
    # 5. Metrics
    cohort_group["retention_rate"] = cohort_group["n_customers"] / cohort_group["cohort_size"]
    cohort_group["avg_revenue_per_user"] = cohort_group["total_revenue"] / cohort_group["n_customers"]
    
    # 6. Pivot tables
    cohort_retention = cohort_group.pivot(index="cohort_month", columns="period_number", values="retention_rate")
    cohort_revenue = cohort_group.pivot(index="cohort_month", columns="period_number", values="avg_revenue_per_user")
    
    return cohort_retention, cohort_revenue


def perform_affinity_analysis(order_items, products):
    """
    Requirement 7: Perform Product Affinity Analysis (Market Basket Analysis).
    Identify product pairs frequently bought together.
    
    Criteria: support >= 0.01 and co_occurrence_count >= 5.
    Returns: df_affinity sorted by confidence_A_to_B DESC.
    """
    total_orders = order_items["order_id"].nunique()
    
    # 1. Group products by order
    order_groups = order_items.groupby("order_id")["product_id"].apply(list)
    
    # 2. Count co-occurrences and individual product frequencies
    pair_counts = Counter()
    product_counts = order_items.groupby("product_id")["order_id"].nunique().to_dict()
    
    for items in order_groups:
        unique_items = sorted(set(items))
        if len(unique_items) > 1:
            for pair in combinations(unique_items, 2):
                pair_counts[pair] += 1
                pair_counts[(pair[1], pair[0])] += 1  # Both directions for confidence A->B and B->A
                
    # 3. Calculate metrics
    results = []
    for (a_id, b_id), count in pair_counts.items():
        support = count / total_orders
        confidence = count / product_counts[a_id]
        
        if support >= 0.01 and count >= 5:
            # Map names and categories
            a_info = products[products["product_id"] == a_id].iloc[0]
            b_info = products[products["product_id"] == b_id].iloc[0]
            
            results.append({
                "product_a_id": a_id,
                "product_a_name": a_info["product_name"],
                "product_a_category": a_info["category"],
                "product_b_id": b_id,
                "product_b_name": b_info["product_name"],
                "product_b_category": b_info["category"],
                "co_occurrence_count": count,
                "support": round(support, 4),
                "confidence_A_to_B": round(confidence, 4)
            })
            
    if not results:
        return pd.DataFrame(columns=[
            "product_a_id", "product_a_name", "product_a_category", 
            "product_b_id", "product_b_name", "product_b_category", 
            "co_occurrence_count", "support", "confidence_A_to_B"
        ])
        
    df_affinity = pd.DataFrame(results).sort_values(by="confidence_A_to_B", ascending=False)
    return df_affinity
