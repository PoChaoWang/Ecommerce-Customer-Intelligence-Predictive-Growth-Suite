import pandas as pd
from itertools import combinations
from collections import Counter


AFFINITY_COLUMNS = [
    "affinity_level",
    "product_a_id",
    "product_a_name",
    "product_a_category",
    "product_b_id",
    "product_b_name",
    "product_b_category",
    "co_occurrence_count",
    "support",
    "confidence_A_to_B",
]


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
    Requirement 7: Perform Affinity Analysis (Market Basket Analysis).

    The first priority is product-level affinity because it supports specific
    SKU-to-SKU recommendations. If product-level pairs are too sparse to pass
    the threshold, fall back to category-level affinity. Category-level signals
    are more stable for merchandising, bundle, and cross-sell decisions when
    the catalog has many SKUs with low pair frequency.

    Product criteria: support >= 0.0001 and co_occurrence_count >= 2.
    Category fallback criteria: support >= 0.01 and co_occurrence_count >= 5.
    Returns: df_affinity sorted by confidence_A_to_B DESC.
    """
    total_orders = order_items["order_id"].nunique()
    if total_orders == 0:
        return pd.DataFrame(columns=AFFINITY_COLUMNS)

    product_affinity = _perform_product_affinity(order_items, products, total_orders)
    if not product_affinity.empty:
        return product_affinity

    return _perform_category_affinity(order_items, products, total_orders)


def _perform_product_affinity(order_items, products, total_orders):
    """Calculate directed product-to-product affinity pairs."""
    order_groups = order_items.groupby("order_id")["product_id"].apply(list)

    pair_counts = Counter()
    product_counts = order_items.groupby("product_id")["order_id"].nunique().to_dict()

    for items in order_groups:
        unique_items = sorted(set(items))
        if len(unique_items) > 1:
            for pair in combinations(unique_items, 2):
                pair_counts[pair] += 1
                pair_counts[(pair[1], pair[0])] += 1  # Both directions for confidence A->B and B->A

    product_lookup = products.set_index("product_id")[["product_name", "category"]].to_dict("index")
    results = []
    for (a_id, b_id), count in pair_counts.items():
        support = count / total_orders
        confidence = count / product_counts.get(a_id, 1)

        if support >= 0.0001 and count >= 2:
            a_info = product_lookup.get(a_id, {})
            b_info = product_lookup.get(b_id, {})

            results.append({
                "affinity_level": "product",
                "product_a_id": a_id,
                "product_a_name": a_info.get("product_name"),
                "product_a_category": a_info.get("category"),
                "product_b_id": b_id,
                "product_b_name": b_info.get("product_name"),
                "product_b_category": b_info.get("category"),
                "co_occurrence_count": count,
                "support": round(support, 4),
                "confidence_A_to_B": round(confidence, 4)
            })

    return _format_affinity_results(results)


def _perform_category_affinity(order_items, products, total_orders):
    """
    Calculate directed category-to-category affinity pairs.

    A category is counted only once per order even if the order contains
    multiple products from the same category.
    """
    items_with_category = order_items.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left"
    ).dropna(subset=["category"])

    order_groups = items_with_category.groupby("order_id")["category"].apply(list)

    pair_counts = Counter()
    category_counts = (
        items_with_category.drop_duplicates(["order_id", "category"])
        .groupby("category")["order_id"]
        .nunique()
        .to_dict()
    )

    for categories in order_groups:
        unique_categories = sorted(set(categories))
        if len(unique_categories) > 1:
            for pair in combinations(unique_categories, 2):
                pair_counts[pair] += 1
                pair_counts[(pair[1], pair[0])] += 1  # Both directions for confidence A->B and B->A

    results = []
    for (category_a, category_b), count in pair_counts.items():
        support = count / total_orders
        confidence = count / category_counts.get(category_a, 1)

        if support >= 0.01 and count >= 5:
            results.append({
                "affinity_level": "category",
                "product_a_id": None,
                "product_a_name": None,
                "product_a_category": category_a,
                "product_b_id": None,
                "product_b_name": None,
                "product_b_category": category_b,
                "co_occurrence_count": count,
                "support": round(support, 4),
                "confidence_A_to_B": round(confidence, 4)
            })

    return _format_affinity_results(results)


def _format_affinity_results(results):
    """Return affinity results with a stable schema and ranking."""
    if not results:
        return pd.DataFrame(columns=AFFINITY_COLUMNS)

    return (
        pd.DataFrame(results)[AFFINITY_COLUMNS]
        .sort_values(by="confidence_A_to_B", ascending=False)
        .reset_index(drop=True)
    )
