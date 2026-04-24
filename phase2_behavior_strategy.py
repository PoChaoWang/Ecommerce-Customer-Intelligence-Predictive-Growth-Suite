import pandas as pd
from collections import Counter
import re


def aggregate_events(events):
    """Phase 2.1: Aggregate behavior data into user features."""
    event_counts = (
        events.groupby(["user_id", "event_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if "view" not in event_counts.columns:
        event_counts["view"] = 0
    if "cart" not in event_counts.columns:
        event_counts["cart"] = 0

    event_counts = event_counts.rename(
        columns={"view": "total_views", "cart": "total_cart_adds"}
    )
    return event_counts[["user_id", "total_views", "total_cart_adds"]]


def extract_pain_points(reviews, products):
    """
    Phase 2.2: Extract product-level insights with 30/45/60-day rolling analysis.
    Joins Review facts with Product dimensions for actionable reporting.
    """
    # Merge review data with product metadata
    # Drop 'rating' from products to avoid conflict with reviews['rating']
    products_no_rating = products.drop(columns=["rating"]) if "rating" in products.columns else products
    df = reviews.merge(products_no_rating, on="product_id", how="inner")

    # Define reference date as the latest review date in the dataset
    ref_date = df["review_date"].max()

    # Create time-window flags
    df["is_30d"] = df["review_date"] >= (ref_date - pd.Timedelta(days=30))
    df["is_45d"] = df["review_date"] >= (ref_date - pd.Timedelta(days=45))
    df["is_60d"] = df["review_date"] >= (ref_date - pd.Timedelta(days=60))

    def get_top_keywords(text_series):
        """Helper to extract top 3 keywords from negative reviews."""
        words = " ".join(text_series.astype(str)).lower()
        words = re.findall(r"\b[a-z]{4,}\b", words)  # Filter for words with length > 3
        stop_words = {
            "this",
            "that",
            "with",
            "from",
            "have",
            "they",
            "what",
            "were",
            "just",
            "very",
            "product",
            "item",
        }
        keywords = [w for w in words if w not in stop_words]

        if not keywords:
            return "N/A"
        return ", ".join([word for word, count in Counter(keywords).most_common(3)])

    product_stats = []
    for pid, group in df.groupby("product_id"):
        info = group.iloc[0]

        # Filter groups for specific windows
        d30 = group[group["is_30d"]]
        d45 = group[group["is_45d"]]
        d60 = group[group["is_60d"]]

        # Filter for reviews marked as 'risk' (negative)
        neg_reviews = group[group["risk_flag"] == 1]["review_text"]

        product_stats.append(
            {
                "product_id": pid,
                "product_name": info["product_name"],
                "category": info["category"],
                "brand": info["brand"],
                "price": info["price"],
                "avg_rating_overall": round(group["rating"].mean(), 2),
                "rating_30d": round(d30["rating"].mean(), 2) if not d30.empty else None,
                "rating_45d": round(d45["rating"].mean(), 2) if not d45.empty else None,
                "rating_60d": round(d60["rating"].mean(), 2) if not d60.empty else None,
                "total_risk_reviews": len(neg_reviews),
                "top_pain_points": get_top_keywords(neg_reviews),
            }
        )

    return pd.DataFrame(product_stats).sort_values(
        by="total_risk_reviews", ascending=False
    )


def apply_automation_logic(df):
    """Phase 2.3: Define automation strategy triggers."""

    def trigger_logic(row):
        if row["segment"] == "VVIP 忠誠高價值客" and row["risk_flag"] == 1:
            return "Trigger: VIP Concierge Outreach"
        elif row["segment"] == "近期新客" and row["total_cart_adds"] > 5:
            return "Trigger: 10% Welcome Discount"
        return "Keep Current Strategy"

    df["automation_trigger"] = df.apply(trigger_logic, axis=1)
    return df


def generate_lookalike_seed(df):
    """
    Requirement 2: Generate high-value customer seed list for Lookalike Audience.
    """
    threshold = df["predicted_profit_90_days"].quantile(0.8)
    lookalike_df = df[
        (df["segment"] == "VVIP 忠誠高價值客") | 
        (df["predicted_profit_90_days"] >= threshold)
    ]
    
    cols = [
        "user_id", "segment", "recency", "frequency", 
        "monetary", "predicted_profit_90_days", "primary_driver"
    ]
    return lookalike_df[cols]


def generate_churn_risk_list(df):
    """
    Requirement 3: Generate churn warning list for retention campaigns.
    """
    threshold = df["predicted_profit_90_days"].quantile(0.4)
    churn_risk_df = df[
        (df["recency_score"] <= 2) & 
        (df["frequency_score"] >= 3) & 
        (df["predicted_profit_90_days"] <= threshold)
    ]
    
    cols = [
        "user_id", "segment", "recency", "recency_score", "frequency_score", 
        "predicted_profit_90_days", "primary_barrier", "risk_flag"
    ]
    return churn_risk_df[cols].sort_values(by="predicted_profit_90_days", ascending=True)


def generate_cart_abandonment_list(df, orders, events):
    """
    Requirement 4: Generate cart abandonment list for EDM/Retargeting.
    """
    # 1. Define 30-day window for active purchasers
    ref_date = orders["order_date"].max()
    active_users = orders[orders["order_date"] >= (ref_date - pd.Timedelta(days=30))]["user_id"].unique()
    
    # 2. Extract product IDs from 'cart' events
    cart_events = events[events["event_type"] == "cart"].copy()
    # Group by user and collect unique product IDs as a string
    abandoned_items = (
        cart_events.groupby("user_id")["product_id"]
        .apply(lambda x: ", ".join(map(str, sorted(set(x)))))
        .reset_index()
        .rename(columns={"product_id": "abandoned_product_ids"})
    )
    
    # 3. Filtering users who have cart adds but NO recent orders
    abandon_df = df[
        (df["total_cart_adds"] > 0) & 
        (~df["user_id"].isin(active_users))
    ].copy()
    
    # 4. Merge with abandoned items
    abandon_df = abandon_df.merge(abandoned_items, on="user_id", how="left")
    
    cols = [
        "user_id", "abandoned_product_ids", "total_cart_adds", "total_views", 
        "segment", "predicted_profit_90_days", "risk_flag"
    ]
    return abandon_df[cols].sort_values(by="total_cart_adds", ascending=False)


def generate_sleepy_product_health_report(customer_df, reviews_processed, products, order_items):
    """
    Requirement 5: Cross-analyze sleepy high-value customers with product health.
    """
    sleepy_users = customer_df[customer_df["segment"] == "沉睡的高價值客"]["user_id"]
    sleepy_product_ids = order_items[order_items["user_id"].isin(sleepy_users)]["product_id"].unique()
    full_health_report = extract_pain_points(reviews_processed, products)
    
    sleepy_health = full_health_report[full_health_report["product_id"].isin(sleepy_product_ids)].copy()
    
    def define_action(count):
        if count > 0:
            return "產品團隊檢討"
        return "考慮重新曝光或推薦給沉睡客"
    
    sleepy_health["action_required"] = sleepy_health["total_risk_reviews"].apply(define_action)
    
    cols = [
        "product_id", "product_name", "category", "brand", "price",
        "avg_rating_overall", "rating_30d", "total_risk_reviews",
        "top_pain_points", "action_required"
    ]
    return sleepy_health[cols]


def generate_brand_health_monitor(reviews_processed, products):
    """
    Requirement 9: Monitor category-level brand health in rolling windows.
    Detects sentiment dips and risk spikes.
    
    Output columns: category, avg_sentiment_score_7d/30d/60d, 
                   risk_review_ratio_7d/30d/60d, review_count_7d/30d/60d, alert_level.
    """
    df = reviews_processed.merge(products[["product_id", "category"]], on="product_id", how="inner")
    ref_date = df["review_date"].max()
    
    # Define windows
    windows = {"7d": 7, "30d": 30, "60d": 60}
    stats = []
    
    for category, group in df.groupby("category"):
        row = {"category": category}
        
        for name, days in windows.items():
            mask = group["review_date"] >= (ref_date - pd.Timedelta(days=days))
            sub = group[mask]
            
            row[f"avg_sentiment_score_{name}"] = sub["compound_score"].mean() if not sub.empty else 0
            row[f"risk_review_ratio_{name}"] = sub["risk_flag"].mean() if not sub.empty else 0
            row[f"review_count_{name}"] = len(sub)
            
        # Alert Logic
        s7, r7 = row["avg_sentiment_score_7d"], row["risk_review_ratio_7d"]
        s30, r30 = row["avg_sentiment_score_30d"], row["risk_review_ratio_30d"]
        
        if s7 < -0.2 and r7 > 0.4:
            row["alert_level"] = "🔴 高風險：立即介入"
        elif s30 < -0.1 and r30 > 0.25:
            row["alert_level"] = "🟡 中風險：持續觀察"
        else:
            row["alert_level"] = "🟢 健康"
            
        stats.append(row)
        
    result_df = pd.DataFrame(stats)
    
    # Sorting
    alert_priority = {"🔴 高風險：立即介入": 0, "🟡 中風險：持續觀察": 1, "🟢 健康": 2}
    result_df["priority"] = result_df["alert_level"].map(alert_priority)
    
    result_df = result_df.sort_values(by=["priority", "avg_sentiment_score_7d"], ascending=[True, True])
    return result_df.drop(columns=["priority"])
