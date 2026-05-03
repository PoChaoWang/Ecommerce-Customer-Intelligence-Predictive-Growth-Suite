import os

import pandas as pd


def _read_csv(input_path, filename):
    """Read a required phase output from the input directory."""
    file_path = os.path.join(input_path, filename)
    return pd.read_csv(file_path)


def _first_existing_column(df, candidates):
    """Return the first available column name from a list of possible aliases."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _safe_numeric(series, default=0):
    """Convert a Series to numeric and replace invalid or missing values."""
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _standardize_c360(c360):
    """
    Normalize customer 360 fields so the business rules work with both the
    requested schema and the existing project outputs.
    """
    df = c360.copy()

    segment_col = _first_existing_column(df, ["RFM_segment", "segment"])
    sentiment_col = _first_existing_column(df, ["sentiment_score", "compound_score"])
    ltv_col = _first_existing_column(df, ["predicted_LTV", "predicted_profit_90_days"])
    churn_col = _first_existing_column(df, ["churn_risk_score", "prob_sigmoid", "risk_flag"])

    df["segment"] = df[segment_col] if segment_col else "Unknown"
    df["sentiment_score"] = _safe_numeric(df[sentiment_col]) if sentiment_col else 0
    df["predicted_LTV"] = _safe_numeric(df[ltv_col]) if ltv_col else 0
    df["churn_risk_score"] = _safe_numeric(df[churn_col]) if churn_col else 0
    df["frequency"] = _safe_numeric(df["frequency"]) if "frequency" in df.columns else 0

    return df[
        [
            "user_id",
            "segment",
            "sentiment_score",
            "predicted_LTV",
            "churn_risk_score",
            "frequency",
        ]
    ]


def _standardize_cart_abandonment(cart):
    """Keep cart-abandonment signals needed for retargeting decisions."""
    df = cart.copy()
    profit_col = _first_existing_column(df, ["predicted_profit_90_days", "predicted_LTV"])

    if "abandoned_product_ids" in df.columns:
        df["has_cart_abandonment"] = df["abandoned_product_ids"].notna()
    else:
        df["has_cart_abandonment"] = True
    df["cart_predicted_profit"] = _safe_numeric(df[profit_col]) if profit_col else 0

    return df[["user_id", "has_cart_abandonment", "cart_predicted_profit"]]


def _standardize_churn_risk(churn):
    """Keep churn-risk details used to enrich retention explanations."""
    df = churn.copy()
    churn_col = _first_existing_column(df, ["churn_risk_score", "prob_sigmoid", "risk_flag"])

    df["list_churn_risk_score"] = _safe_numeric(df[churn_col]) if churn_col else 0
    df["primary_barrier"] = df["primary_barrier"] if "primary_barrier" in df.columns else ""

    return df[["user_id", "list_churn_risk_score", "primary_barrier"]]


def _standardize_lookalike_seed(lookalike):
    """Keep VVIP seed membership for lookalike expansion decisions."""
    df = lookalike.copy()
    df["lookalike_segment"] = df["segment"] if "segment" in df.columns else ""
    df["is_vvip_seed"] = df["lookalike_segment"].astype(str).str.contains(
        "VVIP", case=False, na=False
    )

    return df[["user_id", "lookalike_segment", "is_vvip_seed"]]


def _quantile_or_default(series, quantile, default=0):
    """Calculate a quantile safely for empty or all-missing numeric fields."""
    numeric = _safe_numeric(series)
    if numeric.empty:
        return default
    return numeric.quantile(quantile)


def _build_recommendation(row, thresholds):
    """
    Apply business decision rules and return the best action for the user.

    If multiple rules match, pick the action with the highest business
    precedence. This keeps the output user-level and easy for growth teams to
    operationalize.
    """
    candidates = []

    high_churn = row["churn_risk_score"] >= thresholds["high_churn"]
    high_ltv = row["predicted_LTV"] >= thresholds["high_ltv"]
    high_cart_profit = row["cart_predicted_profit"] >= thresholds["high_cart_profit"]
    high_frequency = row["frequency"] >= thresholds["high_frequency"]
    negative_sentiment = row["sentiment_score"] < 0

    if high_churn and high_ltv:
        barrier = row["primary_barrier"] if pd.notna(row["primary_barrier"]) else ""
        reason = "High churn risk and high LTV customer"
        if barrier:
            reason = f"{reason}; main barrier: {barrier}"
        candidates.append((4, "Retention Campaign (Discount + EDM)", reason))

    if row["has_cart_abandonment"] and high_cart_profit:
        candidates.append(
            (
                2,
                "Retargeting Campaign",
                "Cart abandoned with high predicted 90-day profit",
            )
        )

    if row["is_vvip_seed"]:
        candidates.append(
            (
                1,
                "Lookalike Expansion",
                "VVIP seed user suitable for audience expansion",
            )
        )

    if negative_sentiment and high_frequency:
        candidates.append(
            (
                3,
                "Customer Support Priority",
                "Negative sentiment from a high-frequency customer",
            )
        )

    if not candidates:
        return pd.Series(
            {
                "recommended_action": "Monitor",
                "reason": "No high-priority business trigger matched",
            }
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return pd.Series(
        {"recommended_action": candidates[0][1], "reason": candidates[0][2]}
    )


def generate_business_recommendations(input_path: str, output_path: str) -> None:
    """
    Convert analytical phase outputs into prioritized business recommendations.

    Args:
        input_path: Directory containing C360_Table.csv and action-list CSVs.
        output_path: Directory where recommendation CSVs will be written.
    """
    # Step 1: Load existing analytical outputs.
    c360 = _standardize_c360(_read_csv(input_path, "C360_Table.csv"))
    cart = _standardize_cart_abandonment(
        _read_csv(input_path, "Cart_Abandonment_List.csv")
    )
    churn = _standardize_churn_risk(_read_csv(input_path, "Churn_Risk_List.csv"))
    lookalike = _standardize_lookalike_seed(
        _read_csv(input_path, "Lookalike_Seed_List.csv")
    )

    # Step 2: Merge signals into one user-level decision table.
    users = (
        c360.merge(cart, on="user_id", how="left")
        .merge(churn, on="user_id", how="left")
        .merge(lookalike, on="user_id", how="left")
    )

    users["has_cart_abandonment"] = users["has_cart_abandonment"].where(
        users["has_cart_abandonment"].notna(), False
    ).astype(bool)
    users["cart_predicted_profit"] = _safe_numeric(users["cart_predicted_profit"])
    users["list_churn_risk_score"] = _safe_numeric(users["list_churn_risk_score"])
    users["primary_barrier"] = users["primary_barrier"].fillna("")
    users["is_vvip_seed"] = users["is_vvip_seed"].where(
        users["is_vvip_seed"].notna(), False
    ).astype(bool)

    # Prefer the explicit churn-risk list score when it is available.
    users["churn_risk_score"] = users["list_churn_risk_score"].where(
        users["list_churn_risk_score"] > 0, users["churn_risk_score"]
    )

    # Step 3: Define practical high-value thresholds for business rules.
    thresholds = {
        "high_ltv": _quantile_or_default(users["predicted_LTV"], 0.75),
        "high_churn": max(0.7, _quantile_or_default(users["churn_risk_score"], 0.75)),
        "high_cart_profit": _quantile_or_default(users["cart_predicted_profit"], 0.75),
        "high_frequency": _quantile_or_default(users["frequency"], 0.75),
    }

    recommendation_fields = users.apply(
        _build_recommendation, axis=1, thresholds=thresholds
    )
    users = pd.concat([users, recommendation_fields], axis=1)

    # Step 4: Priority score estimates urgency and value of intervention.
    users["priority_score"] = (
        _safe_numeric(users["predicted_LTV"]) * _safe_numeric(users["churn_risk_score"])
    )

    # Step 5: Estimated revenue assumes a simulated 10% conversion improvement.
    users["estimated_revenue"] = _safe_numeric(users["predicted_LTV"]) * 0.1

    # Step 6: Export the final recommendation table.
    output_columns = [
        "user_id",
        "recommended_action",
        "priority_score",
        "estimated_revenue",
        "segment",
        "reason",
    ]
    final_recommendations = users[output_columns].sort_values(
        by=["priority_score", "estimated_revenue"], ascending=False
    )

    os.makedirs(output_path, exist_ok=True)
    final_recommendations.to_csv(
        os.path.join(output_path, "Business_Recommendation_List.csv"), index=False
    )

    # Step 7: Export the top 100 highest-priority users for immediate action.
    final_recommendations.head(100).to_csv(
        os.path.join(output_path, "Top_Priority_Actions.csv"), index=False
    )


if __name__ == "__main__":
    generate_business_recommendations("schema", "schema")
