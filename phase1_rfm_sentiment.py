import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np


def perform_rfm_analysis(orders):
    """Phase 1.1: Calculate Recency, Frequency, and Monetary metrics."""
    # Define reference date as the day after the latest order
    reference_date = orders["order_date"].max() + pd.Timedelta(days=1)

    rfm = (
        orders.groupby("user_id")
        .agg(
            {
                "order_date": lambda x: (reference_date - x.max()).days,
                "order_id": "count",
                "total_amount": "sum",
            }
        )
        .rename(
            columns={
                "order_date": "recency",
                "order_id": "frequency",
                "total_amount": "monetary",
            }
        )
    )

    # Scoring
    rfm["recency_score"] = pd.qcut(
        rfm["recency"], 5, labels=[5, 4, 3, 2, 1], duplicates="drop"
    )
    rfm["monetary_score"] = pd.qcut(
        rfm["monetary"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop"
    )

    # Because when dividing customers into 5 equal groups based on purchase frequency,
    # the 20%, 40%, and 60% thresholds may all fall into the same value of "1 purchase",
    # F is instead segmented using absolute purchase frequency values (cut).

    # The bins here represent:
    # (0~1], (1~2], (2~3], (3~5], (5~infinity]

    # F should not use qcut, because data skewness would become even more severe.
    # For machine learning prediction, the best practice is actually not to assign it a "score",
    # but to use the original numeric value or a transformed numeric value instead.
    f_bins = [0, 1, 2, 3, 5, np.inf]
    rfm["frequency_score"] = pd.cut(
        rfm["frequency"], bins=f_bins, labels=[1, 2, 3, 4, 5]
    )

    rfm["recency_score"] = rfm["recency_score"].astype(int)
    rfm["frequency_score"] = rfm["frequency_score"].astype(int)
    rfm["monetary_score"] = rfm["monetary_score"].astype(int)

    return rfm.reset_index()


def perform_sentiment_analysis(reviews):
    """Phase 1.2: Analyze sentiment and identify risk_flag."""
    # Ensure VADER lexicon is downloaded
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    sid = SentimentIntensityAnalyzer()

    # Calculate sentiment scores
    reviews["compound_score"] = reviews["review_text"].apply(
        lambda x: sid.polarity_scores(str(x))["compound"]
    )

    # Risk Flag: Compound Score < -0.05 and Rating <= 3
    reviews["risk_flag"] = (
        (reviews["compound_score"] < -0.05) & (reviews["rating"] <= 3)
    ).astype(int)

    # Aggregate by user
    user_sentiment = (
        reviews.groupby("user_id")
        .agg({"compound_score": "mean", "risk_flag": "max"})
        .reset_index()
    )

    return reviews, user_sentiment


def define_segments(rfm, user_sentiment):
    """Phase 1.3: Define customer segments."""
    df = rfm.merge(user_sentiment, on="user_id", how="left").fillna(
        {"risk_flag": 0, "compound_score": 0}
    )

    def segment_logic(row):
        r, f, m = row["recency_score"], row["frequency_score"], row["monetary_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "VVIP 忠誠高價值客"
        elif r <= 2 and m >= 4:
            return "沉睡的高價值客"
        elif r >= 4 and f == 1:
            return "近期新客"
        elif r <= 3 and f >= 3:
            return "即將流失的回購客"
        else:
            return "一般潛力客"

    df["segment"] = df.apply(segment_logic, axis=1)
    return df


def get_sleepy_customer_preferences(customer_df, order_items, products):
    """Phase 1.4: Top 10 product categories for sleepy customers."""
    sleepy_users = customer_df[customer_df["segment"] == "沉睡的高價值客"]["user_id"]
    sleepy_orders = order_items[order_items["user_id"].isin(sleepy_users)]
    sleepy_products = sleepy_orders.merge(products, on="product_id")

    top_10 = (
        sleepy_products.groupby("category")["quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    return top_10
