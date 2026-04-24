import pandas as pd
import warnings
import json  # NEW: For metadata export
import phase1_rfm_sentiment as p1
import phase2_behavior_strategy as p2
import phase3_ltv_model as p3
import phase4_cohort_analysis as p4

# Suppress warnings
warnings.filterwarnings("ignore")


def load_data():
    """Load raw datasets from data/ directory."""
    try:
        orders = pd.read_csv("data/orders.csv", parse_dates=["order_date"])
        products = pd.read_csv("data/products.csv")
        events = pd.read_csv("data/events.csv", parse_dates=["event_timestamp"])
        reviews = pd.read_csv("data/reviews.csv", parse_dates=["review_date"])
        order_items = pd.read_csv("data/order_items.csv")
        return orders, products, events, reviews, order_items
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure data files exist in data/ directory.")
        exit(1)


def main():
    print("--- Starting MarTech Data Analysis Workflow ---")

    # Load Data
    orders, products, events, reviews, order_items = load_data()

    # Phase 1: RFM & Sentiment
    print("Executing Phase 1: RFM & Sentiment Analysis...")
    rfm = p1.perform_rfm_analysis(orders)
    reviews_processed, user_sentiment = p1.perform_sentiment_analysis(reviews)
    customer_df = p1.define_segments(rfm, user_sentiment)
    edm_suggestions = p1.get_sleepy_customer_preferences(
        customer_df, order_items, products
    )

    # Phase 2: Behavior & Strategy
    print("Executing Phase 2: Behavioral Features & Automation Strategy...")
    event_features = p2.aggregate_events(events)
    customer_df = customer_df.merge(event_features, on="user_id", how="left").fillna(0)
    product_health_report = p2.extract_pain_points(reviews_processed, products)
    customer_df = p2.apply_automation_logic(customer_df)

    # Phase 3: LTV Modeling & Performance Monitoring
    print("Executing Phase 3: LTV Prediction & Performance Monitoring...")
    # build_predictive_model now internally calls check_and_retrain()
    c360_table, model_metadata, shap_explanations = p3.build_predictive_model(customer_df, orders)

    # Phase 4: Actionable Lists & Advanced Analysis
    print("Executing Phase 4: Generating Marketing Lists & Advanced Analysis...")
    lookalike_seed = p2.generate_lookalike_seed(c360_table)
    churn_risk_list = p2.generate_churn_risk_list(c360_table)
    # UPDATED: Added 'events' as the third argument
    cart_abandonment_list = p2.generate_cart_abandonment_list(c360_table, orders, events)
    sleepy_product_health = p2.generate_sleepy_product_health_report(
        c360_table, reviews_processed, products, order_items
    )
    
    # Requirement 9: Brand Health Monitor
    brand_health_monitor = p2.generate_brand_health_monitor(reviews_processed, products)
    
    cohort_retention, cohort_revenue = p4.perform_cohort_analysis(orders)
    product_affinity = p4.perform_affinity_analysis(order_items, products)

    # --- Final Outputs ---
    print("\n" + "=" * 50)
    print("BUSINESS IMPACT DATAFRAMES & METADATA")
    print("=" * 50)

    # 1. C360_Table
    print("\n1. Exporting C360_Table...")
    c360_table.to_csv("schema/C360_Table.csv", index=False)

    # 2. EDM & Product Insights
    print("2. Exporting EDM & Product Insights...")
    edm_suggestions.to_csv("schema/EDM_Suggestions.csv", index=False)
    product_health_report.to_csv("schema/Product_Health_Report.csv", index=False)
    sleepy_product_health.to_csv("schema/Sleepy_Product_Health_Report.csv", index=False)
    brand_health_monitor.to_csv("schema/Brand_Health_Monitor.csv", index=False)

    # 3. Model Insights (Note: JSON is handled by check_and_retrain inside Phase 3)
    print("3. Exporting Model Insights (SHAP)...")
    shap_explanations.to_csv("schema/Model_Explanations_Table.csv", index=False)

    # 4. Marketing Action Lists
    print("4. Exporting Actionable Marketing Lists...")
    lookalike_seed.to_csv("schema/Lookalike_Seed_List.csv", index=False)
    churn_risk_list.to_csv("schema/Churn_Risk_List.csv", index=False)
    cart_abandonment_list.to_csv("schema/Cart_Abandonment_List.csv", index=False)

    # 5. Advanced Analysis
    print("5. Exporting Advanced Analysis (Cohort & Affinity)...")
    cohort_retention.to_csv("schema/Cohort_Retention.csv")
    cohort_revenue.to_csv("schema/Cohort_Revenue.csv")
    product_affinity.to_csv("schema/Product_Affinity.csv", index=False)

    print("\nWorkflow completed successfully. All artifacts saved in schema/ directory.")


if __name__ == "__main__":
    main()
