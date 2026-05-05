import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor  # Temporary fallback due to libomp missing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import shap
import warnings
import json
import os

import phase1_rfm_sentiment as p1

# Suppress warnings
warnings.filterwarnings("ignore")


def check_and_retrain(current_metadata):
    """
    Requirement 8: Monitor model performance and decide if retraining warning is needed.
    Appends current run metadata to schema/Model_Run_Metadata.json and compares
    current RMSE with the last recorded run RMSE.
    """
    metadata_path = "schema/Model_Run_Metadata.json"
    metadata_history = {"runs": []}
    retrain_triggered = False

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                existing_metadata = json.load(f)

            if isinstance(existing_metadata, dict) and isinstance(
                existing_metadata.get("runs"), list
            ):
                metadata_history = existing_metadata
            elif isinstance(existing_metadata, dict):
                metadata_history = {"runs": [existing_metadata]}

            previous_run = metadata_history["runs"][-1] if metadata_history["runs"] else {}
            old_rmse = previous_run.get("performance_metrics", {}).get("rmse")
            new_rmse = current_metadata["performance_metrics"]["rmse"]

            if old_rmse and new_rmse > old_rmse * 1.15:
                print(
                    f"⚠️ WARNING: Model performance degraded! RMSE increased "
                    f"from {old_rmse} to {new_rmse} (>15%)."
                )
                retrain_triggered = True
        except Exception as e:
            print(f"Note: Could not parse old metadata for comparison ({e}).")

    now = datetime.now()
    current_metadata["run_id"] = now.strftime("%Y%m%d_%H%M%S")
    current_metadata["run_timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")
    current_metadata["retrain_triggered"] = retrain_triggered
    metadata_history["runs"].append(current_metadata)

    os.makedirs("schema", exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata_history, f, indent=4)

    return current_metadata


def build_model_validation_table(c360_table):
    """
    Build a dashboard-ready model validation table for the 90-day LTV model.

    The validation target keeps customers with no prediction-period purchase by
    filling missing actual profit with 0. This avoids validating only on buyers.
    """
    required_columns = ["user_id", "segment", "predicted_profit_90_days", "actual_profit"]
    missing_columns = [col for col in required_columns if col not in c360_table.columns]
    if missing_columns:
        raise ValueError(
            "c360_table is missing required columns for model validation: "
            + ", ".join(missing_columns)
        )

    validation_table = c360_table[required_columns].copy()
    validation_table["predicted_profit_90_days"] = pd.to_numeric(
        validation_table["predicted_profit_90_days"], errors="coerce"
    ).fillna(0)
    validation_table["actual_profit"] = pd.to_numeric(
        validation_table["actual_profit"], errors="coerce"
    )
    validation_table["actual_profit_filled"] = validation_table["actual_profit"].fillna(0)

    validation_table["prediction_error"] = (
        validation_table["predicted_profit_90_days"]
        - validation_table["actual_profit_filled"]
    )
    validation_table["absolute_error"] = validation_table["prediction_error"].abs()
    validation_table["squared_error"] = validation_table["prediction_error"] ** 2

    n_rows = len(validation_table)
    if n_rows == 0:
        validation_table["prediction_decile"] = pd.Series(dtype="Int64")
    else:
        prediction_rank = validation_table["predicted_profit_90_days"].rank(
            method="first", ascending=False
        )
        validation_table["prediction_decile"] = (
            np.floor((prediction_rank - 1) * 10 / n_rows) + 1
        ).clip(1, 10).astype(int)

    decile_labels = {
        1: "Top 10%",
        2: "10-20%",
        3: "20-30%",
        4: "30-40%",
        5: "40-50%",
        6: "50-60%",
        7: "60-70%",
        8: "70-80%",
        9: "80-90%",
        10: "Bottom 10%",
    }
    validation_table["prediction_decile_label"] = validation_table[
        "prediction_decile"
    ].map(decile_labels)
    validation_table["top_10_flag"] = validation_table["prediction_decile"] <= 1
    validation_table["top_20_flag"] = validation_table["prediction_decile"] <= 2
    validation_table["top_30_flag"] = validation_table["prediction_decile"] <= 3

    output_columns = [
        "user_id",
        "segment",
        "predicted_profit_90_days",
        "actual_profit",
        "actual_profit_filled",
        "prediction_error",
        "absolute_error",
        "squared_error",
        "prediction_decile",
        "prediction_decile_label",
        "top_10_flag",
        "top_20_flag",
        "top_30_flag",
    ]
    return validation_table[output_columns]


def build_predictive_model(df, orders):
    """
    Phase 3: Train LTV model using a 90-day time-window split.
    """
    print("Implementing time-window split (90 days)...")
    
    # 1. Define time windows
    latest_date = orders["order_date"].max()
    split_date = latest_date - pd.Timedelta(days=90)
    
    observation_orders = orders[orders["order_date"] < split_date]
    prediction_orders = orders[orders["order_date"] >= split_date]
    
    # 2. Prepare Target (Y) from Prediction Period
    pred_profit = (
        prediction_orders.groupby("user_id")["total_amount"]
        .sum()
        .reset_index()
    )
    pred_profit["actual_profit"] = pred_profit["total_amount"] * 0.25 - 20
    pred_profit = pred_profit[["user_id", "actual_profit"]]
    
    # 3. Prepare Features (X) from Observation Period
    print("Recalculating RFM for observation period...")
    obs_rfm = p1.perform_rfm_analysis(observation_orders)
    
    # 4. Merge everything back to the main dataframe
    cols_to_drop = ["recency", "frequency", "monetary", "recency_score", "frequency_score", "monetary_score"]
    base_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    merged_df = base_df.merge(obs_rfm, on="user_id", how="left")
    merged_df = merged_df.merge(pred_profit, on="user_id", how="left")
    
    # 5. Features and Training Split
    features = ["recency", "frequency", "total_views", "total_cart_adds", "compound_score"]
    train_pool = merged_df.dropna(subset=["actual_profit"] + features)
    
    X_train_full = train_pool[features]
    y_train_full = train_pool["actual_profit"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    print(f"Training RandomForestRegressor on {len(X_train)} samples...")
    model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 6. Inference
    X_inference = merged_df[features].fillna(0)
    merged_df["predicted_profit_90_days"] = model.predict(X_inference)

    # Scaling
    scaler_minmax = MinMaxScaler()
    merged_df["prob_minmax"] = scaler_minmax.fit_transform(merged_df[["predicted_profit_90_days"]])
    z_scores = StandardScaler().fit_transform(merged_df[["predicted_profit_90_days"]])
    merged_df["prob_sigmoid"] = 1 / (1 + np.exp(-z_scores))

    # Metrics
    y_test_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    mae = float(mean_absolute_error(y_test, y_test_pred))

    # Model Metadata
    model_metadata = {
        "model_type": "RandomForestRegressor",
        "performance_metrics": {
            "rmse": round(rmse, 2),
            "mae": round(mae, 2)
        },
        "feature_importance": {
            feat: round(float(imp), 4) for feat, imp in zip(features, model.feature_importances_)
        }
    }

    # SHAP Values
    print(f"Calculating SHAP values for {len(X_inference)} rows...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_inference, check_additivity=False)
    if isinstance(shap_values, list) and len(shap_values) > 0:
        shap_values = shap_values[0]
    shap_explanations = pd.DataFrame(shap_values, columns=features)
    shap_explanations.insert(0, "user_id", merged_df["user_id"].values)

    # Mapping drivers
    feature_map = {
        "recency": "Time Since Last Order",
        "frequency": "Purchase Frequency",
        "total_views": "Browsing Engagement",
        "total_cart_adds": "Cart Add Intensity",
        "compound_score": "Review Sentiment",
    }
    tmp_shap = shap_explanations.drop(columns=["user_id"])
    merged_df["primary_driver"] = tmp_shap.idxmax(axis=1).map(feature_map).values
    merged_df["primary_barrier"] = tmp_shap.idxmin(axis=1).map(feature_map).values

    # Requirement 8: Check and Retrain logic
    model_metadata = check_and_retrain(model_metadata)

    return merged_df, model_metadata, shap_explanations
