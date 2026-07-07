import os
import pandas as pd
import pandas_gbq
from google.oauth2 import service_account
from google.cloud import bigquery
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ==================== Configuration ====================
# Load sensitive data from .env
SERVICE_ACCOUNT_JSON = os.getenv("SERVICE_ACCOUNT_JSON")
PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID = os.getenv("DATASET_ID")

if not all([SERVICE_ACCOUNT_JSON, PROJECT_ID, DATASET_ID]):
    raise RuntimeError(
        "Error: Please ensure SERVICE_ACCOUNT_JSON, PROJECT_ID, and DATASET_ID are set in the .env file"
    )

# Mapping of source files to BigQuery table names
DATA_FILES = {
    "data/events.csv": "raw_events",
    "data/order_items.csv": "raw_order_items",
    "data/orders.csv": "raw_orders",
    "data/products.csv": "raw_products",
    "data/reviews.csv": "raw_reviews",
    "data/users.csv": "raw_users",
}
# ===============================================


def main():
    # 1. Verify if the credentials file exists
    if not os.path.exists(SERVICE_ACCOUNT_JSON):
        raise FileNotFoundError(
            f"Service Account key not found, please check path: {SERVICE_ACCOUNT_JSON}"
        )

    # 2. Load credentials
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_JSON
    )

    # Initialize BigQuery Client to check/create dataset
    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

    print(
        f"🚀 Starting data ingestion to GCP Project: {PROJECT_ID} (Dataset: {DATASET_ID})"
    )

    # 3. Ensure Dataset exists
    dataset_ref = bigquery.DatasetReference(PROJECT_ID, DATASET_ID)
    try:
        client.get_dataset(dataset_ref)
        print(f"✅ Dataset {DATASET_ID} already exists.")
    except Exception:
        print(f"ℹ️ Dataset {DATASET_ID} not found. Attempting to create it...")
        try:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"  # Adjust location if needed
            client.create_dataset(dataset)
            print(f"✅ Dataset {DATASET_ID} created successfully.")
        except Exception as e:
            print(f"❌ Failed to create or access dataset: {e}")
            print(
                "Please ensure your Service Account has 'BigQuery Data Editor' and 'BigQuery User' roles."
            )
            return

    # 4. Loop through files and upload
    for file_name, table_name in DATA_FILES.items():
        if os.path.exists(file_name):
            print("--------------------------------------------------")
            print(f"📦 Detected file {file_name}, reading...")

            # Read CSV
            df = pd.read_csv(file_name, low_memory=False)

            # ELT best practice: Ingest raw data as-is
            destination = f"{DATASET_ID}.{table_name}"
            print(f"📤 Uploading to BigQuery -> {destination} ({len(df)} rows)...")

            try:
                pandas_gbq.to_gbq(
                    df,
                    destination_table=destination,
                    project_id=PROJECT_ID,
                    credentials=credentials,
                    if_exists="replace",
                )
                print(f"✅ {table_name} uploaded successfully!")
            except Exception as e:
                print(f"❌ Error uploading {table_name}: {e}")
                if "403" in str(e):
                    print(
                        "Hint: This is likely a permission issue. Ensure the Service Account has 'BigQuery Data Editor' role on the project."
                    )
        else:
            print(f"⚠️ File {file_name} not found, skipping.")

    print("\n🎉 All data ingestion completed! You can check the BigQuery console now.")


if __name__ == "__main__":
    main()
