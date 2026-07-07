import os
import sys
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
)

# Load environment variables
load_dotenv()

SERVICE_ACCOUNT_JSON = os.getenv("SERVICE_ACCOUNT_JSON")
PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID = os.getenv("DATASET_ID")

if not all([SERVICE_ACCOUNT_JSON, PROJECT_ID, DATASET_ID]):
    print(
        "❌ Error: Missing configuration in .env file (SERVICE_ACCOUNT_JSON, PROJECT_ID, DATASET_ID)."
    )
    sys.exit(1)

# Convert service account path to absolute path
SERVICE_ACCOUNT_ABS_PATH = os.path.abspath(SERVICE_ACCOUNT_JSON)
if not os.path.exists(SERVICE_ACCOUNT_ABS_PATH):
    print(f"❌ Error: Service Account JSON not found at {SERVICE_ACCOUNT_ABS_PATH}")
    sys.exit(1)

# Set Google Application Credentials env variable for BigQuery Auth
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_ABS_PATH

# Checkpoint Directory
CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spark_checkpoints"
)
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# Define schemas for each topic
users_schema = StructType(
    [
        StructField("user_id", StringType(), True),
        StructField("name", StringType(), True),
        StructField("email", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("city", StringType(), True),
        StructField("signup_date", StringType(), True),
    ]
)

events_schema = StructType(
    [
        StructField("event_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("product_id", StringType(), True),
        StructField("event_type", StringType(), True),
        StructField("event_timestamp", StringType(), True),
    ]
)

orders_schema = StructType(
    [
        StructField("order_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("order_date", StringType(), True),
        StructField("order_status", StringType(), True),
        StructField("total_amount", DoubleType(), True),
    ]
)

order_items_schema = StructType(
    [
        StructField("order_item_id", StringType(), True),
        StructField("order_id", StringType(), True),
        StructField("product_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("item_price", DoubleType(), True),
        StructField("item_total", DoubleType(), True),
    ]
)

reviews_schema = StructType(
    [
        StructField("review_id", StringType(), True),
        StructField("order_id", StringType(), True),
        StructField("product_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("rating", IntegerType(), True),
        StructField("review_text", StringType(), True),
        StructField("review_date", StringType(), True),
    ]
)


def main():
    print("🚀 Initializing Spark Session...")

    # We specify the Spark Kafka & BigQuery Connector dependencies.
    # PySpark will download these JARs automatically.
    spark = (
        SparkSession.builder.appName("EcommerceBigQueryStreaming")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,com.google.cloud.spark:spark-3.5-bigquery:0.44.2",
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    print("✅ Spark Session initialized successfully.")

    # Read from Kafka multi-topic stream
    print("🔌 Connecting to Kafka broker...")
    kafka_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", "localhost:9092")
        .option("subscribe", "db_users,db_events,db_orders,db_order_items,db_reviews")
        .option("startingOffsets", "latest")
        .load()
    )

    # Cast value to string and select topic
    records_str = kafka_stream.selectExpr("CAST(value AS STRING) as json_str", "topic")

    # Define mapping of topics to schema and destination BQ table names
    topics_config = {
        "db_users": {
            "schema": users_schema,
            "table": "raw_users",
            "checkpoint": "users",
        },
        "db_events": {
            "schema": events_schema,
            "table": "raw_events",
            "checkpoint": "events",
        },
        "db_orders": {
            "schema": orders_schema,
            "table": "raw_orders",
            "checkpoint": "orders",
        },
        "db_order_items": {
            "schema": order_items_schema,
            "table": "raw_order_items",
            "checkpoint": "order_items",
        },
        "db_reviews": {
            "schema": reviews_schema,
            "table": "raw_reviews",
            "checkpoint": "reviews",
        },
    }

    queries = []

    def make_writer(table_name):
        def write_to_bq(batch_df, batch_id):
            batch_df.write.format("bigquery").option("table", table_name).option(
                "writeMethod", "direct"
            ).mode("append").save()

        return write_to_bq

    # Loop to launch structured streaming query for each topic
    for topic_name, cfg in topics_config.items():
        dest_table = f"{PROJECT_ID}.{DATASET_ID}.{cfg['table']}"
        checkpoint_path = os.path.join(CHECKPOINT_DIR, cfg["checkpoint"])

        print(f"📡 Setting up stream: Kafka '{topic_name}' ➔ BigQuery '{dest_table}'")

        # Filter by topic, parse JSON value, select fields
        parsed_df = (
            records_str.filter(col("topic") == topic_name)
            .select(from_json(col("json_str"), cfg["schema"]).alias("data"))
            .select("data.*")
        )

        # Start the query and write to BigQuery using foreachBatch
        query = (
            parsed_df.writeStream.foreachBatch(make_writer(dest_table))
            .option("checkpointLocation", checkpoint_path)
            .start()
        )

        queries.append(query)

    print(
        f"🎉 Started {len(queries)} streaming queries. Writing directly to BigQuery raw tables..."
    )
    print("Press Ctrl+C in your terminal to stop.")

    # Wait for termination of all queries
    try:
        for query in queries:
            query.awaitTermination()
    except KeyboardInterrupt:
        print("\n👋 Spark Streaming job stopped by user.")
        for query in queries:
            query.stop()


if __name__ == "__main__":
    main()
