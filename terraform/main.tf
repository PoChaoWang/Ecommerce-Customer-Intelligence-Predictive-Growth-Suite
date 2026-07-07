# main.tf

provider "google" {
  project = var.project_id
  region  = var.region
}

# Define the list of dbt-managed datasets that require dataEditor access
locals {
  dbt_datasets = toset([
    "dbt_ecommerce_staging",
    "dbt_ecommerce_intermediate",
    "dbt_ecommerce_marts"
  ])
}

# ==========================================
# 1. Raw Dataset (Read-Only Source)
# ==========================================

resource "google_bigquery_dataset" "raw_ecommerce" {
  dataset_id = "raw_ecommerce"
  location   = var.region
}

resource "google_bigquery_dataset_iam_binding" "raw_viewer" {
  dataset_id = google_bigquery_dataset.raw_ecommerce.dataset_id
  role       = "roles/bigquery.dataViewer"

  members = [
    "serviceAccount:ecommerce-dataset@${var.project_id}.iam.gserviceaccount.com"
  ]
}

# ==========================================
# 2. DBT Datasets (Staging, Intermediate, Marts)
# ==========================================

# Dynamically declare all dbt datasets using for_each loop
resource "google_bigquery_dataset" "dbt_datasets" {
  for_each   = local.dbt_datasets
  dataset_id = each.key
  location   = var.region
}

# Dynamically grant dataEditor access to the dbt Service Account for each dataset
resource "google_bigquery_dataset_iam_binding" "dbt_datasets_editor" {
  for_each   = local.dbt_datasets
  dataset_id = google_bigquery_dataset.dbt_datasets[each.key].dataset_id
  role       = "roles/bigquery.dataEditor"

  members = [
    "serviceAccount:ecommerce-dataset@${var.project_id}.iam.gserviceaccount.com"
  ]
}

# ==========================================
# 3. Reference Examples
# ==========================================

# Reference Example (Commented by default): Uncomment to grant read-only access to marketing or analyst groups in Marts layer
# resource "google_bigquery_dataset_iam_binding" "analytics_viewer" {
#   dataset_id = google_bigquery_dataset.dbt_datasets["dbt_ecommerce_marts"].dataset_id
#   role       = "roles/bigquery.dataViewer"
# 
#   members = [
#     "group:marketing-team@yourcompany.com",
#     "group:data-analyst-team@yourcompany.com"
#   ]
# }