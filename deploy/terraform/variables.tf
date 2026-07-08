# deploy/terraform/variables.tf

variable "project_id" {
  type        = string
  description = "The GCP Project ID"
}

variable "region" {
  type        = string
  default     = "asia-east1" # 預設使用台灣機房，延遲最低
  description = "The region to deploy resources"
}

variable "machine_type" {
  type        = string
  default     = "e2-standard-2" # 2 vCPU, 8GB RAM，適合執行 Kafka/Airflow 基礎集群
  description = "The machine type for GKE nodes"
}

variable "node_count" {
  type        = number
  default     = 3 # 預設啟動 3 個節點做高可用備份
  description = "Initial number of GKE nodes"
}

variable "gke_service_account" {
  type        = string
  description = "The service account for GKE nodes to impersonate"
}
