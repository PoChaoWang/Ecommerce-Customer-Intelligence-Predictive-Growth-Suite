variable "project_id" {
  type        = string
  description = "The GCP Project ID"
}

variable "region" {
  type        = string
  default     = "asia-east1"
  description = "The region to deploy resources"
}
