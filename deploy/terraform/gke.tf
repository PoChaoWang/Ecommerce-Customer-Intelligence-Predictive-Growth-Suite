# deploy/terraform/gke.tf

# 宣告 GCP Provider
provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. 建立 VPC 網路，提供 GKE 集群專屬的安全隔離網路
resource "google_compute_network" "gke_vpc" {
  name                    = "ecommerce-gke-vpc"
  auto_create_subnetworks = false
}

# 2. 建立 Subnet，並啟用 GKE 採用的 VPC 原生 IP 分配模式 (VPC-Native)
resource "google_compute_subnetwork" "gke_subnet" {
  name          = "ecommerce-gke-subnet"
  ip_cidr_range = "10.0.0.0/16"
  region        = var.region
  network       = google_compute_network.gke_vpc.id

  # GKE Pod 的 IP 範圍
  secondary_ip_range {
    range_name    = "gke-pods"
    ip_cidr_range = "10.1.0.0/16"
  }

  # GKE Service 的 IP 範圍
  secondary_ip_range {
    range_name    = "gke-services"
    ip_cidr_range = "10.2.0.0/20"
  }
}

# 3. 宣告 GKE 集群主控面 (Cluster Master)
resource "google_container_cluster" "primary" {
  name     = "ecommerce-production-cluster"
  location = var.region

  # 移除預設的節點池，我們會在下方自己宣告獨立的節點池 (Node Pool)
  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.gke_vpc.name
  subnetwork = google_compute_subnetwork.gke_subnet.name

  # 啟用 VPC-Native 網路模式，讓網路效率與安全度最大化
  ip_allocation_policy {
    cluster_secondary_range_name  = "gke-pods"
    services_secondary_range_name = "gke-services"
  }

  # 停用基本驗證與客戶端證書，使用安全的 IAM 登入
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }
}

# 4. 宣告通用管理節點池 (Airflow, dbt, Dashboard)
resource "google_container_node_pool" "general_nodes" {
  name       = "ecommerce-general-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  node_count = 2

  node_config {
    preemptible  = false # 管理伺服器保持穩定，不使用搶占式
    machine_type = "e2-standard-4" # 4 vCPU, 16GB RAM，執行 Airflow 基礎組件

    service_account = var.gke_service_account
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = "production"
      workload    = "general"
    }

    tags = ["gke-node", "ecommerce-general-workload"]
  }
}

# 5. 宣告 Kafka 專用節點池 (大記憶體、持久化 SSD，固定 3 節點作高可用)
resource "google_container_node_pool" "kafka_nodes" {
  name       = "ecommerce-kafka-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  node_count = 3 # Kafka Cluster 需要最少 3 台 Broker 節點作副本冗餘

  node_config {
    preemptible  = false # 儲存節點不能突然被中斷
    machine_type = "n2-highmem-4" # 4 vCPU, 32GB RAM，提供大量作業系統 Page Cache 快取

    # 高速 SSD 持久化儲存
    disk_type    = "pd-ssd"
    disk_size_gb = 100

    service_account = var.gke_service_account
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = "production"
      workload    = "kafka"
    }

    tags = ["gke-node", "kafka-broker"]
  }
}

# 6. 宣告 Spark 專用節點池 (運算優化、自動縮放、搶占式 Spot 以節省 80% 成本)
resource "google_container_node_pool" "spark_nodes" {
  name     = "ecommerce-spark-pool"
  location = var.region
  cluster  = google_container_cluster.primary.name

  # 動態自動橫向擴充，夜間尖峰流量大時自動加入節點，離峰時自動釋放
  autoscaling {
    min_node_count = 1
    max_node_count = 10
  }

  node_config {
    spot         = true  # 關鍵：開啟 Spot VM 搶占式虛擬機，大幅省去 70%~80% 雲端成本
    machine_type = "c2-standard-8" # 8 vCPU, 32GB RAM，高運算密集型 CPU

    service_account = var.gke_service_account
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = "production"
      workload    = "spark"
    }

    tags = ["gke-node", "spark-executor"]
  }
}
