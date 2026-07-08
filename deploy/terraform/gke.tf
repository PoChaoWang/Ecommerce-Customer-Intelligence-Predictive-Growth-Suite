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

# 4. 宣告實體工作節點池 (GKE Node Pool)
resource "google_container_node_pool" "primary_nodes" {
  name       = "ecommerce-node-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  node_count = var.node_count

  # 自動縮放設定 (如果任務變多，自動擴展節點數)
  autoscaling {
    min_node_count = 1
    max_node_count = 5
  }

  node_config {
    preemptible  = false # 生產環境不建議用搶占式 VM，以確保 Kafka/Airflow 穩定度
    machine_type = var.machine_type

    # 綁定權限最低的 Service Account，遵循最小權限原則
    service_account = var.gke_service_account

    # 節點的 OAuth 權限範圍
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = "production"
    }

    tags = ["gke-node", "ecommerce-data-pipeline"]
  }
}
