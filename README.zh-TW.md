# 電商商業決策系統 (E-Commerce Business Decision System)

[English](README.md) | [繁體中文](README.zh-TW.md)

這是一個整合了**數據工程 (Data Engineering)**、**數據分析 (Data Analytics)** 與 **AI/ML** 的全棧數據專案，旨在將原始電商數據轉化為可執行的商業決策建議。

> 💡 **核心價值**：本系統不只產出靜態報表，而是主動為 CRM、行銷與產品團隊提供「每日行動方案」——直接回答：**「今天該聯繫哪些客戶？該提供什麼優惠？誰有流失風險？」**

---

## 🚀 專案亮點與工程指標 (Key Highlights)

* **商業價值驅動**：整合 RFM 模型與 BigQuery ML 預測的生命週期價值 (LTV)，自動產出客戶全視角畫像 (C360) 與高優先級 CRM 行動建議。
* **企業級 CI/CD 與 Slim CI**：利用 GitHub Actions 與 GCP Workload Identity Federation (無金鑰驗證)，搭配 Slim CI 僅測試修改的 dbt 模型，**BigQuery 節點運算成本降低 90% 以上**。
* **零查詢成本監控 (Zero-Query-Cost)**：Airflow 每日自動調度，並透過解析本地 `run_results.json` 產出執行日報與即時 Slack/Email 警報，**不對 BigQuery 發起任何額外查詢**。
* **隱私安全合規 (PII Protection)**：Staging 層移除敏感 Email，採 SHA-256 進行不可逆雜湊，並透過 Terraform 實施 BigQuery 資料集層級 (Raw/Staging/Marts) 的精細化 IAM 權限控管。
* **雲端生產級容器編排 (GKE & Terraform)**：使用 Terraform 一鍵佈署 VPC 與自動縮放的 GKE 叢集，並以 Kubernetes 容器化技術部署 Kafka 串流、Postgres PVC 與 Airflow KubernetesExecutor (整合 git-sync)。

---

## 🛠️ 技術棧 (Tech Stack)

| 層級 | 技術與工具 |
|---|---|
| **數據倉庫 (Data Warehouse)** | Google BigQuery |
| **數據轉換 (Data Transformation)** | dbt (SQL), Python |
| **數據管道 / 串流 (Streaming Pipeline)** | Apache Kafka, Apache Spark (Structured Streaming) |
| **機器學習 (Machine Learning)** | BigQuery ML (XGBoost/迴歸模型), Scikit-learn, Pandas |
| **自然語言處理 (NLP)** | VADER (情緒分析) |
| **自動化調度 (Orchestration)** | Apache Airflow |
| **基礎設施即程式碼 (IaC)** | Terraform |
| **容器編排 (Orchestration)** | Google Kubernetes Engine (GKE), Docker, Kubernetes |
| **BI 與視覺化 (BI & Visualization)** | LightDash (相容 Looker) |

---

## 📐 系統架構圖 (System Architecture)

```
[原始 CSV 歷史數據]                [即時模擬數據 (Faker + Python)]
       │                                     │
       │                                     ▼ (即時寫入)
       │                              Apache Kafka (訊息佇列)
       │                                     │
       │ (批次載入)                           ▼ (即時串流消費)
       │                              Apache Spark (Structured Streaming)
       │                                     │
       └───────────────────┬─────────────────┘
                           ▼
                 Google BigQuery (raw_ecommerce 資料集)
                           │
                           ▼
                       dbt Staging
                           │
                           ▼
                     dbt Intermediate (特徵層)
                           │
             ┌─────────────┴─────────────┐
             ▼                           │
    BigQuery ML / Python (機器學習預測)    │
             │                           │
             ▼                           ▼
     prediction tables ──────────►   dbt Marts (業務層)
                                         │ (整合 LTV 計算優先級)
                                         ▼
                             C360 / 商業建議名單 (CRM 行動方案)
                                         │
                                         ▼
                             LightDash 儀表板 + CRM Action
```

---

## 💾 數據流與建模設計 (Data & Modeling Design)

### 1. dbt 分層架構 (dbt Layer Design)
本專案遵循嚴格的 dbt 三層架構，每一層都有明確的職責：
* **Staging (`stg_`)** — 對原始資料進行輕量清洗與標準化（例如 [stg_users.sql](ecommerce_dbt/models/staging/stg_users.sql)）。此層不包含業務邏輯，僅負責統一命名規範與資料型別，並實施敏感資訊遮蔽。
* **Intermediate (`int_`)** — 業務邏輯的核心所在。此層負責計算 RFM 分數 (`int_rfm_scores`)、留存同期群特徵 (`int_cohort_base`) 以及 LTV 訓練特徵 (`int_ltv_training_features`)，將邏輯與最終展現分離以確保模型複用性。
* **Marts (`mart_`)** — 直接面向商業應用的資料表：例如 `mart_c360_table` (客戶全視角畫像) 與 `mart_business_recommendations` (商業建議)。這是 CRM 與行銷團隊真正查詢的對象。

### 2. 機器學習與 AI 應用
* **LTV 價值預測 (BigQuery ML)**：利用客戶的 RFM 特徵與歷史購買行為，透過回歸模型預測每位客戶未來 90 天的預期營收。預測結果直接驅動預算分配邏輯：高 LTV 客戶將獲得高成本的留存服務，低 LTV 客戶則導入低成本的自動化溝通流程。
* **NLP 情緒分析 (Python / VADER)**：處理客戶評論，萃取情緒標籤與產品痛點。系統會將帶有負面情緒的高價值客戶標記為客服優先處理個案，在客戶真正流失前捕捉危險訊號。

---

## 🔒 企業級工程實踐 (Enterprise Engineering Practices)

### 1. PII 安全防護與資料集權限控制 (Data Privacy & IAM)
* **PII 去識別化**：[stg_users.sql](ecommerce_dbt/models/staging/stg_users.sql) 直接剔除明文 Email，僅在獨立的 [stg_users_hashed.sql](ecommerce_dbt/models/staging/stg_users_hashed.sql) 中，將 Email 經由標準化後進行 **SHA-256** 雜湊（`to_hex(sha256(...))`）保存，供第三方廣告投放（Lookalike）安全匹配。
* **精細化 IAM 權限控管 (GCP BigQuery)**：
  - **`raw_ecommerce` (Raw 層)**：僅限數據工程師與 dbt 唯讀 (`dataViewer`)，防止原始數據被意外修改。
  - **`dbt_ecommerce_staging` / `intermediate` (半成品層)**：僅限 dbt 執行服務帳戶 (`dataEditor`) 讀寫，對一般分析師與行銷團隊**完全鎖定**，避免串接錯誤數據或旁路讀取敏感欄位。
  - **`dbt_ecommerce_marts` (業務產出層)**：分析師與行銷團隊僅授予 `dataViewer` (唯讀)，確保決策報表的高效與一致性（Single Source of Truth）。

### 2. 現代化 CI/CD 與 Slim CI
* **Workload Identity Federation (WIF)**：整合 GitHub Actions 與 GCP，採用動態 OIDC 短期 Token 進行**無金鑰驗證 (Keyless Authentication)**，杜絕長期 Service Account JSON 金鑰外洩風險。相關基礎設施定義於 [cicd.tf](terraform/cicd.tf)。
* **dbt Slim CI (狀態比對測試)**：當 PR 提交時，CI 管道 ([dbt_ci.yml](.github/workflows/dbt_ci.yml)) 自動下載 Production 的 `manifest.json`，**僅針對有變動的 Model 及其下游**在臨時資料集（`ci_pr_<pr_num>`）中執行 `dbt build --select state:modified+`，大幅降低 BigQuery 運算成本並縮短 CI 流程。
* **靜態代碼檢查 (Linting)**：整合 `ruff` (Python) 與 `sqlfluff` (SQL) 進行代碼語法與風格排版靜態檢查。

### 3. 生產級 Airflow 自動化調度與無成本監控 (Airflow Orchestration)
* **調度策略**：每日凌晨 01:00 (台北時間) 自動執行，DAG 設計見 [ecommerce_dbt_dag.py](airflow/dags/ecommerce_dbt_dag.py)。設有 3 次自動重試與任務失敗即時警報機制（`on_failure_callback`），出錯時自動向運維團隊寄送精美 HTML 報錯信件。
* **零查詢成本監控日報 (Zero-Query-Cost)**：在 dbt 流程執行完畢後，Airflow 會直接解析本地生成的 `run_results.json` 狀態檔案進行報表編譯並發送 Email/Slack，**完全無需對 BigQuery 進行二次查詢，從而達到零成本的日常監控**。

### 4. 雲端生產部署藍圖 (Kubernetes & IaC)
* **Terraform IaC**：定義於 [gke.tf](deploy/terraform/gke.tf)，一鍵在 GCP 佈署專屬 VPC 網路與自動彈性縮放（1-5 節點）的 GKE 叢集。
* **Kubernetes 容器編排**：
  - **Airflow KubernetesExecutor**：啟用任務級 Pod 動態縮放，並使用 `git-sync` 每 60 秒自動從 GitHub 同步 DAG 代碼，無須重構 Docker 映像檔。
  - **Strimzi Kafka Operator**：一鍵部署 3 節點的 Kafka 與 Zookeeper 高可用叢集。
  - **Postgres PVC**：使用 PersistentVolumeClaim 動態申請 GCP 雲端儲存，確保資料庫持久化。

---

## 📊 主要產出與商業應用 (Key Deliverables)

| 產出項目 | 解決的商業問題 | 核心應用場景 |
|---|---|---|
| **C360 Table** | 誰是我們的 VVIP 客戶？誰又是正在流失的沉睡客？ | 提供 CRM 團隊全視角客戶畫像與 RFM 分群，進行分眾行銷。 |
| **同期群分析 (Cohort Analysis)** | 上一季獲取的新客，長期來看真的是高品質客戶嗎？ | 評估不同時間段獲客的留存曲線，優化獲客預算。 |
| **LTV 價值預測** | 哪些客戶值得在未來 90 天內重點投資？ | 將行銷預算優先分配給高 LTV 客戶；對低 LTV 客戶實施自動化留存。 |
| **商業建議清單** | CRM 團隊「今天」該聯繫誰？該給他們什麼優惠？ | 依優先級分數 (`LTV × Churn Risk`) 篩選出每日最需關注的前 100 名帳戶。 |
| **CRM Agent 報告** | 如何快速向團隊主管彙整每日業務狀況？ | 為團隊主管自動產出的每日繁體中文數據摘要報告。 |

---

## 📂 專案目錄結構 (Project Structure)

```
/data               # 原始 CSV 數據集
/ecommerce_dbt      # 所有的 dbt 模型 (staging / intermediate / marts)
/scripts            # Python 數據導入與預處理腳本
/ipynb              # LTV 建模與情緒分析的 Jupyter Notebooks
/agents             # CRM AI 代理人邏輯
/deploy             # 雲端生產環境部署藍圖 (Terraform & Kubernetes Manifests)
/memo               # 架構設計文件、規劃筆記與詳細部署指引
/dashboard_preview  # 儀表板版面草案 (高層概覽、分群分析)
```

---

## ⚡ 本地快速啟動指引 (Local Quick Start)

本專案支援兩種本地數據基礎設施啟動方式：

### 方案 A：使用 Docker Compose (預設、最輕量)

1. **啟動基礎設施** (Zookeeper, Kafka, Kafka UI):
   ```bash
   # 啟動容器服務
   docker compose up -d
   ```
   *可開啟瀏覽器訪問 [http://localhost:8085](http://localhost:8085) 監控 Kafka 即時訊息與 Topic。*
2. **數據初始化與補齊** (自動補足歷史斷層至昨天):
   ```bash
   python scripts/run_gap_filler.py --action gap-fill
   ```
3. **執行即時串流管道** (請開啟兩個終端機視窗執行):
   * **終端機 1 (模擬數據生產者)**:
     ```bash
     python scripts/run_kafka_producer.py --delay 1.0
     ```
   * **終端機 2 (PySpark 串流消費者，寫入 BigQuery)**:
     ```bash
     python scripts/spark_bigquery_consumer.py
     ```

### 方案 B：使用 Kubernetes (本地 K8s 部署)

1. **一鍵部署 Postgres 與 Kafka 叢集**:
   ```bash
   # 部署資料庫與 PVC 儲存卷
   kubectl apply -f deploy/k8s/postgres.yaml
   # 部署 1-Node KRaft Kafka 叢集與 Topics (將對外暴露 NodePort 30094 & 30095)
   kubectl apply -f deploy/k8s/kafka.yaml
   ```
2. **數據初始化與補齊**:
   ```bash
   python scripts/run_gap_filler.py --action gap-fill
   ```
3. **執行即時串流管道 (指定 K8s Kafka Port)** (請開啟兩個終端機視窗執行):
   * **終端機 1 (模擬數據生產者)**:
     ```bash
     KAFKA_BOOTSTRAP_SERVERS="localhost:30094" python scripts/run_kafka_producer.py --delay 1.0
     ```
   * **終端機 2 (PySpark 串流消費者，寫入 BigQuery)**:
     ```bash
     KAFKA_BOOTSTRAP_SERVERS="localhost:30094" python scripts/spark_bigquery_consumer.py
     ```

⚠️ **注意**：若在方案 A 與方案 B 之間切換，請務必先執行 `rm -rf spark_checkpoints` 刪除舊的 Spark Offset 快取，避免因 Kafka 叢集重置導致 Offset 不一致而報錯。

---

*本專案作為個人作品集，展示了在商業環境中端到端數據工程、運維工程 (Platform/DataOps) 與機器學習的整合實踐能力。*