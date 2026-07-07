# 電商商業決策系統 (E-Commerce Business Decision System)

[English](README.md) | [繁體中文](README.zh-TW.md)

這是一個整合了**數據工程**、**數據分析**與 **AI** 的全棧數據專案，旨在將原始電商數據轉化為可執行的商業建議。本系統使用 **BigQuery**、**dbt**、**BigQuery ML** 以及 **LightDash (相容 Looker)** 建構。

> 我們的目標不只是產出報表，而是直接告訴 CRM、行銷與產品團隊：*「今天該聯繫誰，以及該提供他們什麼優惠。」*

---

## 技術棧 (Tech Stack)

| 層級 | 工具 |
|---|---|
| 數據倉庫 (Data Warehouse) | Google BigQuery |
| 數據轉換 (Data Transformation) | dbt (SQL), Python |
| 數據管道 / 串流 (Streaming Pipeline) | Apache Kafka, Apache Spark (Structured Streaming) |
| 機器學習 (Machine Learning) | BigQuery ML, Scikit-learn, Pandas |
| AI 代理人 (AI Agent) | LLM (自動化 CRM 報告產出) - **[製作中]** |
| BI / 視覺化 | LightDash (相容 Looker) |

---

## 架構概覽 (Architecture Overview)

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

## dbt 分層設計 (dbt Layer Design)

本專案遵循嚴格的 dbt 三層架構，每一層都有明確的職責：

- **Staging (`stg_`)** — 對原始資料進行輕量清洗與標準化（例如 `stg_users`, `stg_orders`）。此層不包含業務邏輯，僅負責統一命名規範與資料型別。

- **Intermediate (`int_`)** — 業務邏輯的核心所在。此層負責計算 RFM 分數 (`int_rfm_scores`)、留存同期群特徵 (`int_cohort_base`) 以及 LTV 訓練特徵 (`int_ltv_training_features`)。將邏輯獨立於此層可確保模型的複用性與可測試性。

- **Marts (`mart_`)** — 直接面向商業應用的資料表：例如 `mart_c360_table` (客戶全視角畫像)、`mart_cart_abandonment_list` (購物車遺棄名單) 以及 `mart_business_recommendations` (商業建議)。這是 CRM 與行銷團隊真正查詢的對象。

這種分層設計確保了原始數據的變動不會對最終報表造成不可預測的連鎖反應，且每個轉換步驟都是可獨立審計的。

---

## 機器學習應用 (Machine Learning Applications)

**LTV 價值預測 (BigQuery ML)**
利用客戶的 RFM 特徵與歷史購買行為，透過回歸模型預測每位客戶未來 90 天的預期營收。預測結果直接驅動預算分配邏輯：高 LTV 客戶將獲得高成本的留存服務；低 LTV 客戶則導入低成本的自動化溝通流程。

**NLP 情緒分析 (Python / VADER)**
透過 `sentiment_analysis_to_bq.ipynb` 處理客戶評論，萃取情緒標籤與產品痛點。系統會將帶有負面情緒的高價值客戶標記為客服優先處理個案，在客戶真正流失前捕捉危險訊號。

---

## 主要產出 (Key Deliverables)

| 產出項目 | 解決的商業問題 |
|---|---|
| **C360 Table** (RFM 分群) | 誰是我的 VVIP 客戶？誰又是正在流失的沉睡客？ |
| **同期群分析 (Cohort Analysis)** | 上一季獲取的新客，長期來看真的是高品質客戶嗎？ |
| **LTV 價值預測** | 哪些客戶值得在未來 90 天內重點投資？ |
| **商業建議清單** | CRM 團隊「今天」該聯繫誰？該給他們什麼優惠？ |
| **CRM Agent 報告** | 為團隊主管自動產出的每日繁體中文摘要報告 |

`mart_business_recommendations` 資料表會計算出一個「優先級分數」(`priority_score` = LTV × Churn Risk)，自動篩選出每日最需要關注的前 100 名帳戶，消除 CRM 團隊在分配工作時的疑慮。

---

## 專案結構 (Project Structure)

```
/data               # 原始 CSV 數據集
/ecommerce_dbt      # 所有的 dbt 模型 (staging / intermediate / marts)
/scripts            # Python 數據導入與預處理腳本
/ipynb              # LTV 建模與情緒分析的 Notebooks
/agents             # CRM AI 代理人邏輯
/memo               # 架構設計文件與規劃筆記
/dashboard_preview  # 儀表板版面草案 (高層概覽、分群分析)
```

---

## 開發階段 (Development Phases)

專案按成熟度分為五個階段執行：

1. **階段 1** — RFM 分群 + 情緒標籤 → C360 Table
2. **階段 2** — 購物車遺棄 + 流失風險行動名單
3. **階段 3** — LTV 預測模型 + 預算分配邏輯
4. **階段 4** — 留存同期群分析 + 交叉銷售關聯規則
5. **階段 5** — 決策引擎整合所有訊號 → 最終商業建議名單

---

## 資料隱私與安全防護 (Data Privacy & Security)

本專案在設計數據倉庫時，遵循 **最小權限原則 (Principle of Least Privilege)** 與 **隱私合規規範 (Privacy Compliance)**，特別針對敏感的用戶資料（PII）以及各資料分層的權限存取進行嚴格管控。

### 1. 使用者敏感資訊處理 (PII Data Protection)
* **移除原始 Email**：為避免敏感的 PII（個人識別資訊）在數據分析與建模過程中外洩，我們已在基礎 Staging 模型 [stg_users.sql](ecommerce_dbt/models/staging/stg_users.sql) 的查詢中，將明文 `email` 欄位**完全移除**。
* **獨立雜湊處理 (Secure Hashing)**：行銷或推薦平台（如 Facebook Ads、Google Ads）在進行廣告投放或 Lookalike 分眾比對時，通常需要 Email 作為比對種子。為此，我們建立了獨立的 [stg_users_hashed.sql](ecommerce_dbt/models/staging/stg_users_hashed.sql) 模型：
  - 使用 `lower(trim(email))` 對 Email 進行標準化處理。
  - 使用安全強度極高的 **SHA-256** 演算法進行雜湊，並轉換為 16 進位字串（`to_hex(sha256(...))`）保存。
  - 此雜湊過程不可逆，既能滿足行銷匹配需求，又確保了使用者隱私安全。
* **最終行銷輸出**：行銷團隊專屬的 [mart_lookalike_hashed_email.sql](ecommerce_dbt/models/marts/marketing/mart_lookalike_hashed_email.sql) 模型僅合併 `hashed_email` 與預測標籤，供行銷團隊安全匯出投放。

### 2. 資料集層級權限控制 (GCP BigQuery IAM)
我們使用 **Terraform** 對 BigQuery 內的各資料集（Datasets）進行精細化權限控管，防範資料誤用與越權存取：

* **Raw 資料集 (`raw_ecommerce`)**：
  * **權限級別**：`roles/bigquery.dataViewer` (唯讀)
  * **存取對象**：僅開放給 dbt 執行用的 Service Account 與數據工程人員。防止原始數據被意外修改或刪除。
* **中間/半成品資料集 (`dbt_ecommerce_staging`, `dbt_ecommerce_intermediate`)**：
  * **權限級別**：`roles/bigquery.dataEditor` (編輯/寫入)
  * **存取對象**：**僅限 dbt 執行用的 Service Account**，對一般分析師與行銷團隊**完全鎖定**。
  * **防護目的**：Staging 與 Intermediate 包含大量的 View（視圖）與半成品，鎖定這兩層能防範使用者串接錯誤的中間資料、防止 BigQuery 重複執行視圖運算導致的高昂成本，並確保敏感欄位不被旁路讀取。
* **業務產出層 (`dbt_ecommerce_marts`)**：
  * **權限級別**：dbt Service Account 擁有 `dataEditor` 進行寫入更新；數據分析師與行銷團隊僅授予 `dataViewer` (唯讀權限)。
  * **防護目的**：使用者僅能存取已清洗、聚合且經過隱私脫敏的 Marts 表，確保決策報表的高效與一致性。

---

## 本地即時串流環境部署與執行指引 (Local Streaming Setup & Execution Guide)

本專案支援完整的實時數據串流管道模擬，以下是本地環境部署與執行的完整步驟：

### 1. 安裝本地 Python 依賴與 PySpark 運行環境
本專案的 Spark 串流消費者需要 **Java 17 或 Java 11** 的支持，請先確保本地已安裝 Java。
接著，在虛擬環境中安裝 Python 套件：
```bash
# 激活您的虛擬環境後執行
pip install -r requirements.txt
```

### 2. 啟動 Docker 容器 (Kafka & Kafka UI)
確保本地 Docker Desktop 已啟動，並在專案根目錄下運行：
```bash
# 啟動 Zookeeper, Kafka Broker 與 Kafka UI 監控網頁
docker compose up -d
```
*   **Kafka UI 網頁**: 可開啟瀏覽器訪問 **[http://localhost:8085](http://localhost:8085)** 觀察即時 Topic 與 Message。

### 3. 資料初始化與歷史補齊 (選填)
若要清空測試數據，並自動補齊從原 Kaggle 最後更新日到昨天為止的歷史空缺：
```bash
# 一鍵還原/重置本地 CSV 至 Kaggle 原始狀態
python scripts/run_gap_filler.py --action reset

# 增量自動補齊歷史斷層（至昨天為止）
python scripts/run_gap_filler.py --action gap-fill
```

### 4. 執行即時串流管道
請開啟兩個終端機視窗，分別啟動生產者與消費者：

*   **終端機 1 - 啟動 Kafka 實時模擬生產者**：
    以每秒 1 筆的速度，持續將符合電商關聯性規則的即時行為/訂單發送至 Kafka：
    ```bash
    python scripts/run_kafka_producer.py --delay 1.0
    ```
*   **終端機 2 - 啟動 Spark Structured Streaming 消費者**：
    Spark 將即時訂閱 Kafka 的多個主題，並使用 BigQuery Storage Write API (Direct Mode) 強制 Commit 寫入 BigQuery raw 資料集：
    ```bash
    python scripts/spark_bigquery_consumer.py
    ```

*(欲中止發送或消費，直接在各自的終端機按下 `Ctrl + C` 即可。)*

---

*本專案作為個人作品集，展示了在商業環境中端到端數據工程與機器學習的應用能力。*