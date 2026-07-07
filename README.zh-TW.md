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
                           ▼  [dbt]
       Staging (基礎層)  →  Intermediate (中間層)  →  Marts (應用層)
                           │
                           ▼
       BigQuery ML / Python  (LTV 價值預測、NLP 情緒分析)
                           │
                           ▼
       LightDash 儀表板  +  商業建議清單 (Business Recommendation List)
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