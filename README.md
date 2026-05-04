# E-Commerce Business Decision System

本專案是一套以電商成長為核心的商業決策系統，目標不是單純產出分析報表或預測模型，而是把分散的顧客、交易、瀏覽、評論資料，轉換成行銷、CRM、產品團隊可以直接採取行動的優先級名單與決策建議。

它回答的是營運團隊每天真正需要解決的問題：

- 哪些顧客最值得優先挽回？
- 哪些購物車放棄顧客最有機會轉換？
- 哪些高價值顧客可以用來擴大相似客群？
- 哪些產品或品類正在造成負面體驗？
- 行銷資源應該先投在哪些人、哪些商品、哪些活動？

---

## 1. Project Overview (Business Framing)

電商公司通常不缺資料，真正的挑戰是資料無法快速轉化為決策。行銷團隊看到營收下降，可能不知道是新客品質變差、老客流失、商品體驗問題，還是轉換漏斗卡住；CRM 團隊有大量會員名單，卻不知道誰應該先被觸達；產品團隊收到負評，但不清楚哪些問題會直接影響留存與營收。

本專案建立一個從資料到行動的決策流程，協助團隊：

- 提升營收：找出高價值顧客與高轉換機會
- 降低流失：提前辨識高風險顧客並安排挽回
- 提高行銷效率：把預算集中在最可能產生回報的人群
- 改善產品體驗：從評論與留存變化發現問題商品
- 建立決策優先順序：讓團隊知道今天應該先做什麼

最終輸出不是一份靜態分析，而是一套可重複執行的 business decision loop。

---

## 2. End-to-End Decision Loop

本系統設計成一個完整的閉環：

**Data → Insight → Action → Measurement → Feedback → Model Improvement**

### Data：整合顧客與營運資料

系統整合訂單、瀏覽行為、購物車、評論、產品與顧客資料，形成單一顧客視角。這讓團隊不只看單筆交易，而是理解每位顧客的價值、活躍度、情緒、流失風險與未來潛力。

### Insight：把資料轉成商業洞察

分析會回答具體的商業問題，例如：

- 哪些顧客是 `VVIP Loyal High-Value Customer` 或 `Dormant High-Value Customer`？
- 哪些人有高流失風險？
- 哪些購物車放棄事件值得再行銷？
- 哪些商品常被一起購買？
- 哪些月份取得的新客留存較差？

### Action：把洞察轉成可執行名單

系統會輸出行動型 CSV，例如流失挽回名單、購物車放棄名單、Lookalike 種子名單與最終推薦名單。這些名單可以直接提供給 CRM、廣告投放、EDM、客服或產品團隊使用。

### Measurement：追蹤行動後的成效

每個行動都可以用商業指標追蹤，例如：

- Retention Campaign 是否降低流失率？
- Retargeting Campaign 是否提升轉換率？
- Lookalike Expansion 是否改善 ROAS？
- Customer Support Priority 是否降低負評與退款？

### Feedback：把結果回饋到決策規則

如果某個族群折扣反應佳，可以提高該族群的優先級；如果某類顧客對 EDM 反應差，則可調整觸達方式或內容。系統的價值在於持續學習行銷行動與顧客反應之間的關係。

### Model Improvement：持續改善預測與優先排序

LTV、流失風險、顧客分群與推薦規則會隨著新資料與新成效持續調整。這讓決策不只是一次性分析，而是可營運、可監控、可優化的成長系統。

---

## 3. Action Effectiveness Review: What If the Recommendation Does Not Work?

在真實商業環境中，並不是每一個 insight 或 recommended action 都會立刻帶來成效。這並不代表分析本身失敗，而是代表原本的 business hypothesis 需要被重新驗證。

因此，本系統不把 recommended action 視為決策終點，而是把它視為一個可被追蹤、驗證與優化的商業假設。

當某個行動沒有達到預期效果時，BA 應該從以下四個層面進行診斷：

### 1. Insight Validation

確認原本的 insight 是否真的抓到主要問題。

例如，系統可能判斷某群顧客是高 LTV 且高流失風險，因此建議執行 retention campaign。但如果顧客流失的主因其實是產品品質、配送體驗或售後服務問題，單純提供折扣可能無法有效提升回購。

### 2. Action Fit

確認提出的行動是否真的對應到顧客的主要阻礙。

例如：

- 如果顧客的主要問題是價格敏感，折扣或優惠券可能有效。
- 如果顧客的主要問題是負面評論或商品體驗，客服介入或產品改善可能更有效。
- 如果顧客只是購物車放棄，再行銷廣告或 reminder EDM 可能比一般 retention campaign 更適合。

因此，recommended action 需要根據顧客的 segment、predicted LTV、churn risk、sentiment signal、cart behavior 和 primary barrier 持續調整。

### 3. Execution Check

確認行動是否被正確執行。

即使 insight 和 strategy 是合理的，實際執行仍可能出現問題，例如：

- EDM 發送對象錯誤
- 優惠券設定錯誤
- 廣告受眾上傳不完整
- Campaign message 與顧客痛點不一致
- 發送時間不符合顧客購買週期
- Landing page 或商品頁體驗不佳

因此，成效不佳時不能只檢討分析模型，也必須檢查 campaign execution。

### 4. Measurement Review

確認 KPI、觀察期間與評估方式是否合理。

不同 action 應該使用不同的 success metrics：

| Recommended Action | Primary KPI | Diagnostic KPI |
|---|---|---|
| Retention Campaign | Repurchase Rate, Retention Rate | Open Rate, CTR, Coupon Usage |
| Retargeting Campaign | Conversion Rate, ROAS | CTR, CPA, Add-to-Cart Rate |
| Lookalike Expansion | New Customer LTV, CAC, ROAS | First Purchase Rate, Audience Quality |
| Customer Support Priority | Negative Review Reduction, Refund Rate Reduction | Contact Rate, Resolution Rate |
| Product Health Action | Rating Improvement, Return Rate Reduction | Pain Point Keyword Trend |

如果沒有 control group，就很難判斷成效是否來自行動本身。因此，實務上應該盡可能設計 treatment group 和 control group，衡量 incremental impact。

### Feedback Into the Decision Loop

當某個 action 沒有效果時，系統應該將結果回饋到下一輪決策中：

- 如果某類高 LTV 顧客對折扣反應差，下一輪可以測試不同 offer 或 channel。
- 如果負面情緒顧客對客服介入反應較好，可以提高 Customer Support Priority 的優先級。
- 如果購物車放棄顧客的 retargeting ROAS 不佳，可以重新檢查商品頁、價格、運費或 audience quality。
- 如果某個 segment 長期對 EDM 沒有反應，可以降低該 action 的優先級，避免浪費行銷資源。

這讓本專案不只是產生一次性的分析結果，而是建立一套可以持續驗證、學習與改善的 business decision loop。

---

## 4. Phase-by-Phase Breakdown

### Phase 1：RFM Segmentation + Sentiment Analysis

**Business Objective**

辨識顧客價值與關係狀態，讓團隊知道誰是高價值客、誰正在沉睡、誰需要被重新喚醒。

**What Analysis Is Done**

系統根據顧客最近購買時間、購買頻率、消費金額與評論情緒，建立顧客分群與情緒標籤。

**Output**

- `C360_Table.csv`
- `EDM_Suggestions.csv`

**Business Impact**

- CRM 團隊可以針對 `VVIP Loyal High-Value Customer`、`Recent New Customer`、`Dormant High-Value Customer` 設計不同溝通策略
- 行銷團隊可以避免用同一套折扣或訊息打所有會員
- 負面情緒顧客可以優先被客服或會員經營團隊處理

### Phase 2：Behavioral Features + Marketing Action Lists

**Business Objective**

把瀏覽、加購、評論與產品訊號轉換成可執行的行銷與產品名單。

**What Analysis Is Done**

系統整合顧客行為資料，找出購物車放棄、流失風險、高價值種子客群與產品健康問題。

**Output**

- `Cart_Abandonment_List.csv`
- `Churn_Risk_List.csv`
- `Lookalike_Seed_List.csv`
- `Product_Health_Report.csv`
- `Sleepy_Product_Health_Report.csv`
- `Brand_Health_Monitor.csv`

**Business Impact**

- 廣告團隊可以針對購物車放棄顧客做再行銷
- CRM 團隊可以針對流失風險顧客做挽回
- 成長團隊可以把高價值客群上傳到廣告平台做 Lookalike
- 產品團隊可以優先處理造成負評或流失的商品問題

### Phase 3：LTV Prediction Model

**Business Objective**

預估顧客未來價值，協助團隊把有限預算投在最有潛力產生回報的顧客身上。

**What Analysis Is Done**

系統根據顧客歷史消費、互動、分群與行為訊號，預估未來 90 天可能產生的價值。

**Output**

- `C360_Table.csv`
- `Model_Explanations_Table.csv`
- `Model_Run_Metadata.json`

**Business Impact**

- 行銷預算可以依照預期價值分配，而不是平均分配
- 高 LTV 顧客可搭配更高成本的挽回或維繫方案
- 低 LTV 顧客可採用低成本自動化溝通，提升整體投資效率

### Phase 4：Cohort Analysis + Product / Category Affinity Analysis

**Business Objective**

評估顧客生命週期品質，並找出商品或品類之間的搭配關係，支援 bundle、cross-sell、add-on recommendation 與 merchandising strategy。

**What Analysis Is Done**

系統觀察不同月份加入的顧客在後續月份的留存與收入表現，並先從 product-level 找出經常一起購買的 SKU 組合。如果 SKU 數量多、單一商品出現次數少，導致 product-level affinity 沒有足夠穩定結果，系統會自動 fallback 到 category-level affinity，從品類層級找出更穩定的交叉銷售與組合銷售機會。

**Output**

- `Cohort_Retention.csv`
- `Cohort_Revenue.csv`
- `Product_Affinity.csv`

**Business Impact**

- 管理層可以判斷某些活動帶來的是高品質新客還是一次性低價客
- 商品團隊可以設計組合包、加購推薦與交叉銷售策略
- 行銷團隊可以用 cohort 表現評估不同月份或活動的長期價值
- 採購與 merchandising 團隊可以從品類搭配關係規劃主題檔期、首頁陳列與推薦模組

**BA Interpretation**

Product-level affinity 適合做具體的 SKU-to-SKU 推薦，但當 SKU 很多、每個商品出現次數較少、商品搭配過於分散時，結果可能為空或不穩定。這不一定代表資料錯誤，也不代表沒有搭配機會，而是代表 SKU-level data 太稀疏，無法形成穩定訊號。因此本專案加入 category-level affinity fallback，用較高層級的品類關係找出更可靠的 cross-sell、bundle 與 add-on recommendation 方向。

### Phase 5：Business Decision Engine

**Business Objective**

把前面所有分析結果整合成最終的商業推薦清單，讓團隊知道每位顧客應該採取什麼行動，以及誰應該優先處理。

**What Analysis Is Done**

系統整合顧客分群、LTV、流失風險、購物車放棄、情緒與 Lookalike 訊號，套用商業規則產生推薦動作與優先級。

**Output**

- `Business_Recommendation_List.csv`
- `Top_Priority_Actions.csv`

**Business Impact**

- 團隊不需要自行判讀多張報表
- 每位顧客都有清楚的 recommended action
- 管理者可以直接從 Top 100 高優先名單開始分派任務
- 行銷、CRM、客服、產品可以共享同一份決策依據

---

## 5. Decision Engine (Phase 5 Highlight)

Phase 5 是本專案的核心商業決策層。它不是單純列出分析結果，而是把分析轉換成「下一步要做什麼」。

### How Recommendations Are Generated

系統會根據不同顧客訊號產生推薦行動：

- 高流失風險 + 高 LTV：`Retention Campaign (Discount + EDM)`
- 購物車放棄 + 高預測利潤：`Retargeting Campaign`
- VVIP 顧客：`Lookalike Expansion`
- 負面情緒 + 高購買頻率：`Customer Support Priority`

如果一位顧客同時符合多個條件，系統會依照商業優先順序選擇最重要的行動。例如，高價值且高流失風險的顧客會優先進入挽回活動，而不是一般廣告再行銷。

### How Priority Is Calculated

優先分數用來衡量「這位顧客值得多快被處理」：

```text
priority_score = predicted_LTV × churn_risk_score
```

這代表兩個商業意義：

- `predicted_LTV` 越高，代表顧客未來價值越大
- `churn_risk_score` 越高，代表不處理的風險越高

因此，高優先顧客通常是「值得挽回，而且需要立即挽回」的人。

### How Teams Decide What To Do First

實際工作流程可以這樣運作：

1. CRM Manager 每天打開 `Top_Priority_Actions.csv`
2. 先處理排名前 100 的高價值高風險顧客
3. 根據 `recommended_action` 分派給不同團隊
4. 根據 `reason` 理解推薦原因
5. 執行後追蹤轉換、回購、留存與客服結果

範例分工：

- CRM：執行 retention campaign、EDM、優惠券
- Performance Marketing：執行 retargeting 與 lookalike audience
- Customer Support：優先聯繫負面情緒高價值客
- Product：處理商品痛點與負評集中品類

---

## 6. Dashboard Layer (Visualization & Monitoring)

本專案的 CSV 輸出可以串接 BI 工具，形成 stakeholder-facing dashboard。Dashboard 的目的不是展示複雜模型，而是讓不同團隊快速做出營運決策。

### Key KPIs

Dashboard 建議追蹤以下核心指標：

- Revenue：總營收、每週營收、活動帶來的增量營收
- Conversion Rate：瀏覽到加購、加購到購買、再行銷轉換率
- Retention：cohort 留存率、回購率、`Dormant High-Value Customer` 喚醒率
- ROAS：廣告投放回收、Lookalike audience 表現
- AOV：平均訂單金額與商品組合銷售表現
- Churn Risk：高風險顧客數與高價值流失風險金額

### Dashboard Modules

**Funnel Analysis**

- 觀察瀏覽、加購、購買之間的流失點
- 判斷是否需要優化商品頁、購物車流程或再行銷策略

**Customer Segmentation**

- 顯示 `VVIP Loyal High-Value Customer`、`Recent New Customer`、`Dormant High-Value Customer`、`At-Risk Repeat Customer` 比例
- 支援 CRM 團隊設計分眾溝通策略

**Churn Risk Distribution**

- 監控高風險顧客數量與預估營收風險
- 協助管理者安排挽回預算與客服資源

**Top Priority Actions**

- 顯示最需要立即處理的顧客與推薦行動
- 讓團隊每天可以從同一份優先清單開始工作

**Product & Brand Health**

- 顯示負評集中商品、品類情緒變化與主要痛點
- 協助產品與營運團隊排定改善順序

### Who Uses This Dashboard

**Marketing Team**

- 決定再行銷受眾
- 評估 ROAS 與轉換漏斗
- 建立 Lookalike audience

**CRM Team**

- 執行高價值顧客維繫
- 安排流失挽回活動
- 設計分眾 EDM 與優惠策略

**Product Team**

- 找出影響留存與情緒的產品問題
- 根據痛點調整商品、描述、品質或供應商
- 設計組合商品與加購推薦

**Management Team**

- 追蹤營收、留存、轉換與高風險顧客變化
- 評估成長策略是否帶來長期價值
- 決定預算分配與跨部門優先順序

---

## 7. Stakeholder Story

### Scenario：轉換率下降後，行銷經理如何使用系統做決策

週一早上，Marketing Manager 發現上週整體營收下降 8%，但網站流量沒有明顯減少。這代表問題可能不是流量不足，而是轉換或顧客品質出現變化。

她先打開 dashboard 的 Funnel Analysis，發現「加購到購買」的轉換率下降，購物車放棄人數增加。接著她查看 `Cart_Abandonment_List.csv`，發現其中一批顧客的 `predicted_profit_90_days` 很高，代表這些人不是低價值流量，而是有明確購買意圖且值得挽回的潛在高價值客。

她再查看 Phase 5 的 `Top_Priority_Actions.csv`，發現系統已經把這些顧客中的高價值、高風險族群排在前面，並建議執行：

- Retargeting Campaign
- Retention Campaign (Discount + EDM)
- Customer Support Priority

她將名單分成三個行動：

- 對購物車放棄且高預測利潤顧客投放動態商品再行銷
- 對高 LTV 且高流失風險顧客發送限時折扣 EDM
- 對高頻購買但近期留下負評的顧客交由客服優先聯繫

一週後，她回到 dashboard 追蹤結果：

- 再行銷受眾的轉換率回升
- Top 100 高優先顧客中，有部分完成回購
- 客服介入後，負面評論比例下降
- 高風險顧客的預估流失金額下降

接著她把活動成效回饋給團隊：

- 高 LTV 顧客對限時折扣反應佳，下次提高該族群優先級
- 某品類的放棄率偏高，交由產品團隊檢查商品頁與評論痛點
- Lookalike audience 的 ROAS 若高於平均，下一輪增加廣告預算

這個流程讓團隊不是只看到「營收下降」，而是能快速定位問題、選擇行動、衡量結果，並把學到的經驗回饋到下一輪決策。

---

## 8. Business Impact Summary

這套系統解決的核心問題是：把資料從「看得到」推進到「能決策、能行動、能衡量」。

### Problems Solved

- 顧客名單太多，不知道誰要優先處理
- 行銷預算分散，無法集中在高回報機會
- 流失風險發生後才反應，缺乏提前預警
- 產品負評與營收影響沒有連在一起
- 分析結果停留在報表，沒有轉成團隊行動

### Decision-Making Improvements

- 從平均行銷改為分眾行銷
- 從事後檢討改為提前預警
- 從單一 KPI 追蹤改為完整決策閉環
- 從人工判斷名單改為系統化優先排序
- 從部門各自解讀資料改為共享同一套 decision layer

### Simulated Business Impact

若此系統導入日常營運，可合理期待以下改善方向：

- 高價值流失顧客挽回率提升
- 購物車放棄再行銷轉換率提升
- 廣告 ROAS 因 Lookalike 種子品質提升而改善
- CRM 團隊處理名單的效率提升
- 產品問題被更早發現，降低負評與退款風險
- 管理層能更快判斷營收變化背後的原因

簡單來說，本專案將資料分析、預測模型、行銷行動與成效衡量串成一個完整的商業決策系統，讓電商團隊能更快找到問題、更精準配置資源，並持續把每一次行動的結果轉化為下一次更好的決策。

---

## Output Data Products

主要輸出檔案位於 `schema/`：

- `C360_Table.csv`：顧客 360 全貌與分群
- `Churn_Risk_List.csv`：流失風險名單
- `Cart_Abandonment_List.csv`：購物車放棄再行銷名單
- `Lookalike_Seed_List.csv`：高價值相似客群種子名單
- `Product_Health_Report.csv`：產品健康度與痛點
- `Brand_Health_Monitor.csv`：品類與品牌情緒監控
- `Cohort_Retention.csv`：同期群留存
- `Cohort_Revenue.csv`：同期群收入
- `Product_Affinity.csv`：商品或品類關聯與組合機會；若 product-level 資料太稀疏，系統會自動改用 category-level fallback
- `Business_Recommendation_List.csv`：完整商業推薦清單
- `Top_Priority_Actions.csv`：前 100 名高優先行動名單

### Output Value Labels

CSV 中的顧客分群與狀態值統一使用英文，方便串接 BI、CRM、廣告平台與自動化工具。

| 欄位 | 可能值 |
| --- | --- |
| `segment` | `VVIP Loyal High-Value Customer` |
| `segment` | `Dormant High-Value Customer` |
| `segment` | `Recent New Customer` |
| `segment` | `At-Risk Repeat Customer` |
| `segment` | `General Potential Customer` |
| `alert_level` | `Healthy` |
| `alert_level` | `Medium Risk: Monitor Closely` |
| `alert_level` | `High Risk: Immediate Intervention` |
| `action_required` | `Product Team Review` |
| `action_required` | `Consider Re-exposure or Recommendation to Dormant Customers` |

### Product_Affinity.csv Schema

`Product_Affinity.csv` 用來支援商品推薦、組合包、交叉銷售與 merchandising strategy。它可能是 product-level 結果，也可能是 category-level fallback 結果，取決於 SKU 層級是否有足夠穩定的共同購買訊號。

| 欄位 | 說明 |
| --- | --- |
| `affinity_level` | affinity 分析層級，可能是 `product` 或 `category` |
| `product_a_id` | 商品 A ID；若為 category-level fallback，則為空 |
| `product_a_name` | 商品 A 名稱；若為 category-level fallback，則為空 |
| `product_a_category` | 商品 A 品類或 Category A |
| `product_b_id` | 商品 B ID；若為 category-level fallback，則為空 |
| `product_b_name` | 商品 B 名稱；若為 category-level fallback，則為空 |
| `product_b_category` | 商品 B 品類或 Category B |
| `co_occurrence_count` | A 和 B 在同一張訂單共同出現的次數 |
| `support` | A 和 B 一起出現的訂單數 / 總訂單數 |
| `confidence_A_to_B` | 出現 A 的訂單中，有多少比例也出現 B |

---

## How To Run

```bash
pip install -r requirements.txt
python main.py
```
