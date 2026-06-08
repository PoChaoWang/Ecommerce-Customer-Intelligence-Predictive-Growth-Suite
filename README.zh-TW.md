# 電商商業決策系統 (E-Commerce Business Decision System)

[English](README.md) | [繁體中文](README.zh-TW.md)

本專案是一套專為電商成長打造的商業決策系統。其目標不僅是產生分析報表或預測模型，而是將分散的客戶、交易、瀏覽、購物車、評論與產品數據，轉化為行銷、CRM 及產品團隊可以直接使用的優先行動清單與決策建議。

它回答了運營團隊每天面臨的實務問題：

- 哪些客戶應該優先挽回？
- 哪些購物車放棄的客戶最有可能轉換？
- 哪些高價值客戶可以用作相似受眾 (Lookalike) 的種子名單？
- 哪些產品或品類正在造成負面的客戶體驗？
- 行銷資源應該優先投入在何處？

---

## 1. 專案概覽 (商業框架)

電商公司通常不缺乏數據，真正的挑戰在於數據往往無法快速轉化為決策。當營收下降時，行銷團隊可能不知道問題是來自新客品質下降、老客流失、產品體驗問題，還是漏斗摩擦。CRM 團隊可能有龐大的會員名單，但卻不清楚應該先聯繫誰。產品團隊可能收到負評，但不知道哪些問題會直接影響留存率和營收。

本專案建立了一個「數據到行動」的決策工作流，幫助團隊：

- 透過識別高價值客戶與高轉換機會來提升營收
- 透過及早偵測高風險客戶並分配挽回行動來降低流失率
- 透過將預算集中在最可能產生回報的客戶來提升行銷效率
- 透過從評論與留存訊號中識別問題產品來改善產品體驗
- 建立行動優先級，讓團隊知道應優先處理什麼

最終輸出不是一份靜態分析，而是一套可重複執行的商業決策閉環。

---

## 2. 端到端決策閉環 (End-to-End Decision Loop)

系統設計為一個完整的閉環：

**數據 (Data) -> 洞察 (Insight) -> 行動 (Action) -> 衡量 (Measurement) -> 回饋 (Feedback) -> 模型改進 (Model Improvement)**

### 數據：整合客戶與營運數據
系統將訂單、瀏覽行為、購物車活動、評論、產品及客戶數據整合到單一客戶視角中。這讓團隊能超越單一交易，理解每位客戶的價值、活躍度、情緒、流失風險及未來潛力。

### 洞察：將數據轉化為商業洞察
分析回答了具體的商業問題，例如：
- 哪些客戶是 `VVIP Loyal High-Value Customer` 或 `Dormant High-Value Customer`？
- 哪些客戶具有高流失風險？
- 哪些購物車放棄事件值得進行再行銷？
- 哪些產品經常被一起購買？
- 哪些獲客時段的留存率較弱？

### 行動：將洞察轉化為可執行的名單
系統輸出以行動為導向的 CSV 檔案，如流失挽回名單、購物車放棄名單、相似客群種子名單以及最終建議名單。這些名單可直接交付給 CRM、付費媒體、EDM、客服或產品團隊。

### 衡量：追蹤行動成效
每個行動都可以透過商業指標進行追蹤，例如：
- 留存活動 (Retention Campaign) 是否降低了流失率？
- 再行銷活動 (Retargeting Campaign) 是否提升了轉換率？
- 相似客群擴展 (Lookalike Expansion) 是否改善了 ROAS？
- 客服優先處理 (Customer Support Priority) 是否減少了負評和退款？

### 回饋：將結果反饋至決策規則
如果某個分群對折扣反應良好，可以提高其優先級。如果某種客戶類型對 EDM 反應不佳，則可調整頻道或訊息。該系統的價值在於能夠持續學習行銷行動與客戶反應之間的關係。

### 模型改進：持續優化預測與優先級排序
LTV、流失風險、客戶分群與推薦規則可以隨著新數據和活動結果的加入而更新。這使決策過程變得可營運、可監控且可優化，而非一次性的分析。

---

## 3. 行動成效審視：如果建議無效該怎麼辦？

在現實商業環境中，並非每個洞察或建議行動都能立即見效。這並不代表分析失敗，而是意味著原始的商業假設需要重新驗證。

因此，本系統不將建議行動視為決策過程的終點，而是將每個行動視為一個可追蹤、可驗證且可改進的商業假設。

當某個行動未達預期結果時，商業分析師 (BA) 應從四個角度進行診斷：

### 1. 洞察驗證 (Insight Validation)
檢查原始洞察是否識別出了真正的問題。
例如，系統可能將一群客戶識別為高 LTV 且高流失風險，隨後建議進行挽回活動。然而，如果流失的真正原因是產品品質、物流體驗或售後服務，單純的折扣可能不足以提升回購率。

### 2. 行動適配 (Action Fit)
檢查建議的行動是否真正解決了客戶的主要障礙。
例如：
- 如果客戶的主要問題是價格敏感度，折扣或優惠券可能有效。
- 如果客戶的主要問題是負評或產品體驗，客服介入或產品改進可能更有效。
- 如果客戶只是單純放棄購物車，再行銷廣告或提醒 EDM 可能比通用的挽回活動更合適。

因此，建議行動應根據分群、預測 LTV、流失風險、情緒訊號、購物車行為及主要障礙持續調整。

### 3. 執行檢查 (Execution Check)
檢查行動是否被正確執行。
即使洞察與策略合理，執行過程中仍可能出現問題，例如：
- EDM 發送給錯誤的對象
- 優惠券設定錯誤
- 廣告受眾上傳不完整
- 活動訊息與客戶痛點不匹配
- 發送時間與客戶購買週期不一致
- 落地頁或產品頁體驗不佳

當表現疲軟時，團隊不應僅審視分析模型，還必須檢查活動執行情況。

### 4. 衡量審視 (Measurement Review)
檢查 KPI、觀察窗口及評估方法是否合適。
不同行動應使用不同的成功指標：

| 建議行動 | 主要 KPI | 診斷 KPI |
|---|---|---|
| 挽回活動 (Retention Campaign) | 回購率, 留存率 | 開啟率, 點擊率 (CTR), 優惠券使用率 |
| 再行銷活動 (Retargeting Campaign) | 轉換率, ROAS | 點擊率 (CTR), CPA, 加入購物車率 |
| 相似客群擴展 (Lookalike Expansion) | 新客 LTV, CAC, ROAS | 首購率, 受眾品質 |
| 客服優先處理 (Customer Support Priority) | 負評減少量, 退款率降低量 | 聯繫率, 解決率 |
| 產品健康行動 (Product Health Action) | 評分提升, 退貨率降低 | 痛點關鍵字趨勢 |

如果沒有對照組，很難判斷改進是否來自行動本身。在實務中，團隊應盡可能設計實驗組與對照組來衡量增量影響 (incremental impact)。

### 反饋至決策閉環
當行動無效時，結果應反饋至下一個決策週期：
- 如果高 LTV 客戶群對折扣反應不佳，下次測試不同的優惠或頻道。
- 如果負面情緒客戶對客服介入反應更好，則增加「客服優先處理」的權重。
- 如果購物車放棄客戶的再行銷 ROAS 較低，重新檢查產品頁、價格、運費或受眾品質。
- 如果某個分群在 EDM 上的表現持續不佳，降低該行動的優先級以避免浪費行銷資源。

這讓專案不僅是一次性的分析，而是建立了一個能持續驗證、學習與改進的商業決策閉環。

---

## 4. 各階段詳細分解

### 階段 1：RFM 分群 + 情緒分析
**商業目標**
識別客戶價值與關係狀態，讓團隊知道誰是高價值客、誰處於沉睡狀態、誰需要被重新激活。

**進行了哪些分析**
系統根據最近一次購買時間 (Recency)、購買頻率 (Frequency)、消費金額 (Monetary) 與評論情緒建立客戶分群與情緒標籤。

**輸出**
- `C360_Table.csv`
- `EDM_Suggestions.csv`

**商業影響**
- CRM 團隊可針對 `VVIP Loyal High-Value Customer`、`Recent New Customer` 與 `Dormant High-Value Customer` 設計不同的溝通策略。
- 行銷團隊可避免對所有會員使用相同的折扣或訊息。
- 負面情緒客戶可由客服或會員運營團隊優先處理。

### 階段 2：行為特徵 + 行銷行動清單
**商業目標**
將瀏覽、購物車、評論與產品訊號轉化為可執行的行銷與產品行動清單。

**進行了哪些分析**
系統整合客戶行為數據，識別購物車放棄、流失風險、高價值種子受眾及產品健康問題。

**輸出**
- `Cart_Abandonment_List.csv`
- `Churn_Risk_List.csv`
- `Lookalike_Seed_List.csv`
- `Product_Health_Report.csv`
- `Sleepy_Product_Health_Report.csv`
- `Brand_Health_Monitor.csv`

**商業影響**
- 付費媒體團隊可以針對購物車放棄客戶進行再行銷。
- CRM 團隊可以為流失風險客戶執行挽回活動。
- 成長團隊可以將高價值客戶上傳至廣告平台進行相似客群擴展。
- 產品團隊可以優先處理造成負評或流失的問題。

### 階段 3：LTV 預測模型
**商業目標**
估計客戶未來價值，以便團隊將有限的預算分配給預期回報最高的客戶。

**進行了哪些分析**
系統根據歷史支出、互動、分群及行為訊號，預測未來 90 天的潛在價值。

**輸出**
- `C360_Table.csv`
- `Model_Explanations_Table.csv`
- `Model_Validation_Table.csv`
- `Model_Run_Metadata.json`

**商業影響**
- 行銷預算可按預期價值分配，而非平均分配給所有客戶。
- 高 LTV 客戶可獲得更高成本的挽回或忠誠度待遇。
- 低 LTV 客戶可透過低成本的自動化溝通進行管理，提升整體投資效率。

**模型驗證 (Model Validation)**
`Model_Validation_Table.csv` 專為模型驗證儀表板設計，用於評估階段 3 LTV 預測模型是否可靠。它回答了以下問題：
- 模型整體上是高估還是低估了？
- 預測誤差是否隨分群而異？
- 預測價值最高的客戶實際上是否產生了更高的利潤？
- 預測的前段客戶是否比平均客戶更有價值？

`predicted_profit_90_days` 為模型預測值，不代表該價值一定會發生。模型驗證不應僅關注整體的 MAE 或 RMSE，因為該模型的主要用途是客戶價值排序與行銷資源分配優先級。因此，預測十分位 (decile) 驗證與前段預測群組的表現也應納入審視。如果預測前 10% 或 20% 客戶的實際利潤明顯高於平均水平，即使模型未能完美預測每位客戶的精確金額，該模型仍具有商業決策價值。

`actual_profit` 在缺失時補 0，以避免僅驗證有購買行為的客戶而高估模型表現。

每個 `Model_Validation_Table.csv` 還記錄了 `validation_run_date`、`prediction_window_start` 與 `prediction_window_end`。儀表板可利用這些欄位顯示驗證結果的生成時間，以及用於計算實際利潤的觀察期間。

**模型執行元數據 (Model Run Metadata)**
`Model_Run_Metadata.json` 存儲模型執行歷史，而不僅是最新結果。每次執行 `main.py` 時，都會在 `runs` 數組中添加新記錄，包括：
- `run_id`
- `run_timestamp`
- `model_type`
- `performance_metrics`
- `feature_importance`
- `retrain_triggered` (是否觸發重新訓練)

此歷史紀錄用於監控模型表現是否隨時間惡化，並支援模型監控與重新訓練決策。`retrain_triggered` 邏輯為：如果當前 RMSE 較前次執行增加超過 15%，則標記為 `true`。

### 階段 4：同期群分析 (Cohort Analysis) + 產品/品類關聯分析
**商業目標**
評估客戶生命週期品質，並識別產品或品類間的關係，以支援組合包、交叉銷售、加價購推薦及商品企劃策略。

**進行了哪些分析**
系統觀察不同月份獲取的客戶在留存與營收上的表現。系統首先識別產品層級經常共同購買的 SKU 組。如果 SKU 數量過多且單一產品出現頻率過低，導致產品層級的關聯不穩定或為空，系統會自動切換至品類層級的關聯分析，以識別更穩定的交叉銷售與組合機會。

**輸出**
- `Cohort_Retention.csv`
- `Cohort_Revenue.csv`
- `Product_Affinity.csv`

**商業影響**
- 管理層可判斷活動帶來的是高品質新客還是單次低價買家。
- 產品團隊可設計組合包、加價購推薦與交叉銷售策略。
- 行銷團隊可利用同期群表現評估不同月份或活動的長期價值。
- 採購與企劃團隊則可根據品類關係規劃活動、首頁版位及推薦模組。

**BA 解讀**
產品層級的關聯分析適用於具體的 SKU 到 SKU 推薦。然而，當 SKU 眾多、產品頻率低且組合高度分散時，結果可能為空或不穩定。這不一定代表數據問題或缺乏搭配機會，而是意味著 SKU 層級數據太稀疏，無法形成穩定訊號。因此，本專案包含品類層級的關聯回退機制，以在更高層次識別更可靠的交叉銷售與組合方向。

### 階段 5：商業決策引擎
**商業目標**
整合先前所有的分析結果，生成最終的商業建議清單，讓團隊知道對每位客戶應採取什麼行動，以及應該優先處理誰。

**進行了哪些分析**
系統結合客戶分群、LTV、流失風險、購物車放棄、情緒及相似客群訊號，應用商業規則生成建議行動與優先級。

**輸出**
- `Business_Recommendation_List.csv`
- `Top_Priority_Actions.csv`

**商業影響**
- 團隊無需手動判讀多個報表。
- 每位客戶都有明確的建議行動。
- 管理者可以直接從前 100 名優先行動清單開始分配工作。
- 行銷、CRM、客服及產品團隊共享相同的決策基礎。

---

## 5. 決策引擎 (階段 5 亮點)

階段 5 是本專案的核心商業決策層。它不僅是列出分析結果，而是將分析轉化為「接下來該做什麼」。

### 建議是如何生成的
系統根據不同的客戶訊號生成建議行動：
- 高流失風險 + 高 LTV：`Retention Campaign (Discount + EDM)`
- 購物車放棄 + 高預測利潤：`Retargeting Campaign`
- VVIP 客戶：`Lookalike Expansion`
- 負面情緒 + 高購買頻率：`Customer Support Priority`

如果客戶符合多個條件，系統會根據商業優先級選擇最重要的行動。例如，高價值且高流失風險的客戶將優先進行挽回，而非一般的廣告再行銷。

### 優先級是如何計算的
優先級分數衡量了處理客戶的緊急程度：
```text
priority_score = predicted_LTV x churn_risk_score
```
這具有兩個商業意義：
- `predicted_LTV` 越高，客戶的未來價值越大。
- `churn_risk_score` 越高，不採取行動的風險越大。

因此，高優先級客戶通常是那些值得挽回且需要快速挽回的人。

### 團隊如何決定先做什麼
實際工作流可能如下：
1. CRM 經理每天打開 `Top_Priority_Actions.csv`。
2. 團隊首先處理前 100 名高價值、高風險客戶。
3. 根據 `recommended_action` 將工作分配給不同團隊。
4. 團隊透過 `reason` 理解為何提出該建議。
5. 執行後追蹤轉換、回購、留存及客服結果。

範例分工：
- CRM：挽回活動、EDM、優惠券。
- 效能行銷：再行銷與相似受眾擴展。
- 客服：優先聯繫負面情緒的高價值客戶。
- 產品：處理產品痛點與負評集中的品類。

---

## 6. 儀表板層 (視覺化與監控)

本專案的 CSV 輸出可連接至 BI 工具以創建面向利益相關者的儀表板。儀表板的目標不是展示複雜模型，而是幫助不同團隊快速做出營運決策。

### 核心 KPI
建議的儀表板 KPI 包括：
- 營收：總營收、週營收、活動帶來的增量營收。
- 轉換率：瀏覽到購物車、購物車到購買、再行銷轉換率。
- 留存：同期群留存、回購率、`Dormant High-Value Customer` 激活率。
- ROAS：廣告回報與相似受眾表現。
- AOV：平均訂單價值與組合包銷售表現。
- 流失風險：高風險客戶數量與受流失威脅的高價值營收。

### 儀表板模組

**漏斗分析 (Funnel Analysis)**
- 監控瀏覽、加入購物車與購買之間的流失點。
- 判斷產品頁、購物車流程或再行銷策略是否需要優化。

**客戶分群 (Customer Segmentation)**
- 顯示 `VVIP Loyal High-Value Customer`、`Recent New Customer`、`Dormant High-Value Customer` 及 `At-Risk Repeat Customer` 的比例。
- 支援 CRM 團隊設計分層溝通策略。

**流失風險分佈 (Churn Risk Distribution)**
- 監控高風險客戶數量與估計的營收風險。
- 幫助管理者分配挽回預算與客服資源。

**最優先行動 (Top Priority Actions)**
- 顯示需要立即關注的客戶及其建議行動。
- 幫助團隊每天從同一個優先級清單開始工作。

**模型驗證 (Model Validation)**
| 儀表板部分 | 使用欄位 | 目的 |
| --- | --- | --- |
| 整體模型表現 | `absolute_error`, `squared_error`, `prediction_error`, `predicted_profit_90_days`, `actual_profit_filled` | 顯示 MAE, RMSE, 平均預測利潤, 平均實際利潤及偏差 (Bias) |
| 分群層級誤差 | `segment`, `absolute_error`, `squared_error`, `prediction_error` | 比較不同客戶分群的模型誤差 |
| 預測十分位驗證 | `prediction_decile_label`, `predicted_profit_90_days`, `actual_profit_filled` | 檢查預測利潤較高的組別是否確實有較高的實際利潤 |
| 前段預測群組表現 | `top_10_flag`, `top_20_flag`, `top_30_flag`, `actual_profit_filled` | 驗證預測前段的客戶是否比平均客戶更有價值 |

**產品與品牌健康度**
- 顯示負評集中產品、品類情緒變化及主要痛點。
- 幫助產品與營運團隊優先處理改進事項。

### 誰使用此儀表板

**行銷團隊**
- 決定再行銷受眾。
- 評估 ROAS 與轉換漏斗。
- 建立相似受眾。

**CRM 團隊**
- 執行高價值客戶留存。
- 安排流失挽回活動。
- 設計分層 EDM 與優惠策略。

**產品團隊**
- 識別影響留存與情緒的產品問題。
- 根據痛點調整產品、描述、品質或供應商。
- 設計組合包與加價購推薦。

**管理團隊**
- 追蹤營收、留存、轉換及高風險客戶的變化。
- 評估成長策略是否創造長期價值。
- 決定預算分配與跨職能優先級。

---

## 7. 利益相關者故事

### 情境：行銷經理在轉換率下降後如何使用系統

週一早上，行銷經理發現上週總營收下降了 8%，而網站流量並未顯著下降。這表明問題可能不是流量不足，而是轉換或客戶品質發生了變化。

她首先打開漏斗分析儀表板，發現「購物車到購買」的轉換率下降，且購物車放棄量增加。接著她檢查 `Cart_Abandonment_List.csv`，發現有一群客戶的 `predicted_profit_90_days` 很高。這意味著他們不是低價值流量；他們有明確的購買意向，是值得挽回的潛在高價值客戶。

隨後她檢查階段 5 的 `Top_Priority_Actions.csv`，發現系統已經將高價值、高風險客戶排在首位，並建議：
- 再行銷活動 (Retargeting Campaign)
- 挽回活動 (折扣 + EDM)
- 客服優先處理 (Customer Support Priority)

她將清單拆分為三個行動：
- 針對高預測利潤的購物車放棄客戶運行動態產品再行銷。
- 向高 LTV 且高流失風險的客戶發送限時折扣 EDM。
- 指派客服優先聯繫近期留下負評的高頻買家。

一週後，她回到儀表板追蹤結果：
- 再行銷受眾的轉換率有所回升。
- 前 100 名高優先客戶中有部分進行了回購。
- 客服介入後，負評比例下降。
- 估計的流失風險營收下降。

隨後她向團隊反饋活動結果：
- 高 LTV 客戶對限時折扣反應良好，因此下一個週期應增加該組別的優先級。
- 某個品類的放棄率較高，產品團隊應檢查產品頁與評論痛點。
- 如果相似受眾的 ROAS 高於平均水平，下一個週期應增加廣告預算。

此工作流讓團隊超越了單純看到「營收下降」的表面現象。它幫助他們快速定位問題、選擇行動、衡量結果，並將習得的經驗反饋至下一個決策週期。

---

## 8. 商業影響總結

該系統解決的核心問題是將數據從「可視化」推進到「決策就緒、可行動且可衡量」。

### 解決的問題
- 客戶名單過多，缺乏明確優先級。
- 行銷預算過於分散，而非集中在高回報機會上。
- 流失風險僅在發生後才被偵測到，缺乏預警。
- 產品負評未與營收影響掛鉤。
- 分析停留在報表層面，未轉化為團隊行動。

### 決策改進
- 從平均行銷轉向分眾行銷。
- 從事後檢討轉向事前預警。
- 從單一 KPI 追蹤轉向完整的決策閉環。
- 從人工判斷名單轉向系統化優先級排序。
- 從各部門各自解讀轉向共享決策層。

### 模擬商業影響
若將此系統引入日常營運，可合理預期以下改進：
- 高價值流失風險客戶的挽回率提升。
- 購物車放棄客戶的再行銷轉換率提升。
- 由於相似種子受眾品質提升，廣告 ROAS 得到改善。
- CRM 名單處理效率提升。
- 產品問題更早被發現，降低負評與退款風險。
- 管理層能更快診斷營收變化的背後原因。

簡而言之，本專案將數據分析、預測模型、行銷行動與表現衡量串聯成一個完整的商業決策系統。它幫助電商團隊更快發現問題、更精準地配置資源，並持續將每次行動的結果轉化為下一個週期更好的決策。

---

## 產出數據產品 (Output Data Products)

主要的輸出文件位於 `schema/` 目錄下：

- `C360_Table.csv`：客戶 360 視角與分群
- `Model_Run_Metadata.json`：LTV 模型執行歷史，用於追蹤每次執行的表現、特徵重要性與重新訓練警告。
- `Model_Validation_Table.csv`：模型驗證儀表板專用表，用於驗證 LTV 預測誤差、排序能力及高預測群組的實際價值。
- `Churn_Risk_List.csv`：流失風險客戶名單
- `Cart_Abandonment_List.csv`：購物車放棄再行銷名單
- `Lookalike_Seed_List.csv`：高價值相似受眾種子名單
- `Product_Health_Report.csv`：產品健康度與痛點
- `Brand_Health_Monitor.csv`：品類與品牌情緒監控
- `Cohort_Retention.csv`：同期群留存
- `Cohort_Revenue.csv`：同期群營收
- `Product_Affinity.csv`：產品或品類關聯與組合機會；若產品層級數據太稀疏，系統自動回退至品類層級。
- `Business_Recommendation_List.csv`：完整的商業建議清單
- `Top_Priority_Actions.csv`：前 100 名高優先行動名單

### Model_Run_Metadata.json 結構 (Schema)
`Model_Run_Metadata.json` 存儲了階段 3 LTV 模型的多個執行記錄。頂層 JSON 對象包含一個 `runs` 數組。每次執行 `main.py` 時，都會附加一條新記錄，而非覆蓋舊記錄。

| 欄位 | 描述 |
| --- | --- |
| `run_id` | 模型執行 ID，格式為 `YYYYMMDD_HHMMSS` |
| `run_timestamp` | 模型執行時間 |
| `model_type` | 使用的模型類型 |
| `performance_metrics` | 該次執行的模型表現指標，如 RMSE 和 MAE |
| `feature_importance` | 該次執行的特徵重要性 |
| `retrain_triggered` | 是否觸發了重新訓練警告 |

`retrain_triggered` 支援模型監控與重新訓練決策。如果當前 RMSE 較前次執行增加超過 15%，則當前執行將被標記為 `true`，提醒團隊檢查數據漂移、模型退化或需要重新訓練。

### Model_Validation_Table.csv 結構 (Schema)
`Model_Validation_Table.csv` 支援模型驗證儀表板，驗證階段 3 LTV 預測模型是否能提供可靠的商業排序與資源分配訊號。

| 欄位 | 描述 |
| --- | --- |
| `validation_run_date` | 生成此模型驗證表的時間 |
| `prediction_window_start` | 實際驗證窗口的開始日期，用於計算預測期間的實際利潤 |
| `prediction_window_end` | 實際驗證窗口的結束日期，用於計算預測期間的實際利潤 |
| `user_id` | 客戶 ID |
| `segment` | 客戶分群 |
| `predicted_profit_90_days` | 模型預測的未來 90 天利潤 |
| `actual_profit` | 原始實際 90 天利潤；如果客戶在預測期間未購買，可能為空 |
| `actual_profit_filled` | 將缺失的 `actual_profit` 補 0 後的實際利潤，用於模型驗證 |
| `prediction_error` | 預測利潤減去實際利潤；正值表示高估，負值表示低估 |
| `absolute_error` | 絕對誤差，用於計算 MAE |
| `squared_error` | 平方誤差，用於計算 RMSE |
| `prediction_decile` | 按預測利潤降序將客戶分為 10 組；1 表示前 10% |
| `prediction_decile_label` | 易讀的十分位標籤 |
| `top_10_flag` | 客戶是否屬於預測利潤的前 10% |
| `top_20_flag` | 客戶是否屬於預測利潤的前 20% |
| `top_30_flag` | 客戶是否屬於預測利潤的前 30% |

### 模型驗證儀表板使用方式
| 儀表板部分 | 使用欄位 | 目的 |
| --- | --- | --- |
| 整體模型表現 | `absolute_error`, `squared_error`, `prediction_error`, `predicted_profit_90_days`, `actual_profit_filled` | 顯示 MAE, RMSE, 平均預測利潤, 平均實際利潤及偏差 |
| 分群層級誤差 | `segment`, `absolute_error`, `squared_error`, `prediction_error` | 比較不同客戶分群的模型誤差 |
| 預測十分位驗證 | `prediction_decile_label`, `predicted_profit_90_days`, `actual_profit_filled` | 檢查預測利潤較高的組別是否確實有較高的實際利潤 |
| 前段預測群組表現 | `top_10_flag`, `top_20_flag`, `top_30_flag`, `actual_profit_filled` | 驗證預測前段的客戶是否比平均客戶更有價值 |

`validation_run_date` 可顯示驗證結果的生成時間。`prediction_window_start` 與 `prediction_window_end` 可顯示用於模型驗證的實際利潤觀察期間，例如：
```text
此驗證結果基於從 prediction_window_start 到 prediction_window_end 的實際客戶利潤。
```

目前 `Model_Validation_Table.csv` 代表最新模型執行的驗證結果。如果未來需要追蹤跨多次模型執行的表現，可以擴展為追加模式的 `Model_Validation_History.csv`。

### 輸出值標籤 (Output Value Labels)
CSV 文件中的客戶分群與狀態值統一使用英文，以支援 BI、CRM、廣告平台及自動化工具。

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

### Product_Affinity.csv 結構 (Schema)
`Product_Affinity.csv` 支援產品推薦、組合包、交叉銷售及商品企劃。它可能包含產品層級結果或品類層級回退結果，取決於 SKU 層級數據是否有足夠穩定的共同購買訊號。

| 欄位 | 描述 |
| --- | --- |
| `affinity_level` | 關聯分析層級，`product` 或 `category` |
| `product_a_id` | 產品 A ID；品類層級回退時為空 |
| `product_a_name` | 產品 A 名稱；品類層級回退時為空 |
| `product_a_category` | 產品 A 品類或品類 A |
| `product_b_id` | 產品 B ID；品類層級回退時為空 |
| `product_b_name` | 產品 B 名稱；品類層級回退時為空 |
| `product_b_category` | 產品 B 品類或品類 B |
| `co_occurrence_count` | A 與 B 在同一個訂單中出現的次數 |
| `support` | 同時包含 A 和 B 的訂單數 / 總訂單數 |
| `confidence_A_to_B` | 在包含 A 的訂單中，同時包含 B 的比例 |

---

## 如何運行

```bash
pip install -r requirements.txt
python main.py
```