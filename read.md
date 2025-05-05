📝 模擬資料分析程式說明（segmentation_analysis.py）
本程式模擬一組顧客資料，並進行三種機器學習任務：分類、回歸、分群。目的是模擬真實商業應用中對顧客行為的預測與分析。

🔧 1. 資料模擬與前處理
使用 make_classification 和 make_regression 建立 1000 筆樣本的模擬資料，並整合成一個 DataFrame，包含以下欄位：

Age：顧客年齡（連續數值）

Income：收入

PurchaseFreq：購買頻率

Membership：會員指標（例如：是否為付費會員）

Segment：模擬的顧客類型（共三類，作為分類標籤）

PurchaseAmount：購買金額（連續變數，用來做回歸）

Gender：隨機產生的性別欄位（僅做為類別參考）

📊 2. 分群分析（Clustering）
使用 KMeans 進行非監督式分群，根據 Age、Income、PurchaseFreq 三個欄位將顧客分為 3 群。

為了方便視覺化，利用主成分分析（PCA）將原始特徵降到 2 維，並繪製出不同群組的分佈圖。

目標：找出顧客的自然分群結構，協助行銷分眾策略。

🎯 3. 分類任務（Classification）
使用 RandomForestClassifier 建立一個分類模型，目標是預測顧客屬於哪個 Segment（顧客類型），使用的特徵為：

Age、Income、PurchaseFreq、Membership

訓練完成後輸出：

分類報告（Precision、Recall、F1-score）

混淆矩陣圖，視覺化預測準確度

應用情境：推薦系統可預測新顧客屬於哪種類型，給予對應產品或優惠。

📈 4. 回歸任務（Regression）
使用 GradientBoostingRegressor 預測每位顧客的 PurchaseAmount（購買金額），使用相同的四個特徵欄位。

模型輸出：

RMSE（均方根誤差）和 R²（決定係數）

預測值 vs 實際值散點圖

應用情境：評估顧客未來可能貢獻的消費金額，作為潛在價值參考。

📌 結論
這份程式碼展示了完整的機器學習實驗流程，包括資料模擬、分群可視化、分類預測、回歸預測，適合用來做為資料分析與模型比較的範例。使用者也可以修改特徵與模型，觀察效能變化。