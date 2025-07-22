# clip_autophoto

專案檔案結構解析
traffic_intent_clf/

├── images/             ← 儲存所有行人圖片的資料夾

│   ├── img1.jpg

│   ├── img2.jpg

│   └── ...

├── labels.csv          ← 圖片檔名及其對應意圖類別的標籤檔

├── extract_features.py ← 負責從圖片中提取特徵向量的腳本

└── train_classifier.py ← 負責訓練 SVM 分類器並評估其性能的腳本


labels.csv 檔案說明
這個 CSV 檔案是整個專案的基礎，它定義了每張圖片的真實標籤 (Ground Truth Label)，也就是我們希望模型能夠學習辨識的行人意圖。


extract_features.py
這個腳本的目的是將圖片轉換成機器學習模型可以理解的數值表示 (Numerical Representation)，也就是特徵向量。它利用了 CLIP 模型來完成這項任務。

核心概念：CLIP 模型
CLIP 是一種由 OpenAI 開發的強大模型，它能夠理解圖片和文字之間的關聯性。在這裡，我們主要使用它的圖片編碼器 (Image Encoder) 部分。這個編碼器能夠將任何圖片轉換成一個固定維度的向量（在這個專案中是 512 維），這個向量捕捉了圖片的視覺資訊。


train_classifier.py 
這個腳本的目的是使用 image_features.csv 中提取出的圖片特徵來訓練一個支援向量機 (SVM) 分類器。

核心概念：支援向量機 (SVM)
SVM 是一種強大的監督式學習模型，常用於分類任務。它的目標是找到一個「最佳」的超平面（或多個超平面），將不同類別的資料點分開，並且使分類邊界到最近資料點的距離最大化，從而提高泛化能力。


專案執行流程總結
準備資料：確保 images/ 資料夾中有圖片，並且 labels.csv 正確地標註了每張圖片的意圖。

執行特徵提取：
python extract_features.py
這會讀取 images/ 中的圖片和 labels.csv，然後利用 CLIP 模型為每張圖片生成一個 512 維的特徵向量，並將結果儲存到 image_features.csv。

執行分類器訓練：
python train_classifier.py
這會讀取 image_features.csv，將資料分為訓練集和測試集，訓練一個線性 SVM 分類器，並最終輸出模型在測試集上的性能報告。

專案的優點與可能改進方向
優點：
利用預訓練模型 (CLIP)：CLIP 是一個非常強大的預訓練模型，它已經從大量的圖片和文字資料中學習到了豐富的視覺語義資訊，這使得我們能夠在相對較少的資料上獲得不錯的特徵表示，避免從頭訓練一個大型的深度學習模型。
模組化設計：將特徵提取和分類器訓練分開，使得專案結構清晰，便於管理和除錯。
傳統分類器 (SVM)：在特徵足夠好的情況下，SVM 是一種性能優異且解釋性較強的分類器。
標準評估：使用 classification_report 提供了全面的模型性能指標。

可能的改進方向：
資料增強 (Data Augmentation)：對於圖片數量較少的專案，可以考慮在特徵提取之前對圖片進行資料增強，例如旋轉、翻轉、裁剪等，以增加訓練資料的多樣性，提高模型的泛化能力。

模型選擇與超參數調優 (Hyperparameter Tuning)：
除了線性 SVM，可以嘗試其他分類器，例如：邏輯迴歸 (Logistic Regression)、隨機森林 (Random Forest)、梯度提升樹 (Gradient Boosting Trees) (如 LightGBM 或 XGBoost) 或神經網路。
對於 SVM，可以嘗試不同的核函數（例如 rbf 高斯核）和 C 參數值，甚至對 test_size 等參數進行調整，以找到最佳的組合。可以使用 GridSearchCV 或 RandomizedSearchCV 進行系統性的超參數搜索。

