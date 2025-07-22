# train_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 讀取特徵檔
df = pd.read_csv("image_features.csv")

# 分離 X, y
X = df.drop(columns=["label"]).values
y = df["label"].values

# Label encode 類別名稱
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 分割訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 訓練 SVM 分類器
clf = SVC(kernel='linear', C=1.0)
clf.fit(X_train, y_train)

# 預測並評估
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_, labels=le.transform(le.classes_)))