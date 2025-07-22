# extract_features.py

import os
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# 載入模型與處理器
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

# 讀取標籤檔
df = pd.read_csv("labels.csv")

features = []
labels = []

# 遍歷圖片資料夾
for idx, row in df.iterrows():
    path = os.path.join("images", row["filename"])
    label = row["label"]

    try:
        image = Image.open(path).convert("RGB")
    except:
        continue

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        image_embed = outputs[0].cpu().numpy()  # shape: [512]

    features.append(image_embed)
    labels.append(label)

# 儲存成 CSV
import numpy as np
features = np.vstack(features)

df_out = pd.DataFrame(features)
df_out["label"] = labels
df_out.to_csv("image_features.csv", index=False)
print("✅ 特徵提取完成，已儲存 image_features.csv")

