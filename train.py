import os
import json
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

# ------------------------------
# 1. 路径准备
# ------------------------------
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

DATA_PATH = "data/imdb_balanced_10k.csv"

# ------------------------------
# 2. 加载数据（请确保文件真实存在）
# ------------------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH)

# 常见列名兼容处理
text_col = "review" if "review" in df.columns else "text"
label_col = "sentiment" if "sentiment" in df.columns else "label"

X = df[text_col].astype(str).values
y_raw = df[label_col].values

# 标签转 0/1
if y_raw.dtype == object:
    y = np.array([1 if str(s).lower() in ["positive", "pos"] else 0 for s in y_raw])
else:
    y = y_raw.astype(int)

# ------------------------------
# 3. TF‑IDF 向量化
# ------------------------------
print("TF-IDF vectorizing...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_vec = vectorizer.fit_transform(X).toarray()

# ------------------------------
# 4. 训练 / 验证 分割
# ------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)

# ------------------------------
# 5. 神经网络定义
# ------------------------------
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev = h
        layers.append(nn.Linear(prev, 2))  # 二分类
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ------------------------------
# 6. 训练配置
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FeedForwardNN(input_dim=X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train_t = X_train_t.to(device)
y_train_t = y_train_t.to(device)
X_val_t = X_val_t.to(device)
y_val_t = y_val_t.to(device)

# ------------------------------
# 7. 训练循环
# ------------------------------
EPOCHS = 20
BATCH_SIZE = 128

print("Training Neural Network...")
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X_train_t))
    total_loss = 0.0

    for i in range(0, len(X_train_t), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        batch_x = X_train_t[idx]
        batch_y = y_train_t[idx]

        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 验证
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_preds = torch.argmax(val_logits, dim=1)
        acc = accuracy_score(y_val_t.cpu(), val_preds.cpu())
        f1 = f1_score(y_val_t.cpu(), val_preds.cpu())

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Acc: {acc:.4f} | F1: {f1:.4f}")

# ------------------------------
# 8. 保存所有产物
# ------------------------------
print("Saving artifacts...")

# 模型权重
torch.save(model.state_dict(), "model/model.pt")

# TF‑IDF 向量器
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# 配置
config = {
    "model_type": "TFIDF_FeedForward_NN",
    "input_dim": X_train.shape[1],
    "hidden_dims": [512, 256],
    "max_features": 5000,
    "stop_words": "english",
}
with open("model/config.json", "w") as f:
    json.dump(config, f, indent=2)

# 评估指标
with torch.no_grad():
    final_preds = torch.argmax(model(X_val_t), dim=1).cpu().numpy()
metrics = {
    "val_accuracy": float(accuracy_score(y_val, final_preds)),
    "val_f1": float(f1_score(y_val, final_preds)),
}
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ 训练完成")
print(f"Accuracy : {metrics['val_accuracy']:.4f}")