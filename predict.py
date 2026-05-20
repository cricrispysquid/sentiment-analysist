import pickle
import json
import torch
import torch.nn as nn
import numpy as np

#加载训练好的模型，对新评论进行情感预测
# ------------------------------
# 模型定义（必须和 train.py 保持一致）
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
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# ------------------------------
# 加载模型
# ------------------------------
def load_model(model_path="model/"):
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)
    
    with open(f"{model_path}/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    model = FeedForwardNN(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"]
    )
    model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location="cpu"))
    model.eval()
    
    return model, vectorizer


# ------------------------------
# 预测单条
# ------------------------------
def predict(text, model, vectorizer):
    X = vectorizer.transform([text]).toarray()
    X_tensor = torch.FloatTensor(X)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    
    sentiment = "positive" if pred == 1 else "negative"
    return sentiment, confidence


# ------------------------------
# 批量预测
# ------------------------------
def predict_batch(texts, model, vectorizer):
    X = vectorizer.transform(texts).toarray()
    X_tensor = torch.FloatTensor(X)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1).numpy()
        confidences = probs[np.arange(len(preds)), preds].numpy()
    
    sentiments = ["positive" if p == 1 else "negative" for p in preds]
    return list(zip(sentiments, confidences))