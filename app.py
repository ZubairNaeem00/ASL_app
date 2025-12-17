from flask import Flask, render_template, request, jsonify
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import base64

# --------------------------
# CONFIG
# --------------------------
device = torch.device("cpu")
IMG_SIZE = (32, 32)

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

# --------------------------
# MODEL
# --------------------------
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = CNN(len(labels)).to(device)
model.load_state_dict(torch.load("asl_cnn_model.pth", map_location=device))
model.eval()

# --------------------------
# TRANSFORM
# --------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# --------------------------
# FLASK
# --------------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]
    image_bytes = base64.b64decode(data.split(",")[1])

    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    img = cv2.resize(img, IMG_SIZE)
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    return jsonify({
        "prediction": labels[pred.item()],
        "confidence": round(conf.item(), 2)
    })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
