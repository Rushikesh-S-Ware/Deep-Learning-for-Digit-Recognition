"""Gradio app for MNIST CNN digit classifier.

Loads the trained CNN checkpoint and exposes a simple web interface where users
upload a handwritten-digit image and receive a predicted digit (0-9).

Run locally:
    pip install -r requirements.txt
    python app.py

Then open http://localhost:7860.
"""

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms


MODEL_PATH = "digit_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DigitCNN(nn.Module):
    """CNN with 2 conv layers + 2 FC layers, trained on MNIST."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def load_model():
    model = DigitCNN()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Warning: {MODEL_PATH} not found. Using untrained weights.")
    model.eval().to(DEVICE)
    return model


MODEL = load_model()

PREPROCESS = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def predict(image: Image.Image) -> str:
    """Predict a digit (0-9) from an uploaded image."""
    if image is None:
        return "Please upload an image."
    # MNIST is white digits on black background; invert if needed.
    gray = image.convert("L")
    if np.array(gray).mean() > 127:
        gray = ImageOps.invert(gray)
    tensor = PREPROCESS(gray).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(tensor)
        probs = F.softmax(logits, dim=1)[0]
        digit = int(torch.argmax(probs).item())
        confidence = float(probs[digit].item())
    return f"Predicted digit: {digit}  (confidence {confidence:.1%})"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a digit image"),
    outputs=gr.Text(label="Prediction"),
    title="MNIST CNN Digit Classifier",
    description="Upload a handwritten digit (0-9). Reaches >97% test accuracy on MNIST.",
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch()
