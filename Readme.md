# MNIST CNN Image Classifier

> A digit-recognition web app built on a Convolutional Neural Network (CNN) trained on MNIST. Live Gradio demo on Hugging Face Spaces.

[![Live Demo](https://img.shields.io/badge/HuggingFace-Live%20Demo-yellow?logo=huggingface)](https://huggingface.co/spaces/Rushikesh-S-Ware/Deep-Learning-for-Digit-Recognition)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Live Demo

👉 https://huggingface.co/spaces/Rushikesh-S-Ware/Deep-Learning-for-Digit-Recognition

Upload a handwritten digit image; the model returns the predicted digit (0–9) in real time.

## Project Overview

- **Dataset:** MNIST handwritten digits
- **Architecture:** CNN — 2 convolutional layers, ReLU, MaxPooling, fully connected head
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam
- **Accuracy:** > 99% training, > 97% test
- **Deployment:** Hugging Face Spaces (Gradio)

## Repository Layout

| File | Purpose |
|---|---|
| `MNIST_Digit_Classifier_Training.ipynb` | Data loading, model training, evaluation |
| `app.py` | Gradio app — loads model, serves predictions |
| `digit_model.pth` | Trained CNN checkpoint |
| `digit_classification_report.txt` | Per-class precision/recall/F1 |
| `confusion_matrix.png` | Confusion matrix on test set |
| `training_loss_accuracy_curve.png` | Training curves |
| `requirements.txt` | Python dependencies |
| `Final_Report_MNIST_Classification.pdf` | Full report |
| `Presentation_Model_Accuracy_MNIST.pptx` | Slides |

## Highlights

- Custom preprocessing to clean and invert uploaded images
- Live hosted app — no setup required to try it
- Full training + deployment code included
- Clean, interpretable Gradio UI

## Run Locally

```bash
git clone https://github.com/Rushikesh-S-Ware/Deep-Learning-for-Digit-Recognition
cd Deep-Learning-for-Digit-Recognition
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:7860` and upload a digit image.

## License

MIT
