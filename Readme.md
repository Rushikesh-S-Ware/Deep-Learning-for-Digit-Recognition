# 🧠 MNIST CNN Image Classifier

This project is a digit recognition web application built using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The model is deployed using Gradio and hosted on [Hugging Face Spaces](https://huggingface.co/spaces/Rushikesh-S-Ware/Deep-Learning-for-Digit-Recognition), allowing users to upload an image and receive a predicted digit (0-9) in real-time.

## 🌐 Live Demo

👉 [Check out the live demo on Hugging Face Spaces](https://huggingface.co/spaces/Rushikesh-S-Ware/Deep-Learning-for-Digit-Recognition)

---

## 📁 Project Structure

| File Name                                 | Description                                                                 |
|------------------------------------------|-----------------------------------------------------------------------------|
| `MNIST_Digit_Classifier_Training.ipynb`      | Jupyter Notebook with data loading, model training, and evaluation         |
| `app.py`                                 | Gradio app script to load model and serve predictions from uploaded images |
| `digit_model.pth`                        | Trained CNN model checkpoint file                                          |
| `digit_classification_report.txt`       | Precision, recall, F1-score for each digit class                           |
| `confusion_matrix.png`                  | Confusion matrix visualized for test results                               |
| `training_loss_accuracy_curve.png`      | Training and validation loss/accuracy graphs                               |
| `requirements.txt`                      | Python dependencies required for the app                                   |
| `Final_Report_MNIST_Classification.pdf` | Final report of the project outlining methodology and results              |
| `Presentation_Model_Accuracy_MNIST.pptx`| Supporting PowerPoint presentation                                         |

---

## 📊 Project Overview

- **Dataset:** [MNIST handwritten digits](http://yann.lecun.com/exdb/mnist/)
- **Model Architecture:** CNN with 2 convolutional layers, ReLU activations, MaxPooling, and fully connected layers.
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam
- **Accuracy Achieved:** >99% on training set, >97% on test set
- **Deployment:** Hugging Face Spaces using Gradio interface
- **Use Case:** Upload your handwritten digit image (28x28 grayscale or larger) and get prediction

---

## 🚀 How to Use

1. Upload a clear image of a digit (0-9) using the web interface.
2. The model will process the image and return the predicted digit.
3. Try various styles of handwritten digits to test robustness.

---

## 📌 Highlights

- Custom pre-processing to clean and invert uploaded images
- Live hosted app—no setup required to test
- Full code for model training and deployment provided
- Clean, interpretable UI using Gradio

---

## 🛠 Setup Instructions (For Local Use)

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Rushikesh-S-Ware/Deep-Learning-for-Digit-Recognition.git
   cd Deep-Learning-for-Digit-Recognition
