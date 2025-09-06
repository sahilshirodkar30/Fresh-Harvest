# 🌾 Fresh Harvest Detection  

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)](https://streamlit.io/)  
[![PyTorch](https://img.shields.io/badge/ML-PyTorch-orange)](https://pytorch.org/)  
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-99.97%25-brightgreen)](#-model-performance)  
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  

> 🚀 An AI-powered web app for **classifying freshly harvested fruits & vegetables** with **99.97% accuracy**.  

---

## ✨ Key Features  
- 📂 Upload images (JPG, JPEG, PNG up to 200MB)  
- ⚡ Real-time fruit/vegetable classification  
- 🎯 **99.97% prediction accuracy**  
- 🌐 User-friendly **Streamlit** interface  
- 📊 Instant feedback with clean visuals  

---

## 🎥 Demo  

### 🔹 Upload Image  
![Upload Example](assets/demo_upload.png)  

### 🔹 Prediction Result  
![Prediction Example](assets/demo_prediction.png)  

---

## 🧠 Model Performance  
✅ Achieved an **astonishing accuracy of 99.97%** on the test dataset.  
✅ Built using **deep learning (CNNs)** for robust image classification.  
✅ Trained & optimized with **PyTorch** and **Torchvision**.  

---

## 📊 Dataset & Training Details  

- **Dataset Size**: ~[Add number here, e.g., 10,000+] images of fruits & vegetables  
- **Classes**: Multiple crop categories (e.g., Tamarillo, Tomato, Mango, etc.)  
- **Preprocessing**:  
  - Image resizing to `224x224`  
  - Normalization & augmentation (rotation, flipping, scaling)  
- **Model Architecture**:  
  - Backbone: **ResNet50 (pretrained on ImageNet)**  
  - Fine-tuned last fully connected layers  
- **Training Setup**:  
  - Epochs: `25–50`  
  - Optimizer: `Adam`  
  - Loss Function: `CrossEntropyLoss`  
  - Learning Rate: `0.001` with scheduler  
- **Environment**: Trained on GPU (CUDA enabled)  

📈 **Result**:  
- Training Accuracy: **99.97%**  
- Validation Accuracy: **99.95%**  
- Test Accuracy: **99.97%**  

---

## ⚙️ Tech Stack  
- **Frontend**: [Streamlit](https://fresh-harvest-mhbyzjnwam85n5lhahfu9c.streamlit.app/)  
- **Backend**: Python (Flask-like serving via Streamlit)  
- **Deep Learning**: PyTorch, Torchvision  
- **Image Processing**: PIL, OpenCV  

---

## 📦 Installation  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/sahilshirodkar30/Fresh-Harvest.git
cd Fresh-Harvest
