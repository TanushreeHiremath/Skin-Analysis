# Skin-Analysis
AI-powered skin analyzer that detects acne, dark spots, wrinkles, puffy eyes, and normal skin using deep learning models (CNN, DenseNet121, EfficientNetB0). Includes a Tkinter GUI for image upload, prediction, and personalized skincare treatment suggestions.

# 🧴 Skin Condition Analyzer with Treatment Suggestions

A deep learning-based application to analyze facial skin conditions and recommend personalized skincare treatments. This project uses image classification models (CNN, DenseNet121, EfficientNetB0) to detect common skin issues like acne, dark spots, puffy eyes, wrinkles, and normal skin. It also features a user-friendly Tkinter GUI to help users upload images and receive condition-specific suggestions.

---

## 📁 Dataset

The dataset contains labeled facial skin images belonging to the following categories:

- `acne`
- `dark spots`
- `normal skin`
- `puffy eyes`
- `wrinkles`

Images are preprocessed by resizing them to 128x128 pixels and normalizing pixel values to the `[0, 1]` range.

---

## 📌 Features

- Trained deep learning models: CNN, DenseNet121, EfficientNetB0
- Real-time image prediction using a GUI interface
- Tailored treatment recommendations for each skin condition
- Confusion matrix and classification reports for evaluation
- Model saving and reloading functionality

---

## 🧠 Model Architectures

### 1. 📦 CNN Model
- 3 convolutional layers with ReLU and max pooling
- Fully connected dense layers
- Dropout for regularization

Saved as: `cnn_model.keras`

### 2. 🌿 DenseNet121
- Pretrained DenseNet121 (ImageNet) as feature extractor
- Global average pooling and fully connected layers
- Fine-tuned with frozen base layers

Saved as: `densenet_model.keras`

### 3. ⚡ EfficientNetB0
- Pretrained EfficientNetB0 backbone
- Custom top layers for classification
- Fine-tuned for 5 skin condition classes

Saved as: `efficientnet_model.keras`

---

## 🎯 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**
- **Classification Report**

These metrics are displayed after model training and testing on a separate validation set.

---

## 🧪 Training

Models are trained with:

- Image size: 128×128
- Epochs: 10
- Batch size: 32
- Optimizer: Adam
- Loss Function: Categorical Crossentropy

---

## 🖥️ GUI Interface

The GUI (Tkinter) allows users to:
- Upload a face image
- View a preview
- Get real-time prediction
- See recommended treatments based on skin condition

Example treatments for conditions like **acne**, **wrinkles**, etc., include skincare routines, product types, and lifestyle suggestions.

Run with:
python predict_skin_condition.py

## Setup

1. Clone the repo
git clone https://github.com/your-username/skin-condition-analyzer.git
cd skin-condition-analyzer

2. Install dependencies
pip install -r requirements.txt
<details> <summary><strong>requirements.txt (example)</strong></summary>
tensorflow
opencv-python
numpy
matplotlib
seaborn
pillow
scikit-learn
</details>

3. Prepare Dataset
Organize images in this format:
DATASET/
├── acne/
├── dark spots/
├── normal skin/
├── puffy eyes/
└── wrinkles/
Update data_dir in each script if needed.

4. Train Models (optional)
python cnn_model.py
python densenet_model.py
python efficientnet_model.py

## 🛠 File Structure

├── cnn_model.py              # CNN training and evaluation
├── densenet_model.py         # DenseNet121 training and evaluation
├── efficientnet_model.py     # EfficientNetB0 training and evaluation
├── predict_skin_condition.py # Tkinter GUI app
├── README.md                 # Project documentation

## 💡 Future Improvements

Deploy as a web app with Flask or Streamlit
Integrate real-time webcam capture
Add more skin conditions
Collect a larger, more diverse dataset
