# Transfer Learning for Multi-Class Waste Classification

A Streamlit-powered interactive application showcasing a **full deep learning pipeline for waste classification** using **transfer learning**, data augmentation, and four state-of-the-art pre-trained models.



## Project Overview

This project illustrates a real-world image classification pipeline for automated waste segregation:
- **Dataset:** Custom curated dataset with 9 waste categories.
```python
waste_classes = ["Cardboard", "Food Organics", "Glass", "Metal", "Miscellaneous Trash", "Paper", "Plastic", "Textile Trash", "Vegetation"]
``` 
- **Pre-trained Models:**  
  - EfficientNetB0  
  - ResNet-50  
  - ResNet-101  
  - VGG-16
- **Techniques Applied:**  
  - Convolutional Neural Networks & Transfer Learning from ImageNet  
  - Custom classification head with dense layers, regularization, dropout  
  - Extensive on-the-fly data augmentation  
  - Hyper-parameter Tuning with Training Callbacks
  - Mixed precision training with Apple MPS and NVIDIA GPUs  
  - Macro and Weighted Evaluations  


## 🧭 Streamlit App Structure

The app is organized into multiple interactive sections:
- 🏠 **Home / Landing Page:** Introduction, tools used, highlights.
- 🔎 **Overview:** Dataset, split details, objectives.
- ⚙️ **Model & Training Details:** Architectures, strategy, hyperparameters, environment specs.
- 📊 **Performance & Results:** Metrics, classification reports, AUC radar charts, comparison table.
- 🎮 **Playground:** Upload your own image for inference and compare models.


## 🖥️ Tech Stack

- Python 3.11.11
- TensorFlow 2.18
- scikit-learn 1.2.2
- Streamlit 1.46.1
- Plotly 6.2.0


## 📂 Project Structure
```
.
├── pages/          # Modular Streamlit pages
├── models/         # Saved .keras models
├── data/           # Helpers for model metadata, metrics, architecture
├── utils/          # Utility functions (e.g., radar plot, model loader)
├── assets/         # Static files 
├── Home.py         # Landing page (entry point)
└── README.md       # This file
```


## 🛠️ Setup and Run Locally

- **Clone the repository:**
```bash
git clone <repo-url>
cd <repo-dir>
```

- Install dependencies:
```bash
pip install -r requirements.txt
```
- Launch the app:
```bash
streamlit run Home.py
```