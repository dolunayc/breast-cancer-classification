# Breast Cancer Classification

This project implements a **Breast Cancer Classification** model using machine learning techniques. The dataset used is the **Breast Cancer Wisconsin Dataset**, which is available in the `sklearn.datasets` module.

## 📂 Project Structure
```
.
├── breast_cancer_classification.py  # Main script for classification
├── README.md                        # Project documentation
```

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/dolunayc/breast-cancer-classification.git
cd breast-cancer-classification
```

### 2️⃣ Install Dependencies
Make sure you have the required dependencies installed:
```sh
pip install numpy pandas scikit-learn
```

### 3️⃣ Run the Classification Script
```sh
python breast_cancer_classification.py
```

## 🏥 Dataset Information
- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Features:** 30 numerical attributes
- **Target:** Binary classification (0 = malignant, 1 = benign)
- **Objective:** Classify tumors as malignant or benign based on features.

## 🧠 Machine Learning Models Used
The script implements the following models:
1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Decision Trees (Optional for extension)**

## 📊 Model Evaluation Metrics
- **Accuracy Score**
- **Precision & Recall**
- **F1 Score**

## 📝 Notes
- The dataset is preprocessed using **StandardScaler** to normalize feature values.
- The script automatically splits data into **training (80%)** and **testing (20%)** sets.

