# Heart Disease Prediction & Production Pipeline

An end-to-end machine learning project that leverages clinical patient data to predict the likelihood of heart disease. This project spans initial exploratory data analysis and multi-model benchmarking all the way to a robust, deployment-ready serialization pipeline.

---

## 📋 Project Overview & Medical Impact

Heart disease remains a leading cause of mortality globally. Early detection allows for preventive lifestyle changes, optimized healthcare resource allocation, and significant reductions in late-stage medical costs. 

This project explores multiple classification algorithms to determine the most effective predictive architecture, ultimately packaging the highest-performing ensemble model into a resilient inference pipeline.

---

## 📊 Dataset & Feature Breakdown

The clinical data is sourced from the [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset). It contains patient history and diagnostic measurements across the following features:

* **Numerical Fields:** `age`, `trestbps` (resting blood pressure), `chol` (serum cholesterol), `thalach` (maximum heart rate achieved), and `oldpeak` (ST depression).
* **Categorical Fields:** `sex`, `cp` (chest pain type), `fbs` (fasting blood sugar), `restecg` (resting ECG results), `exang` (exercise-induced angina), `slope` (slope of peak exercise ST segment), `ca` (number of major vessels colored by fluoroscopy), and `thal` (thalassemia type).

---

## ⚙️ Model Benchmarking & Performance Summary

During the experimental phase, multiple classification algorithms were trained and evaluated on testing data using accuracy, ROC-AUC, precision, and recall metrics:

* **Logistic Regression & XGBoost:** Strong foundational models showing solid baseline classification metrics.
* **K-Nearest Neighbors (KNN) & Support Vector Machines (SVM):** Underperformed relative to tree-based architectures, indicating a need for heavy hyperparameter tuning.
* **Decision Trees, Random Forest, & Gradient Boosting:** Top-tier performers. While the raw Decision Tree showed exceptional training accuracy, the **Random Forest Classifier** was selected for production due to its superior generalization capabilities and resistance to overfitting.

---

## 🏆 Model Performance & Results Validation

The final production pipeline was evaluated using an end-to-end validation test suite. The serialized `Random Forest` architecture achieved highly stable, production-grade metrics across both patient target classes:

### 📈 Core Performance Metrics
* **Overall Evaluation Accuracy:** **92.0%** (Highly reliable generalization with zero overfitting)
* **Receiver Operating Characteristic (ROC-AUC):** **0.954** (Excellent class separation power)

### 📋 Detailed Classification Report
| Target Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Healthy (0)** | 0.90 | 0.92 | 0.91 | 50 |
| **Heart Disease (1)** | 0.93 | 0.90 | 0.92 | 60 |
| **Macro Average** | 0.92 | 0.91 | 0.91 | 110 |

### 🌲 Top Feature Drivers (Gini Importance)
The model's decision boundaries are heavily driven by the following top clinical factors, matching real-world medical significance:
1. `ca` (Number of major vessels colored by fluoroscopy) — **Highest Predictive Weight**
2. `thalach` (Maximum heart rate achieved)
3. `oldpeak` (ST depression induced by exercise)
4. `cp` (Chest pain type mappings)

---

## 🚀 Production Pipeline Architecture

To transition from raw model weights to a reliable production asset, an inference pipeline was built to eliminate live-data crashes:

1. **Resilient Alignment Layer:** Implements a strict schema-mapping reindexing method. If a live API payload or application form lacks specific categorical states, it dynamically handles the missing features instead of throwing a shape mismatch error.
2. **High-Performance Serialization:** Uses `joblib` to decouple and save the scaling weights (`StandardScaler`) from the core ensemble model (`Random Forest`), preserving exact feature constraints.
3. **Verification Hub:** Concludes with an automated dashboard that evaluates live binary assets directly from disk, generating a clean Confusion Matrix, an ROC-AUC curve, and real-time Gini feature importances.

---

## 📁 Repository Blueprint

* `Heart_Disease_Predictive_Pipeline_End_to_End.ipynb` - Core Jupyter/Colab notebook containing data exploration, model comparison, training, and the final analytics dashboard.
* `heart_disease_classifier_rf.pkl` - Serialized, production-validated Random Forest model weights.
* `heart_disease_standard_scaler.pkl` - Serialized preprocessing weights mapping the 5 key numerical dimensions.

---

## 🛠️ Setup & Quick Start

### 1. Environment Installation
Ensure you have Python installed, then clone the repo and install the core dependencies:
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
pip install pandas numpy scikit-learn joblib matplotlib seaborn
