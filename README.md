# 📉 Customer Churn Prediction and Automated Reporting

## 📝 Description

This project predicts **customer churn** using machine learning models such as **Logistic Regression** and **Random Forest**. It performs thorough **Exploratory Data Analysis (EDA)**, **feature engineering**, model training, and evaluation. A key highlight is the **automated generation of a comprehensive PDF report** that includes all visualizations, performance metrics, and insights.

---

## 🧰 Tech Stack

- **Data Extraction & Preparation:** SQL
- **Data Analysis & Feature Engineering:** Python (`pandas`, `numpy`)
- **Machine Learning Models:** Scikit-learn (`Logistic Regression`, `Random Forest`)
- **Visualization & Reporting:** `matplotlib`, `seaborn`, `matplotlib.backends.backend_pdf.PdfPages`

---

## 🚀 Project Workflow

### 🔍 1. Exploratory Data Analysis (EDA)

Initial data analysis includes:
- Churn distribution
- Churn rate by contract type and payment method
- Distribution of **tenure** and **monthly charges** for churned vs. retained customers

### 🧼 2. Data Preprocessing & Feature Engineering

- **Cleaning:** Handles missing values (e.g., in `TotalCharges`), corrects data types.
- **Encoding:** Converts categorical variables using **one-hot encoding**.
- **Feature Creation:** Adds meaningful features like **tenure bins** (e.g., `'0-1 years'`, `'1-2 years'`) to capture customer loyalty.

### 🤖 3. Model Training and Evaluation

- **Train/Test Split & Scaling:** Uses `StandardScaler` for stable performance.
- **Models Implemented:**
  - **Logistic Regression:** A robust baseline model
  - **Random Forest Classifier:** Higher accuracy and interpretable feature importances
- **Performance Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix

### 📄 4. Automated PDF Reporting

All key plots and evaluation results are saved to:
