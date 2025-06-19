import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------------------------------
# 1. Data Loading and Initial Exploration
# -----------------------------------------------------
# Load the dataset
# Make sure the 'Telco-Customer-Churn.csv' file is in the same directory
try:
    df = pd.read_csv('Telco-Customer-Churn.csv')
except FileNotFoundError:
    print("Dataset file not found. Please ensure 'Telco-Customer-Churn.csv' is in the correct directory.")
    exit()

# Create a copy for EDA to keep the original data intact
df_eda = df.copy()

print("--- Initial Data Overview ---")
print(df.head())

# --- Setup PDF for output ---
pdf_pages = PdfPages('churn_analysis_report.pdf')
print("\nGenerating PDF report: churn_analysis_report.pdf")


# -----------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# -----------------------------------------------------
print("\n--- Starting Exploratory Data Analysis ---")

# Set plot style
sns.set_style('whitegrid')

# Churn Distribution
fig = plt.figure(figsize=(7, 5))
sns.countplot(x='Churn', data=df_eda, palette='viridis')
plt.title('Churn Distribution (No vs. Yes)')
plt.xlabel('Churn')
plt.ylabel('Number of Customers')
pdf_pages.savefig(fig, bbox_inches='tight')
plt.close(fig)

# Churn by Contract Type
fig = plt.figure(figsize=(10, 6))
sns.countplot(x='Contract', hue='Churn', data=df_eda, palette='magma')
plt.title('Churn by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
pdf_pages.savefig(fig, bbox_inches='tight')
plt.close(fig)

# Churn by Payment Method
fig = plt.figure(figsize=(12, 6))
sns.countplot(x='PaymentMethod', hue='Churn', data=df_eda, palette='plasma')
plt.title('Churn by Payment Method')
plt.xlabel('Payment Method')
plt.xticks(rotation=45)
plt.ylabel('Number of Customers')
pdf_pages.savefig(fig, bbox_inches='tight')
plt.close(fig)

# Distribution of numerical features by Churn
numerical_features = ['tenure', 'MonthlyCharges']
for feature in numerical_features:
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(data=df_eda, x=feature, hue='Churn', multiple='stack', kde=True, palette='coolwarm')
    plt.title(f'Distribution of {feature} by Churn')
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------
# 3. Data Cleaning and Preprocessing
# -----------------------------------------------------
# The 'TotalCharges' column might have spaces, causing it to be an 'object' type.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop the customerID column as it's not a predictive feature
df.drop('customerID', axis=1, inplace=True)

# Handle missing values. For 'TotalCharges', we can impute with the median.
median_total_charges = df['TotalCharges'].median()
df['TotalCharges'].fillna(median_total_charges, inplace=True)

# Convert the target variable 'Churn' to binary (0/1)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Convert categorical features to numerical using one-hot encoding
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# -----------------------------------------------------
# 4. Feature Engineering
# -----------------------------------------------------
# Create tenure groups to categorize customer loyalty
bins = [0, 12, 24, 36, 48, 60, 72]
labels = ['0-1 year', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5-6 years']
df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=False)

# One-hot encode the new tenure_group feature
df = pd.get_dummies(df, columns=['tenure_group'], drop_first=True)


# -----------------------------------------------------
# 5. Model Training and Evaluation
# -----------------------------------------------------
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Logistic Regression ---
print("\n--- Training Logistic Regression Model ---")
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)

print("\nLogistic Regression Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_log_reg))

fig = plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_log_reg), annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
pdf_pages.savefig(fig, bbox_inches='tight')
plt.close(fig)

# --- Random Forest ---
print("\n--- Training Random Forest Model ---")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True, n_jobs=-1)
rf_clf.fit(X_train_scaled, y_train)
y_pred_rf = rf_clf.predict(X_test_scaled)

print("\nRandom Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Out-of-Bag Score: {rf_clf.oob_score_:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

fig = plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
pdf_pages.savefig(fig, bbox_inches='tight')
plt.close(fig)

# -----------------------------------------------------
# 6. Feature Importance Analysis (from Random Forest)
# -----------------------------------------------------
importances = rf_clf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

fig = plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), palette='viridis')
plt.title('Top 20 Feature Importances from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
pdf_pages.savefig(fig, bbox_inches='tight')
plt.close(fig)

# --- Close the PDF file ---
pdf_pages.close()
print("\nPDF report 'churn_analysis_report.pdf' has been successfully generated.")
