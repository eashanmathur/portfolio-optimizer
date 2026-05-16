
# evaluate_models.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

print("📊 MODEL EVALUATION REPORT")

# -----------------------------
# 1. K-Means Clustering
# -----------------------------
print("\n=== K-MEANS CLUSTERING ===")
scaler = joblib.load('../models/user_scaler.pkl')
kmeans = joblib.load('../models/user_clustering_model.pkl')

def generate_user():
    return {
        "Age": np.random.randint(20, 60),
        "Income": np.random.randint(10000, 150000),
        "Risk_Tolerance": np.random.choice([0, 1, 2]),
        "Experience_Level": np.random.choice([0, 1, 2]),
        "Investment_Goal": np.random.choice([0, 1, 2]),
        "Investment_Horizon": np.random.choice([0, 1, 2])
    }

users = [generate_user() for _ in range(200)]
df_users = pd.DataFrame(users)
X_scaled = scaler.transform(df_users)
score = silhouette_score(X_scaled, kmeans.predict(X_scaled))
print(f"🧠 Silhouette Score: {score:.2f}")

# -----------------------------
# 2. Random Forest Classifier
# -----------------------------
print("\n=== RANDOM FOREST CLASSIFIER ===")
df_stock = pd.read_csv('../data/stock_features.csv')
X = df_stock[['Volatility', 'Avg_Return', 'Sharpe']]
y = df_stock['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_model = joblib.load('../models/volatility_classifier.pkl')
y_pred_rf = rf_model.predict(X_test)

print(f"🌲 Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print("📋 Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# -----------------------------
# 3. Linear Regression
# -----------------------------
print("\n=== LINEAR REGRESSION ===")
# Simulate ProjectedGrowth if missing
df_stock['Is_Volatile'] = df_stock['Label'].apply(lambda x: 1 if x == 1 else 0)
np.random.seed(42)
df_stock['ProjectedGrowth'] = df_stock['Sharpe'] * 150 + df_stock['Volatility'] * 400 + df_stock['Is_Volatile'] * 100 + np.random.normal(0, 20, len(df_stock))

X_lr = df_stock[['Volatility', 'Sharpe', 'Is_Volatile']]
y_lr = df_stock['ProjectedGrowth']
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)

lr_model = joblib.load('../models/projected_return_model.pkl')
y_pred_lr = lr_model.predict(X_test_lr)

print(f"📈 R² Score: {r2_score(y_test_lr, y_pred_lr):.2f}")
print(f"📉 RMSE: {mean_squared_error(y_test_lr, y_pred_lr, squared=False):.2f}")
print(f"📊 MAE: {mean_absolute_error(y_test_lr, y_pred_lr):.2f}")
