import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os

# -----------------------------
# 📁 Load Features Dataset
# -----------------------------
data_path = '../data/stock_features.csv'
df = pd.read_csv(data_path)

# -----------------------------
# 🧪 Feature Engineering
# -----------------------------
# Label: 1 if Volatile, 0 if Stable
df['Is_Volatile'] = df['Label'].apply(lambda x: 1 if x == 1 else 0)

# Simulate target growth (in % over 5 years)
# ❗ Replace this with real return data if available
np.random.seed(42)
df['ProjectedGrowth'] = df['Sharpe'] * 150 + df['Volatility'] * 400 + df['Is_Volatile'] * 100 + np.random.normal(0, 20, len(df))

# -----------------------------
# 🔍 Features and Target
# -----------------------------
X = df[['Volatility', 'Sharpe', 'Is_Volatile']]
y = df['ProjectedGrowth']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 🧠 Train Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 📈 Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📈 Linear Regression R² Score: {r2:.2f}")
print(f"📉 RMSE: {rmse:.2f}")
print(f"📊 MAE: {mae:.2f}")

# -----------------------------
# 💾 Save Model
# -----------------------------
model_path = '../models/projected_return_model.pkl'
joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")
