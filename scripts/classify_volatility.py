import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# -----------------------------
# 📁 Load All Stock CSVs
# -----------------------------
data_dir = '../data/'
stock_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
feature_list = []

for file in stock_files:
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path, on_bad_lines='skip')

    # ✅ Skip files without 'Close' column
    if 'Close' not in df.columns:
        print(f"⚠️ Skipping {file} — 'Close' column not found.")
        continue

    # Ensure 'Close' is numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)

    # Calculate daily return
    df['Return'] = df['Close'].pct_change(fill_method=None)
    df.dropna(inplace=True)

    # Calculate features
    volatility = np.std(df['Return'])
    avg_return = np.mean(df['Return'])
    sharpe_ratio = avg_return / (volatility + 1e-6)

    feature_list.append({
        'Ticker': df['Ticker'].iloc[0] if 'Ticker' in df.columns else file.replace('.csv', ''),
        'Volatility': volatility,
        'Avg_Return': avg_return,
        'Sharpe': sharpe_ratio
    })

# -----------------------------
# 📊 Convert to DataFrame
# -----------------------------
features_df = pd.DataFrame(feature_list)

# Label as Volatile if volatility > 0.025
features_df['Label'] = features_df['Volatility'].apply(lambda x: 1 if x > 0.025 else 0)

print("\n🔍 Feature Table:\n", features_df)

# -----------------------------
# 🧠 Train Random Forest Classifier
# -----------------------------
X = features_df[['Volatility', 'Avg_Return', 'Sharpe']]
y = features_df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Evaluate the Model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🌲 Random Forest Accuracy: {accuracy:.2f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Save Model & Features

joblib.dump(model, '../models/volatility_classifier.pkl')
features_df.to_csv('../data/stock_features.csv', index=False)

print("✅ Model saved to '../models/volatility_classifier.pkl'")
print("✅ Features saved to '../data/stock_features.csv'")
