import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score  # ✅ Added for evaluation
import joblib
import random

# -----------------------------
# 🔄 Generate Synthetic Users
# -----------------------------
def generate_user():
    return {
        "Age": random.randint(20, 60),
        "Income": random.randint(10000, 150000),
        "Risk_Tolerance": random.choice([0, 1, 2]),              # 0 = Low, 1 = Medium, 2 = High
        "Experience_Level": random.choice([0, 1, 2]),            # 0 = Newbie, 1 = Intermediate, 2 = Advanced
        "Investment_Goal": random.choice([0, 1, 2]),             # 0 = Steady Income, 1 = Balanced, 2 = Growth
        "Investment_Horizon": random.choice([0, 1, 2])           # 0 = Short, 1 = Mid, 2 = Long term
    }

users = [generate_user() for _ in range(200)]
df = pd.DataFrame(users)

# -----------------------------
# 🔢 Normalize for KMeans
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# -----------------------------
# 🧠 Train KMeans
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# -----------------------------
# 📈 Evaluate KMeans using Silhouette Score
# -----------------------------
sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f"\n🧠 Silhouette Score (K-Means Clustering): {sil_score:.2f}")

# -----------------------------
# 💾 Save model + scaler
# -----------------------------
joblib.dump(kmeans, '../models/user_clustering_model.pkl')
joblib.dump(scaler, '../models/user_scaler.pkl')

print("✅ KMeans model & scaler saved.")
