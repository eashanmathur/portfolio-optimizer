import pandas as pd
import joblib
import random

# Step 1: Load classified stock data (from previous step)
stock_features = pd.read_csv('../data/stock_features.csv')  # We'll save this file from Step 3 in a moment

# Step 2: Simulated user input
# In real app, this would come from the frontend form
user_profile = {
    'Age': 28,
    'Income': 45000,
    'Risk_Tolerance': 2,       # 0 = Low, 1 = Medium, 2 = High
    'Experience_Level': 1      # 0 = Newbie, 1 = Medium, 2 = Advanced
    'Investment_Goal': 2,       # 0 = Steady Income, 1 = Balanced, 2 = Growth
    'Investment_Horizon': 1     # 0 = Short, 1 = Mid, 2 = Long term
}

# Step 3: Load models
scaler = joblib.load('../models/user_scaler.pkl')
kmeans = joblib.load('../models/user_clustering_model.pkl')

# Convert user input into dataframe
user_df = pd.DataFrame([user_profile])
scaled_user = scaler.transform(user_df)

# Predict cluster
cluster = kmeans.predict(scaled_user)[0]

# Map cluster to profile
cluster_map = {0: 'Balanced', 1: 'Aggressive', 2: 'Conservative'}
profile_type = cluster_map.get(cluster, 'Balanced')

print(f"\n🧑‍💼 User Profile Type: {profile_type}")

# Step 4: Recommend stocks based on profile

# Split stocks
stable_stocks = stock_features[stock_features['Label'] == 0]
volatile_stocks = stock_features[stock_features['Label'] == 1]

# Rules
if profile_type == 'Conservative':
    selected_stocks = pd.concat([stable_stocks.sample(5), volatile_stocks.sample(1)])
    stable_pct = 85
    volatile_pct = 15

elif profile_type == 'Aggressive':
    selected_stocks = pd.concat([stable_stocks.sample(2), volatile_stocks.sample(4)])
    stable_pct = 30
    volatile_pct = 70

else:  # Balanced
    selected_stocks = pd.concat([stable_stocks.sample(3), volatile_stocks.sample(3)])
    stable_pct = 50
    volatile_pct = 50

# Step 5: Assign weights randomly within rules
selected_stocks = selected_stocks.copy()
selected_stocks['Weight (%)'] = 0

# Assign weights proportionally
if profile_type == 'Conservative':
    selected_stocks.loc[selected_stocks['Label'] == 0, 'Weight (%)'] = round(stable_pct / len(stable_stocks.sample(5)), 2)
    selected_stocks.loc[selected_stocks['Label'] == 1, 'Weight (%)'] = volatile_pct
elif profile_type == 'Aggressive':
    selected_stocks.loc[selected_stocks['Label'] == 0, 'Weight (%)'] = round(stable_pct / len(stable_stocks.sample(2)), 2)
    selected_stocks.loc[selected_stocks['Label'] == 1, 'Weight (%)'] = round(volatile_pct / len(volatile_stocks.sample(4)), 2)
else:
    selected_stocks['Weight (%)'] = round(100 / len(selected_stocks), 2)

# Show result
print("\n📈 Recommended Portfolio:\n")
print(selected_stocks[['Ticker', 'Volatility', 'Label', 'Weight (%)']])
