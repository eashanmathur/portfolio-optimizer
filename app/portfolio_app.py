import os
import joblib
import pandas as pd
import streamlit as st


#  Load models and data

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

kmeans = joblib.load(os.path.join(BASE_DIR, 'models', 'user_clustering_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'user_scaler.pkl'))
return_model = joblib.load(os.path.join(BASE_DIR, 'models', 'projected_return_model.pkl'))

stock_data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'stock_features.csv'))


#  Streamlit UI Setup

st.set_page_config(page_title="Smart Portfolio Optimizer 🇮🇳", layout="wide")
st.title("💹 Smart Portfolio Optimizer")

st.markdown("""
This app recommends a personalized investment plan using 3 ML models:
- It classifies you as an investor type
- Picks a mix of stocks for your risk level
- Tells you how much to invest in each
- Predicts how much it may grow based on your selected investment duration
""")

st.header("📋 Tell Us About Yourself")


# User Input Form

with st.form("user_form"):
    income = st.number_input("Your Monthly Income (₹)", min_value=1000, value=30000, step=1000)
    amount = st.number_input("How much do you want to invest? (₹)", min_value=1000, value=10000, step=1000)

    risk = st.selectbox("How much risk are you okay with?", ["Low", "Medium", "High"])
    experience = st.selectbox("How experienced are you with investing?", ["Newbie", "Intermediate", "Advanced"])
    goal = st.selectbox("What is your main investment goal?", ["Wealth Growth", "Steady Income", "Balanced"])
    horizon = st.selectbox("How long do you plan to invest?", ["< 3 years", "3–5 years", "5+ years"])

    submit = st.form_submit_button("🎯 Generate My Investment Plan")


# ML Processing

if submit:
    # Step 1: Encode inputs
    risk_map = {"Low": 0, "Medium": 1, "High": 2}
    exp_map = {"Newbie": 0, "Intermediate": 1, "Advanced": 2}
    goal_map = {"Wealth Growth": 2, "Balanced": 1, "Steady Income": 0}
    horizon_map = {"< 3 years": 3, "3–5 years": 5, "5+ years": 7}

    user_input = pd.DataFrame([{
        "Age": 30,
        "Income": income,
        "Risk_Tolerance": risk_map[risk],
        "Experience_Level": exp_map[experience],
        "Investment_Goal": goal_map[goal],
        "Investment_Horizon": list(horizon_map.values())[list(horizon_map.keys()).index(horizon)]
    }])

    scaled_input = scaler.transform(user_input)
    cluster = kmeans.predict(scaled_input)[0]
    profile_map = {0: "Aggressive", 1: "Balanced", 2: "Conservative"}
    profile = profile_map.get(cluster, "Balanced")

    st.success(f"You are classified as a **{profile} Investor**")

    # Step 2: Stock selection
    stable = stock_data[stock_data['Label'] == 0]
    volatile = stock_data[stock_data['Label'] == 1]

    if profile == 'Conservative':
        selected = pd.concat([stable.sample(5), volatile.sample(1)])
    elif profile == 'Aggressive':
        selected = pd.concat([stable.sample(2), volatile.sample(4)])
    else:
        selected = pd.concat([stable.sample(3), volatile.sample(3)])

    selected = selected.reset_index(drop=True)

    # Step 3: Allocate based on Sharpe Ratio
    selected['Sharpe'] = selected['Sharpe'].apply(lambda x: x if x > 0 else 0)
    sharpe_total = selected['Sharpe'].sum()

    selected['Recommended % Allocation'] = (
        (selected['Sharpe'] / sharpe_total * 100).round(2)
        if sharpe_total > 0 else round(100 / len(selected), 2)
    )

    selected['Amount to Invest (₹)'] = (selected['Recommended % Allocation'] / 100 * amount).round(2)

    # Step 4: Predict future value
    selected['Is_Volatile'] = selected['Label']
    features = selected[['Volatility', 'Sharpe', 'Is_Volatile']]
    predictions = return_model.predict(features)

    years = horizon_map.get(horizon, 5)
    projected_column = f'Estimated Value in {years} Years (₹)'

    selected[projected_column] = (
        selected['Amount to Invest (₹)'] * (predictions / 10000) ** (years / 5)
    ).round(2)

    # Step 5: Rename for clarity
    selected['Stock Symbol'] = selected['Ticker']
    selected['Stock Type'] = selected['Label'].apply(lambda x: "Volatile" if x == 1 else "Stable")
    selected['Risk (0 = Low, 1 = High)'] = selected['Volatility']
    selected['Reward-to-Risk Ratio'] = selected['Sharpe']

    final = selected[[
        'Stock Symbol',
        'Stock Type',
        'Risk (0 = Low, 1 = High)',
        'Reward-to-Risk Ratio',
        'Recommended % Allocation',
        'Amount to Invest (₹)',
        projected_column
    ]]

    # Step 6: Display results
    st.subheader("📊 Your Personalized Investment Plan")
    st.dataframe(final)

    # Step 7: Total growth summary
    total_projected = final[projected_column].sum().round(2)
    total_invested = final['Amount to Invest (₹)'].sum().round(2)
    growth = ((total_projected - total_invested) / total_invested * 100) if total_invested > 0 else 0

    st.subheader("💰 Total Portfolio Growth")
    st.success(f"Projected Portfolio Value after {years} years: ₹{total_projected:,.2f}")
    st.info(f"Your investment of ₹{total_invested:,.2f} may grow by **{growth:.2f}%**.")

    # Step 8: Knowledge Center
    st.subheader("📘 Knowledge Center")
    st.markdown(f"""
**Stock Symbol**: Short name of the company (e.g., INFY, TCS)  
**Stock Type**: Stable = safer; Volatile = higher risk  
**Risk (0 = Low, 1 = High)**: Measures volatility  
**Reward-to-Risk Ratio**: Sharpe Ratio — higher is better  
**Recommended % Allocation**: Portion of your total ₹ to invest in this stock  
**Amount to Invest (₹)**: Actual ₹ suggested for this stock  
**{projected_column}**: Predicted return after {years} years using ML
    """)
