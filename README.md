# Smart Portfolio Optimizer 

A user-friendly Machine Learning project that helps you create a **personalized investment plan** based on your risk tolerance, income, and experience level.

Built with:
- Python 
- Streamlit 
- scikit-learn 

---

## What This App Does

1. Predicts your **risk profile** (Conservative / Balanced / Aggressive)
2. Selects a **mix of stable and volatile stocks**
3. Recommends how much ₹ to invest in each stock
4. Predicts how much each stock may grow in **5 years**
5. Calculates total **portfolio value** and **% growth**

---

## Machine Learning Used

| Purpose | ML Model |
|--------|----------|
| Classify stock type (Stable / Volatile) | Random Forest |
| Cluster user into investor profile | KMeans |
| Predict future stock value | Linear Regression |

---

## Technologies Used

- Python
- pandas, scikit-learn
- Streamlit (for the web app UI)
- joblib (for saving/loading models)

---

## How to Run the App

```bash
# 1. Activate virtual environment
cd portfolio-optimizer
source venv/bin/activate

# 2. Launch the app
cd app
streamlit run portfolio_app.py

#
cd ~/Desktop/portfolio-optimizer/app
source ../venv/bin/activate
streamlit run portfolio_app.py

#how to run evaluation model
cd ~/Desktop/portfolio-optimizer/scripts
python3 evaluate_models.py
