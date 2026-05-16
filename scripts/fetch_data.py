import yfinance as yf
import pandas as pd
import os

# List of stock tickers (can be modified)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'JPM', 'UNH', 'TCS.NS']

# Date range
start_date = '2022-01-01'
end_date = '2024-01-01'

# Output directory
output_dir = '../data/'
os.makedirs(output_dir, exist_ok=True)

# Fetch and save data
for ticker in tickers:
    print(f"Fetching {ticker}...")
    stock = yf.download(ticker, start=start_date, end=end_date)
    stock['Ticker'] = ticker  # Add a column for ticker name
    stock.to_csv(os.path.join(output_dir, f"{ticker}.csv"))

print("✅ All stock data fetched and saved.")
