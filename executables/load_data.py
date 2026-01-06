import yfinance as yf
import pandas as pd
import os

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
tickers_csv = os.path.join(script_dir, "../data/tickers.csv")
output_dir = os.path.join(script_dir, "../data/yfinance")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load tickers from centralized CSV
tickers_df = pd.read_csv(tickers_csv)
tickers = dict(zip(tickers_df['yfinance_id'], tickers_df['name']))

# Download and save each ticker
print(f"Downloading {len(tickers)} tickers...\n")

for ticker, filename in tickers.items():
    try:
        print(f"Downloading {ticker}...", end=" ")
        data = yf.download(ticker, period="max", progress=False, auto_adjust=True)

        if data.empty:
            print("NO DATA")
            continue

        filepath = f"{output_dir}/{filename}.csv"
        data.to_csv(filepath)
        print(f"OK ({len(data)} rows)")

    except Exception as e:
        print(f"ERROR: {e}")

print("\nDone!")
