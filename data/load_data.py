import yfinance as yf
import os

# Create output directory
output_dir = "./data/yfinance"
os.makedirs(output_dir, exist_ok=True)

tickers = {
    # Big Tech - Major technology companies driving market trends
    "AAPL": "apple",
    "MSFT": "microsoft",
    "GOOGL": "alphabet",
    "NVDA": "nvidia",
    "META": "meta",
    
    # Finance - Banks, payment processors, and financial conglomerates
    "JPM": "jpmorgan",
    "V": "visa",
    "BRK-B": "berkshire_hathaway",
    
    # Healthcare - Insurance, pharmaceuticals, and medical devices
    "UNH": "unitedhealth",
    "JNJ": "johnson_and_johnson",
    "PFE": "pfizer",
    
    # Consumer Discretionary - E-commerce, automotive, restaurants
    "AMZN": "amazon",
    "TSLA": "tesla",
    "MCD": "mcdonalds",
    
    # Consumer Staples - Retail, beverages, household products
    "WMT": "walmart",
    "KO": "coca_cola",
    "PG": "procter_gamble",
    
    # Energy - Oil and gas majors
    "XOM": "exxonmobil",
    "CVX": "chevron",
    
    # Industrials - Heavy machinery, railroads, aerospace
    "CAT": "caterpillar",
    "UNP": "union_pacific",
    "BA": "boeing",
    
    # Utilities - Power generation and distribution
    "NEE": "nextera_energy",
    "DUK": "duke_energy",
    
    # Real Estate - REITs covering logistics and telecom infrastructure
    "PLD": "prologis",
    "AMT": "american_tower",
    
    # Materials - Industrial gases and mining
    "LIN": "linde",
    "FCX": "freeport_mcmoran",
    
    # Telecom - Wireless and communications services
    "VZ": "verizon",
    "TMUS": "tmobile",
    
    # International - Major non-US companies (ADRs)
    "TSM": "taiwan_semiconductor",
    "ASML": "asml",
    "TM": "toyota",
    "BABA": "alibaba",
    
    # Crypto - Major cryptocurrencies
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "SOL-USD": "solana",
    
    # Market Indices - Broad market benchmarks
    "^GSPC": "sp500",
    "^NDX": "nasdaq100",
    "^DJI": "dow_jones",
    "^RUT": "russell2000",
}

# Download and save each ticker
print(f"Downloading {len(tickers)} tickers...\n")

for ticker, filename in tickers.items():
    try:
        print(f"Downloading {ticker}...", end=" ")
        data = yf.download(ticker, period="max", progress=False)
        
        if data.empty:
            print("NO DATA")
            continue
        
        filepath = f"{output_dir}/{filename}.csv"
        data.to_csv(filepath)
        print(f"OK ({len(data)} rows)")
        
    except Exception as e:
        print(f"ERROR: {e}")

print("\nDone!")
