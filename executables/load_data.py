import yfinance as yf
import os

# Create output directory
output_dir = "../data/yfinance"
os.makedirs(output_dir, exist_ok=True)

tickers = {
    # --- TECHNOLOGY & AI ---
    # Big Tech - The "Magnificent" drivers of the SP500
    "AAPL": "apple", "MSFT": "microsoft", "GOOGL": "alphabet", 
    "NVDA": "nvidia", "META": "meta", "AMZN": "amazon", "TSLA": "tesla",
    
    # Semiconductors & Hardware - The AI backbone
    "AMD": "amd", "AVGO": "broadcom", "ARM": "arm_holdings",
    "MU": "micron", "SMCI": "super_micro_computer", "INTC": "intel",
    "QCOM": "qualcomm", "AMAT": "applied_materials",
    
    # Cloud, Software & SaaS
    "CRM": "salesforce", "ORCL": "oracle", "NOW": "servicenow", 
    "PLTR": "palantir", "SNOW": "snowflake", "ADBE": "adobe",
    
    # Cybersecurity - Mission-critical enterprise tech
    "PANW": "palo_alto_networks", "CRWD": "crowdstrike", 
    "FTNT": "fortinet", "ZS": "zscaler", "OKTA": "okta",
    
    # --- FINANCE & FINTECH ---
    # Banking & Conglomerates
    "JPM": "jpmorgan", "BAC": "bank_of_america", "GS": "goldman_sachs", 
    "MS": "morgan_stanley", "BRK-B": "berkshire_hathaway",
    
    # Payments & Fintech
    "V": "visa", "MA": "mastercard", "PYPL": "paypal", 
    "SQ": "block", "COIN": "coinbase",
    
    # --- HEALTHCARE & BIOTECH ---
    # Weight Loss & Longevity (The 2025 GLP-1 Boom)
    "LLY": "eli_lilly", "NVO": "novo_nordisk", 
    
    # Pharmaceuticals & Insurance
    "UNH": "unitedhealth", "JNJ": "johnson_and_johnson", 
    "PFE": "pfizer", "ABBV": "abbvie", "AMGN": "amgen",
    
    # Genomics & Innovation
    "ILMN": "illumina", "CRSP": "crispr_therapeutics", "VRTX": "vertex",
    
    # --- CONSUMER SECTORS ---
    # Consumer Discretionary (Retail & Travel)
    "MCD": "mcdonalds", "SBUX": "starbucks", "NKE": "nike", 
    "BKNG": "booking_holdings", "HD": "home_depot", "COST": "costco",
    
    # Consumer Staples (Essentials)
    "WMT": "walmart", "KO": "coca_cola", "PEP": "pepsico", "PG": "procter_gamble",
    
    # --- ENERGY, MATERIALS & INDUSTRIALS ---
    # Energy (Traditional & Nuclear/Green)
    "XOM": "exxonmobil", "CVX": "chevron", "NEE": "nextera_energy", 
    "VST": "vistra_corp", "OKLO": "oklo_inc", # Nuclear/Data Center focus
    
    # Industrials & Materials
    "CAT": "caterpillar", "UNP": "union_pacific", "BA": "boeing", 
    "GE": "general_electric", "LIN": "linde", "FCX": "freeport_mcmoran",
    
    # --- REAL ESTATE & UTILITIES ---
    "PLD": "prologis", "AMT": "american_tower", "DUK": "duke_energy",
    
    # --- TELECOM & MEDIA ---
    "VZ": "verizon", "TMUS": "tmobile", "NFLX": "netflix", "DIS": "disney",
    
    # --- INTERNATIONAL (ADRs) ---
    "TSM": "taiwan_semiconductor", "ASML": "asml", "TM": "toyota", 
    "BABA": "alibaba", "HDB": "hdfc_bank", "MELI": "mercadolibre",
    
    # --- CRYPTO & DIGITAL ASSETS ---
    "BTC-USD": "bitcoin", "ETH-USD": "ethereum", "SOL-USD": "solana", "BNB-USD": "binance_coin",
    
    # --- INDICES & MACRO ---
    "^GSPC": "sp500", "^NDX": "nasdaq100", "^DJI": "dow_jones", 
    "^RUT": "russell2000" , "DX-Y.NYB": "us_dollar_index", 
    "GC=F": "gold_futures",
    "CL=F": "crude_oil",
}

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
