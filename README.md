# Attention Trader

A prototype attention-based neural network for financial time series prediction.

## Project Overview

Attention Trader implements a deep learning model using attention mechanisms to predict financial time series data. The attention mechanism allows the model to focus on the most relevant temporal patterns in historical price data.

## Author

Dmytro Zadokhin

## Features

- Financial data collection from Yahoo Finance (yfinance) - **101 tickers**
- Comprehensive dataset covering 18 categories:
  - **Technology & AI** (26): Big Tech, Semiconductors, Cloud/SaaS, Cybersecurity
  - **Finance & Fintech** (10): Banking, Payments & Digital Assets
  - **Healthcare & Biotech** (10): Weight Loss/Longevity, Pharma, Genomics
  - **Consumer** (10): Discretionary (Retail, Travel), Staples (Essentials)
  - **Energy & Industrials** (11): Traditional/Nuclear/Green Energy, Materials
  - **Real Estate, Utilities, Telecom & Media** (7)
  - **International** (6): ADRs from Asia, Europe, Latin America
  - **Crypto** (4): BTC, ETH, SOL, BNB
  - **Indices & Macro** (7): Market indices, Dollar Index, Gold, Crude Oil
- Advanced data analysis with category-based visualizations
- Price and volume charting for all 18 sectors
- Jupyter notebook support for interactive development

## Project Structure

```
AttentionTrader/
├── data/
│   ├── load_data.py       # Data collection script
│   └── yfinance/          # Downloaded CSV files for 101 tickers
├── venv/                  # Python virtual environment
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── CLAUDE.md             # Claude AI development instructions
```

## Setup

### Prerequisites

- Python 3.13 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AttentionTrader
```

2. Create and activate virtual environment:
```bash
# Create virtual environment
python3 -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Collection

Download historical financial data for all configured tickers:

```bash
python data/load_data.py
```

This will download maximum available historical data for **101 tickers** across:

**Technology & AI (26 tickers):**
- Big Tech: AAPL, MSFT, GOOGL, NVDA, META, AMZN, TSLA
- Semiconductors: AMD, AVGO, ARM, MU, SMCI, INTC, QCOM, AMAT
- Cloud/SaaS: CRM, ORCL, NOW, PLTR, SNOW, ADBE
- Cybersecurity: PANW, CRWD, FTNT, ZS, OKTA

**Finance & Fintech (10 tickers):**
- Banking: JPM, BAC, GS, MS, BRK-B
- Payments: V, MA, PYPL, SQ, COIN

**Healthcare & Biotech (10 tickers):**
- Weight Loss: LLY, NVO
- Pharma: UNH, JNJ, PFE, ABBV, AMGN
- Genomics: ILMN, CRSP, VRTX

**Consumer (10 tickers):**
- Discretionary: MCD, SBUX, NKE, BKNG, HD, COST
- Staples: WMT, KO, PEP, PG

**Energy & Industrials (11 tickers):**
- Energy: XOM, CVX, NEE, VST, OKLO
- Industrials: CAT, UNP, BA, GE, LIN, FCX

**Other Sectors (7 tickers):**
- Real Estate & Utilities: PLD, AMT, DUK
- Telecom & Media: VZ, TMUS, NFLX, DIS

**International (6 tickers):**
- TSM, ASML, TM, BABA, HDB, MELI

**Crypto (4 tickers):**
- BTC-USD, ETH-USD, SOL-USD, BNB-USD

**Indices & Macro (7 tickers):**
- Indices: ^GSPC, ^NDX, ^DJI, ^RUT
- Macro: DX-Y.NYB (Dollar Index), GC=F (Gold), CL=F (Crude Oil)

Data is saved as CSV files in `data/yfinance/` directory.

### Development

Start Jupyter notebook for interactive development:

```bash
jupyter notebook
```

Or use Jupyter Lab:

```bash
jupyter lab
```

## Dependencies

- **pandas** (>=2.0.0) - Data manipulation and analysis
- **numpy** (>=1.24.0) - Numerical computing
- **matplotlib** (>=3.7.0) - Visualization
- **seaborn** (>=0.12.0) - Statistical visualization
- **kagglehub** (>=0.2.0) - Kaggle dataset integration
- **jupyter** (>=1.0.0) - Interactive notebooks
- **ipykernel** (>=6.23.0) - Jupyter kernel
- **yfinance** - Yahoo Finance data downloader

## Data Sources

Financial data is sourced from Yahoo Finance using the yfinance library. The dataset includes:
- Daily OHLCV (Open, High, Low, Close, Volume) data
- Maximum available historical data per ticker
- Market indices, commodities, and forex for macro analysis
- Focus on high-growth sectors: AI/Tech, Biotech, Fintech, Clean Energy

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Acknowledgments

- Yahoo Finance for providing free financial data API
- The attention mechanism research community
