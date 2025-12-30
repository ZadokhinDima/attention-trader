# Attention Trader

A prototype attention-based neural network for financial time series prediction.

## Project Overview

Attention Trader implements a deep learning model using attention mechanisms to predict financial time series data. The attention mechanism allows the model to focus on the most relevant temporal patterns in historical price data.

## Author

Dmytro Zadokhin

## Features

- Financial data collection from Yahoo Finance (yfinance)
- Comprehensive dataset covering multiple sectors:
  - Big Tech (AAPL, MSFT, GOOGL, NVDA, META)
  - Finance (JPM, V, BRK-B)
  - Healthcare (UNH, JNJ, PFE)
  - Consumer sectors (AMZN, TSLA, WMT, KO)
  - Energy, Industrials, Utilities
  - Cryptocurrencies (BTC, ETH, SOL)
  - Market indices (S&P 500, NASDAQ, Dow Jones, Russell 2000)
- Data preprocessing and analysis tools
- Jupyter notebook support for interactive development

## Project Structure

```
AttentionTrader/
├── data/
│   ├── load_data.py       # Data collection script
│   └── yfinance/          # Downloaded CSV files for 40+ tickers
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

This will download maximum available historical data for 41 tickers including:
- Major US stocks across all sectors
- International companies (TSM, ASML, TM, BABA)
- Cryptocurrencies (BTC, ETH, SOL)
- Market indices (S&P 500, NASDAQ, Dow Jones, Russell 2000)

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
- Major market indices for benchmarking

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Acknowledgments

- Yahoo Finance for providing free financial data API
- The attention mechanism research community
