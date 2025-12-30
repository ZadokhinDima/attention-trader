# Claude AI Development Instructions

This document provides instructions for Claude AI when working on the AttentionTrader project.

## Python Environment Setup

**IMPORTANT**: This project uses a Python virtual environment. Always activate the venv before running any Python commands.

### Virtual Environment Activation

Before running ANY Python code or commands in this project, you MUST activate the virtual environment:

```bash
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```

### Verification

After activation, verify you're using the correct Python:
```bash
which python  # Should show: /Users/dimonster/Desktop/repos/AttentionTrader/venv/bin/python
python --version  # Should show Python 3.13+
```

### Installing Dependencies

If dependencies are missing or requirements.txt is updated:

```bash
# Make sure venv is activated first!
source venv/bin/activate
pip install -r requirements.txt
```

### Adding New Dependencies

When adding new Python packages:

1. Activate venv
2. Install the package: `pip install package_name`
3. Update requirements.txt: `pip freeze > requirements.txt`

## Project Context

### Technology Stack
- **Language**: Python 3.13+
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Data Source**: yfinance (Yahoo Finance)
- **Development**: Jupyter notebooks
- **Package Management**: pip + requirements.txt

### Project Structure
```
AttentionTrader/
├── data/
│   ├── load_data.py          # Data collection script (40+ tickers)
│   └── yfinance/             # CSV files with historical price data
├── venv/                     # Virtual environment (DO NOT MODIFY)
├── requirements.txt          # Python dependencies
├── README.md                # User-facing documentation
└── CLAUDE.md                # This file (AI assistant instructions)
```

### Key Files

- **[data/load_data.py](data/load_data.py)**: Downloads historical data from Yahoo Finance for 40+ tickers across:
  - Big Tech (AAPL, MSFT, GOOGL, NVDA, META)
  - Finance (JPM, V, BRK-B)
  - Healthcare (UNH, JNJ, PFE)
  - Consumer (AMZN, TSLA, MCD, WMT, KO, PG)
  - Energy (XOM, CVX)
  - Industrials (CAT, UNP, BA)
  - Utilities (NEE, DUK)
  - Real Estate REITs (PLD, AMT)
  - Materials (LIN, FCX)
  - Telecom (VZ, TMUS)
  - International (TSM, ASML, TM, BABA)
  - Crypto (BTC-USD, ETH-USD, SOL-USD)
  - Indices (^GSPC, ^NDX, ^DJI, ^RUT, ^VIX)

- **[requirements.txt](requirements.txt)**: Python package dependencies with version constraints

## Development Workflow

### Running Python Scripts

Always use the venv Python:

```bash
# Activate venv first
source venv/bin/activate

# Run scripts
python data/load_data.py
```

### Data Collection

To refresh or update financial data:

```bash
source venv/bin/activate
python data/load_data.py
```

This downloads maximum available historical data and saves CSVs to `data/yfinance/`.

### Jupyter Notebooks

Start Jupyter with the venv kernel:

```bash
source venv/bin/activate
jupyter notebook
# or
jupyter lab
```

## Code Conventions

### Python Style
- Follow PEP 8 style guidelines
- Use descriptive variable names
- Add comments for complex logic
- Include docstrings for functions and classes

### Data Processing
- Use pandas for data manipulation
- Store processed data as CSV or Parquet files
- Include data validation checks
- Handle missing values explicitly

### Version Control
- This is a Git repository (not yet initialized with remote)
- Make atomic commits with clear messages
- Don't commit large data files (they're already in `.gitignore` if configured)
- Don't commit the `venv/` directory

## Common Tasks

### Adding a New Ticker

Edit [data/load_data.py](data/load_data.py:8-78) and add to the `tickers` dictionary:

```python
tickers = {
    # ... existing tickers ...
    "NEW": "ticker_name",  # Add your ticker here
}
```

Then run:
```bash
source venv/bin/activate
python data/load_data.py
```

### Installing New Packages

```bash
source venv/bin/activate
pip install package_name
pip freeze > requirements.txt  # Update requirements
```

### Data Analysis

Create Jupyter notebooks for analysis:

```bash
source venv/bin/activate
jupyter notebook
```

Use pandas to load CSV data:
```python
import pandas as pd
df = pd.read_csv('data/yfinance/apple.csv', index_col=0, parse_dates=True)
```

## Environment Notes

- **OS**: macOS (Darwin 24.5.0)
- **Python**: 3.13+ (venv uses system Python)
- **Working Directory**: `/Users/dimonster/Desktop/repos/AttentionTrader`
- **Git**: Repository initialized but no remote configured

## Important Reminders for Claude

1. **ALWAYS activate venv first**: `source venv/bin/activate`
2. **Use venv Python**: Not system Python
3. **Check requirements.txt**: Before running code that needs dependencies
4. **Don't modify venv/**: This directory is auto-generated
5. **Update requirements.txt**: When adding new packages
6. **Run scripts from project root**: Use relative paths like `python data/load_data.py`
7. **Verify environment**: Use `which python` to confirm venv activation

## Project Goals

This is a prototype for attention-based neural networks applied to financial time series prediction. Future development will include:

- Attention mechanism implementation
- Time series preprocessing pipelines
- Model training and evaluation
- Prediction and backtesting frameworks
- Performance visualization

## Questions?

When uncertain about:
- **Dependencies**: Check [requirements.txt](requirements.txt)
- **Data sources**: See [data/load_data.py](data/load_data.py)
- **Project structure**: Refer to this document
- **Environment**: Verify venv is activated with `which python`
