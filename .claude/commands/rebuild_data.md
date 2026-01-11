Rebuild all data by running the data loading script, adding temporal features, and re-executing the analysis notebook.

Execute the following steps:

1. Run the data loading script using the Python virtual environment:
   ```bash
   source venv/bin/activate && python executables/load_data.py
   ```

2. Add temporal features (days_delta) to create enriched dataset:
   ```bash
   source venv/bin/activate && python executables/add_days_delta.py
   ```

3. Re-execute the analysis notebook to regenerate all outputs and charts:
   ```bash
   source venv/bin/activate && jupyter nbconvert --to notebook --execute executables/analyze_data.ipynb --output analyze_data.ipynb --ExecutePreprocessor.timeout=600
   ```

Report the results of each step, including any errors encountered.

**Output:**
- `data/yfinance/` - Raw OHLCV data from Yahoo Finance (91 tickers)
- `data/enriched/` - OHLCV data with days_delta temporal feature (91 tickers)
- `executables/analyze_data.ipynb` - Updated notebook with fresh outputs
- `images/` - Regenerated category charts (18 PNG files)
