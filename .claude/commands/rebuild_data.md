Rebuild all data by running the data loading script and re-executing the analysis notebook.

Execute the following steps:

1. Run the data loading script using the Python virtual environment:
   ```bash
   source venv/bin/activate && python data/load_data.py
   ```

2. Re-execute the analysis notebook to regenerate all outputs and charts:
   ```bash
   source venv/bin/activate && jupyter nbconvert --to notebook --execute analyze_data.ipynb --output analyze_data.ipynb --ExecutePreprocessor.timeout=600
   ```

Report the results of each step, including any errors encountered.
