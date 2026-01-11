"""
Add Days Delta Feature to OHLCV Data

This script adds a 'days_delta' column to financial time series data,
representing the number of days since the last trading day.

This feature helps ML models understand:
- Weekend gaps (days_delta = 3 for Monday after Friday)
- Holiday gaps (days_delta > 3 for extended closures)
- Normal trading days (days_delta = 1)

Input: CSV files from data/yfinance/
Output: Enriched CSV files saved to data/enriched/

The days_delta column is inserted as the first column after the Date index.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def add_days_delta(df):
    """
    Add days_delta column to a dataframe with DatetimeIndex.

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        DataFrame with new 'days_delta' column as first column
    """
    # Calculate days since last observation
    # First row gets NaN, then we calculate the difference
    days_delta = df.index.to_series().diff().dt.days

    # First trading day has no previous day, so we set it to NaN or 0
    # Using 0 makes sense: "0 days since start of data"
    days_delta.iloc[0] = 0

    # Insert as first column
    df_enriched = df.copy()
    df_enriched.insert(0, 'days_delta', days_delta.astype(int))

    return df_enriched


def process_all_tickers(input_dir='data/yfinance', output_dir='data/enriched'):
    """
    Process all CSV files in input directory and save enriched versions to output directory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all CSV files
    csv_files = sorted(input_path.glob('*.csv'))

    if not csv_files:
        print(f"ERROR: No CSV files found in {input_dir}")
        sys.exit(1)

    print(f"\nFound {len(csv_files)} CSV files to process")
    print("=" * 80)

    success_count = 0
    error_count = 0
    errors = []

    for i, csv_file in enumerate(csv_files, 1):
        ticker_name = csv_file.stem
        print(f"\n[{i}/{len(csv_files)}] Processing: {ticker_name}")

        try:
            # Read CSV with multi-level headers from yfinance
            df = pd.read_csv(csv_file, header=[0, 1, 2])

            # Flatten multi-level columns
            df.columns = df.columns.get_level_values(0)
            df = df.set_index('Price')
            df.index.name = 'Date'
            df.index = pd.to_datetime(df.index)

            # Remove rows with invalid dates
            df = df[df.index.notna()]

            # Sort by date to ensure chronological order
            df = df.sort_index()

            print(f"  Loaded {len(df)} rows from {df.index.min().date()} to {df.index.max().date()}")

            # Add days_delta feature
            df_enriched = add_days_delta(df)

            # Show statistics about gaps
            gap_stats = df_enriched['days_delta'].value_counts().sort_index()
            print(f"  Days delta distribution:")
            for days, count in gap_stats.items():
                if days <= 5 or count > 5:  # Show common gaps or unusual ones
                    print(f"    {days} day(s): {count} occurrences")

            # Save to output directory
            output_file = output_path / f"{ticker_name}.csv"
            df_enriched.to_csv(output_file)

            print(f"  ✓ Saved to: {output_file}")
            print(f"  Columns: {list(df_enriched.columns)}")

            success_count += 1

        except Exception as e:
            error_count += 1
            error_msg = f"{ticker_name}: {str(e)}"
            errors.append(error_msg)
            print(f"  ✗ ERROR: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("\nPROCESSING SUMMARY")
    print(f"  Successful: {success_count}/{len(csv_files)}")
    print(f"  Errors: {error_count}/{len(csv_files)}")

    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  - {error}")

    print(f"\nEnriched files saved to: {output_path.absolute()}")
    print("=" * 80)

    return success_count, error_count


def main():
    """Main entry point."""
    print("=" * 80)
    print("ADD DAYS DELTA FEATURE SCRIPT")
    print("=" * 80)
    print("\nThis script adds 'days_delta' column to OHLCV data.")
    print("The column represents days since the last trading day:")
    print("  - 1 = next trading day (normal)")
    print("  - 3 = Monday after Friday (weekend)")
    print("  - 4+ = holiday or extended closure")
    print("\nInput:  data/yfinance/")
    print("Output: data/enriched/")
    print("=" * 80)

    success_count, error_count = process_all_tickers()

    if error_count == 0:
        print("\n✓ All files processed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠ Completed with {error_count} error(s)")
        sys.exit(1)


if __name__ == "__main__":
    main()
