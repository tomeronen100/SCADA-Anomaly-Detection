import pandas as pd
import os
from datetime import datetime

def split_train_test(input_file, train_file, test_file, stats_file):
    """
    Split CSV file into train (90%) and test (10%) sets sequentially.
    Save statistics to a text file.
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    
    # Read the CSV file
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Calculate split point (90% for train, 10% for test)
    total_rows = len(df)
    split_point = int(total_rows * 0.9)
    
    # Split the data
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]
    
    print(f"Saving training data to {train_file}...")
    train_df.to_csv(train_file, index=False)
    
    print(f"Saving test data to {test_file}...")
    test_df.to_csv(test_file, index=False)
    
    # Generate statistics
    stats = []
    stats.append("=" * 60)
    stats.append("DATA SPLIT STATISTICS")
    stats.append("=" * 60)
    stats.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    stats.append(f"\nInput File: {input_file}")
    stats.append(f"Total Rows: {total_rows:,}")
    stats.append(f"Total Columns: {len(df.columns)}")
    stats.append(f"\n{'-' * 60}")
    stats.append("TRAINING SET")
    stats.append(f"{'-' * 60}")
    stats.append(f"File: {train_file}")
    stats.append(f"Rows: {len(train_df):,}")
    stats.append(f"Percentage: {(len(train_df) / total_rows * 100):.2f}%")
    stats.append(f"\n{'-' * 60}")
    stats.append("TEST SET")
    stats.append(f"{'-' * 60}")
    stats.append(f"File: {test_file}")
    stats.append(f"Rows: {len(test_df):,}")
    stats.append(f"Percentage: {(len(test_df) / total_rows * 100):.2f}%")
    stats.append(f"\n{'-' * 60}")
    stats.append("COLUMN NAMES")
    stats.append(f"{'-' * 60}")
    for i, col in enumerate(df.columns, 1):
        stats.append(f"{i}. {col}")
    stats.append("=" * 60)
    
    # Save statistics to file
    with open(stats_file, 'w') as f:
        f.write('\n'.join(stats))
    
    print(f"\nStatistics saved to {stats_file}")
    print("\nSummary:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Training rows: {len(train_df):,} ({len(train_df) / total_rows * 100:.2f}%)")
    print(f"  Test rows: {len(test_df):,} ({len(test_df) / total_rows * 100:.2f}%)")
    print("\nDone!")

def main():
    # Configuration
    # input_file = "data/clean_226_plc_table_no_gaps.csv"
    input_file = "data/clean_161_plc_table_no_gaps.csv"
    
    # Output file paths
    # train_file = "data/final_226_snd_train.csv"
    # test_file = "data/final_226_snd_test.csv"
    # stats_file = "results/split_226_statistics.txt"
    
    train_file = "data/final_161_snd_train.csv"
    test_file = "data/final_161_snd_test.csv"
    stats_file = "results/split_161_statistics.txt"
    
    split_train_test(input_file, train_file, test_file, stats_file)

if __name__ == "__main__":
    main()