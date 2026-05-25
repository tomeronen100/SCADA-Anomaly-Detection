import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import pickle
from pathlib import Path
import argparse
from datetime import datetime
from sklearn.cluster import KMeans


class KMeansDiscretizer:
    """KMeans Discretization - uses KMeans clustering to create bins"""
    
    def __init__(self, n_bins=4, random_state=42):
        self.n_bins = n_bins
        self.random_state = random_state
        self.kmeans = None
        self.bin_edges = None
        self.cluster_centers = None
    
    def fit(self, data):
        """Fit the discretizer on training data"""
        # Reshape data for sklearn (needs 2D array)
        data_reshaped = data.reshape(-1, 1)
        
        # Fit KMeans clustering
        self.kmeans = KMeans(n_clusters=self.n_bins, random_state=self.random_state, n_init=10)
        self.kmeans.fit(data_reshaped)
        
        # Store cluster centers
        self.cluster_centers = np.sort(self.kmeans.cluster_centers_.flatten())
        
        # Create bin edges as midpoints between cluster centers
        if self.n_bins > 1:
            self.bin_edges = np.zeros(self.n_bins + 1)
            self.bin_edges[0] = np.min(data)
            self.bin_edges[-1] = np.max(data)
            
            # Midpoints between consecutive cluster centers
            for i in range(1, self.n_bins):
                self.bin_edges[i] = (self.cluster_centers[i-1] + self.cluster_centers[i]) / 2
        else:
            self.bin_edges = np.array([np.min(data), np.max(data)])
        
        return self
    
    def transform(self, data):
        """Transform data into discrete bins"""
        if self.kmeans is None:
            raise ValueError("Discretizer must be fitted before transform")
        
        # Use digitize to assign each value to a bin based on bin edges
        # bins are 0-indexed
        discretized = np.digitize(data, self.bin_edges[1:-1])
        return discretized
    
    def fit_transform(self, data):
        """Fit and transform in one step"""
        self.fit(data)
        return self.transform(data)


class MinMaxScaler:
    """Min-Max normalization - scales data to [0, 1] range"""
    
    def __init__(self):
        self.min = None
        self.max = None
    
    def fit(self, data):
        """Fit the scaler on training data"""
        self.min = np.min(data)
        self.max = np.max(data)
        return self
    
    def transform(self, data):
        """Transform data to [0, 1] range"""
        if self.min is None or self.max is None:
            raise ValueError("Scaler must be fitted before transform")
        
        # Handle case where min == max
        if self.max - self.min == 0:
            return np.zeros_like(data)
        
        normalized = (data - self.min) / (self.max - self.min)
        return normalized
    
    def fit_transform(self, data):
        """Fit and transform in one step"""
        self.fit(data)
        return self.transform(data)


def process_data(train_csv_path: str, test_csv_path: str, n_bins: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """
    Process pre-split train and test CSV data with discretization and normalization
    
    Args:
        train_csv_path: Path to training CSV file
        test_csv_path: Path to test CSV file
        n_bins: Number of bins for KMeans discretization (default: 10)
    
    Returns:
        train_processed: Processed training data
        test_processed: Processed test data
        discretizers: Dictionary of fitted KMeans discretizers for 'Register_' columns (excluding '_time')
        scalers: Dictionary of fitted MinMax scalers for ALL columns
    """
    
    # Load train data
    print(f"Loading training data from: {train_csv_path}")
    train_df = pd.read_csv(train_csv_path)
    print(f"Train data shape: {train_df.shape}")
    print(f"Columns: {list(train_df.columns)}")
    
    # Load test data
    print(f"\nLoading test data from: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    print(f"Test data shape: {test_df.shape}")
    print(f"Columns: {list(test_df.columns)}")
    
    # Verify columns match
    if list(train_df.columns) != list(test_df.columns):
        raise ValueError("Train and test data must have the same columns!")
    
    # Find columns starting with 'Register_' but NOT ending with '_time'
    reg_columns = [col for col in train_df.columns 
                   if col.startswith('Register_') and not col.endswith('_time')]
    print(f"\nFound {len(reg_columns)} columns starting with 'Register_' (excluding '_time' columns, for discretization):")
    for col in reg_columns:
        print(f"  - {col}")
    
    # All columns for normalization
    all_columns = list(train_df.columns)
    print(f"\nAll {len(all_columns)} columns will be normalized:")
    for col in all_columns:
        print(f"  - {col}")
    
    # Store discretizers and scalers (fitted on train data only)
    discretizers = {}
    scalers = {}
    
    # STEP 1: Process TRAIN data
    print(f"\n{'='*60}")
    print("STEP 1: Processing TRAIN data...")
    print(f"{'='*60}")
    
    # 1a. Discretize 'Register_' columns (excluding '_time') in train
    print(f"\n  1a. Discretizing 'Register_' columns in TRAIN...")
    for col in reg_columns:
        print(f"\n    Processing: {col}")
        print(f"      - Applying KMeans discretization (n_bins={n_bins})")
        discretizer = KMeansDiscretizer(n_bins=n_bins)
        train_df[col] = discretizer.fit_transform(train_df[col].values)
        discretizers[col] = discretizer
        print(f"      - Discretized range: [{train_df[col].min()}, {train_df[col].max()}]")
        print(f"      - Cluster centers: {discretizer.cluster_centers}")
        print(f"      - Bin edges: {discretizer.bin_edges}")
    
    # 1b. Normalize ALL columns in train
    print(f"\n  1b. Normalizing ALL columns in TRAIN...")
    for col in all_columns:
        print(f"\n    Processing: {col}")
        print(f"      - Applying MinMax normalization")
        scaler = MinMaxScaler()
        train_df[col] = scaler.fit_transform(train_df[col].values)
        scalers[col] = scaler
        print(f"      - Normalized range: [{train_df[col].min():.4f}, {train_df[col].max():.4f}]")
        print(f"      - Min: {scaler.min:.4f}, Max: {scaler.max:.4f}")
    
    # STEP 2: Process TEST data (using TRAIN transformers - NO DATA LEAKAGE)
    print(f"\n{'='*60}")
    print("STEP 2: Processing TEST data (using TRAIN transformers)...")
    print(f"{'='*60}")
    
    # 2a. Discretize 'Register_' columns in test using TRAIN discretizers
    print(f"\n  2a. Discretizing 'Register_' columns in TEST...")
    for col in reg_columns:
        print(f"\n    Processing: {col}")
        print(f"      - Applying TRAIN discretizer")
        # Use the train discretizer (NO fitting on test data)
        test_df[col] = discretizers[col].transform(test_df[col].values)
        print(f"      - Discretized range: [{test_df[col].min()}, {test_df[col].max()}]")
        print(f"      - Using cluster centers from train: {discretizers[col].cluster_centers}")
        print(f"      - Using bin edges from train: {discretizers[col].bin_edges}")
    
    # 2b. Normalize ALL columns in test using TRAIN scalers
    print(f"\n  2b. Normalizing ALL columns in TEST...")
    for col in all_columns:
        print(f"\n    Processing: {col}")
        print(f"      - Applying TRAIN scaler")
        # Use the train scaler (NO fitting on test data)
        test_df[col] = scalers[col].transform(test_df[col].values)
        print(f"      - Normalized range: [{test_df[col].min():.4f}, {test_df[col].max():.4f}]")
        print(f"      - Using min/max from train: Min={scalers[col].min:.4f}, Max={scalers[col].max:.4f}")
    
    return train_df, test_df, discretizers, scalers


def save_results(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                 discretizers: Dict, scalers: Dict,
                 train_csv_path: str, test_csv_path: str,
                 output_prefix: str, n_bins: int):
    """
    Save processed train data, processed test data, fitted transformers, and detailed statistics
    
    Args:
        train_df: Processed training data
        test_df: Processed test data
        discretizers: Dictionary of fitted discretizers (from train data)
        scalers: Dictionary of fitted scalers (from train data)
        train_csv_path: Original train CSV path (for naming)
        test_csv_path: Original test CSV path (for naming)
        output_prefix: Prefix for output files (e.g., "processed")
        n_bins: Number of bins used for discretization
    """
    
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    # Extract base name from train and test CSV (without extension)
    train_base_name = Path(train_csv_path).stem
    test_base_name = Path(test_csv_path).stem
    
    # Method name
    method_name = "KMeans"  # KMeans Discretization
    
    # Build file names: PREFIX_ORIGINAL_NAME_methodOfDisc_numOfBins
    train_filename = f"{output_prefix}_{train_base_name}_{method_name}_{n_bins}_train.csv"
    test_filename = f"{output_prefix}_{test_base_name}_{method_name}_{n_bins}_test.csv"
    stats_filename = f"{output_prefix}_{train_base_name}_{method_name}_{n_bins}_statistics.txt"
    transformers_filename = f"{output_prefix}_{train_base_name}_{method_name}_{n_bins}_transformers.pkl"
    
    # Save train data (processed)
    train_path = f"results/{train_filename}"
    train_df.to_csv(train_path, index=False)
    print(f"\nSaved processed train data to: {train_path}")
    print(f"  Shape: {train_df.shape}")
    
    # Save test data (processed)
    test_path = f"results/{test_filename}"
    test_df.to_csv(test_path, index=False)
    print(f"\nSaved processed test data to: {test_path}")
    print(f"  Shape: {test_df.shape}")
    
    # Save transformers for future use
    transformers_path = f"results/{transformers_filename}"
    with open(transformers_path, 'wb') as f:
        pickle.dump({
            'discretizers': discretizers,
            'scalers': scalers,
            'n_bins': n_bins,
            'method': method_name
        }, f)
    print(f"\nSaved transformers to: {transformers_path}")
    
    # Create detailed statistics report
    stats_path = f"results/{stats_filename}"
    with open(stats_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DATA PROCESSING STATISTICS AND LOGGER\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Discretization Method: {method_name}\n")
        f.write(f"Number of Bins: {n_bins}\n")
        f.write(f"Train Input File: {train_csv_path}\n")
        f.write(f"Test Input File: {test_csv_path}\n")
        f.write(f"Train Output File: {train_path}\n")
        f.write(f"Test Output File: {test_path}\n")
        f.write("\n" + "="*80 + "\n\n")
        
        # TRAIN DATA STATISTICS
        f.write("TRAIN DATA STATISTICS\n")
        f.write("-"*80 + "\n\n")
        
        f.write(f"Shape: {train_df.shape}\n")
        f.write(f"Rows: {train_df.shape[0]}\n")
        f.write(f"Columns: {train_df.shape[1]}\n\n")
        
        f.write("Describe (Statistical Summary):\n")
        f.write("-"*80 + "\n")
        f.write(train_df.describe().to_string())
        f.write("\n\n")
        
        f.write("Data Types:\n")
        f.write("-"*80 + "\n")
        f.write(train_df.dtypes.to_string())
        f.write("\n\n")
        
        f.write("Missing Values:\n")
        f.write("-"*80 + "\n")
        missing_train = train_df.isnull().sum()
        f.write(missing_train.to_string())
        f.write(f"\nTotal missing values: {missing_train.sum()}\n\n")
        
        f.write("Value Ranges per Column:\n")
        f.write("-"*80 + "\n")
        for col in train_df.columns:
            f.write(f"{col}: [{train_df[col].min():.6f}, {train_df[col].max():.6f}]\n")
        f.write("\n")
        
        # TEST DATA STATISTICS
        f.write("="*80 + "\n\n")
        f.write("TEST DATA STATISTICS\n")
        f.write("-"*80 + "\n\n")
        
        f.write(f"Shape: {test_df.shape}\n")
        f.write(f"Rows: {test_df.shape[0]}\n")
        f.write(f"Columns: {test_df.shape[1]}\n\n")
        
        f.write("Describe (Statistical Summary):\n")
        f.write("-"*80 + "\n")
        f.write(test_df.describe().to_string())
        f.write("\n\n")
        
        f.write("Data Types:\n")
        f.write("-"*80 + "\n")
        f.write(test_df.dtypes.to_string())
        f.write("\n\n")
        
        f.write("Missing Values:\n")
        f.write("-"*80 + "\n")
        missing_test = test_df.isnull().sum()
        f.write(missing_test.to_string())
        f.write(f"\nTotal missing values: {missing_test.sum()}\n\n")
        
        f.write("Value Ranges per Column:\n")
        f.write("-"*80 + "\n")
        for col in test_df.columns:
            f.write(f"{col}: [{test_df[col].min():.6f}, {test_df[col].max():.6f}]\n")
        f.write("\n")
        
        # TRANSFORMATION DETAILS
        f.write("="*80 + "\n\n")
        f.write("TRANSFORMATION DETAILS\n")
        f.write("-"*80 + "\n\n")
        
        f.write(f"Discretized Columns ({len(discretizers)}):\n")
        f.write("-"*80 + "\n")
        for col in sorted(discretizers.keys()):
            f.write(f"\n{col}:\n")
            f.write(f"  Number of bins (clusters): {discretizers[col].n_bins}\n")
            f.write(f"  Cluster centers: {discretizers[col].cluster_centers}\n")
            f.write(f"  Bin edges: {discretizers[col].bin_edges}\n")
            f.write(f"  Train value range: [{train_df[col].min()}, {train_df[col].max()}]\n")
            f.write(f"  Test value range: [{test_df[col].min()}, {test_df[col].max()}]\n")
            f.write(f"  Train unique values: {train_df[col].nunique()}\n")
            f.write(f"  Test unique values: {test_df[col].nunique()}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write(f"\nNormalized Columns ({len(scalers)}):\n")
        f.write("-"*80 + "\n")
        for col in sorted(scalers.keys()):
            f.write(f"\n{col}:\n")
            f.write(f"  Scaler fitted on train - Min: {scalers[col].min:.6f}, Max: {scalers[col].max:.6f}\n")
            f.write(f"  Train normalized range: [{train_df[col].min():.6f}, {train_df[col].max():.6f}]\n")
            f.write(f"  Test normalized range: [{test_df[col].min():.6f}, {test_df[col].max():.6f}]\n")
            f.write(f"  Train mean: {train_df[col].mean():.6f}, std: {train_df[col].std():.6f}\n")
            f.write(f"  Test mean: {test_df[col].mean():.6f}, std: {test_df[col].std():.6f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\nSaved detailed statistics to: {stats_path}")


def main():
    """Main execution function"""
    
    ################## Configuration - 226
    # train_csv = "train_data/final_226_snd_train__table_1_time_and_values.csv"
    # test_csv = "test_data/final_226_snd_test__table_1_time_and_values.csv"

    # train_csv = "train_data/final_226_snd_train__table_2_time_and_duration.csv"
    # test_csv = "test_data/final_226_snd_test__table_2_time_and_duration.csv"

    # train_csv = "train_data/final_226_snd_train__table_3_time_and_state.csv"
    # test_csv = "test_data/final_226_snd_test__table_3_time_and_state.csv"

    ################### Configuration - 161
    # train_csv = "train_data/final_161_snd_train__table_1_time_and_values.csv"
    # test_csv = "test_data/final_161_snd_test__table_1_time_and_values.csv"

    # train_csv = "train_data/final_161_snd_train__table_2_time_and_duration.csv"
    # test_csv = "test_data/final_161_snd_test__table_2_time_and_duration.csv"

    train_csv = "train_data/final_161_snd_train__table_3_time_and_state.csv"
    test_csv = "test_data/final_161_snd_test__table_3_time_and_state.csv"
    
    # Prefix for output files
    output_prefix = "processed"
    
    # List of bin sizes to test
    bins_list = [4, 5, 6, 7]
    
    # Process data for each bin size
    for n_bins in bins_list:
        print("\n" + "="*80)
        print(f"PROCESSING WITH {n_bins} BINS (KMeans Discretization)")
        print("="*80 + "\n")
        
        # Process data
        print("STARTING DATA PROCESSING")
        print("-"*80 + "\n")
        
        train_df, test_df, discretizers, scalers = process_data(
            train_csv_path=train_csv,
            test_csv_path=test_csv,
            n_bins=n_bins
        )
        
        # Save results
        print("\n" + "-"*80)
        print("SAVING RESULTS")
        print("-"*80)
        
        save_results(train_df, test_df, discretizers, scalers,
                     train_csv_path=train_csv,
                     test_csv_path=test_csv,
                     output_prefix=output_prefix,
                     n_bins=n_bins)
        
        print("\n" + "="*80)
        print(f"PROCESSING COMPLETE FOR {n_bins} BINS!")
        print("="*80 + "\n")
    
    print("\n" + "="*80)
    print("ALL PROCESSING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

