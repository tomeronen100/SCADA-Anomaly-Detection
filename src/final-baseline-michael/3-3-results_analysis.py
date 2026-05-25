import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def extract_features(test_folder):
    """
    Extract table type (t1/t2/t3), discretization method (EFD/EWD/KMeans), 
    and number of bins (4/5/6/7) from test folder name
    """
    # Pattern: test_plc_226_snd_{table_type}_{method}_{bins}_timestamp
    pattern = r'test_plc_226_snd_(t\d+)_(EFD|EWD|KMeans)_(\d+)_'
    match = re.search(pattern, test_folder)
    
    if match:
        table_type = match.group(1)
        method = match.group(2)
        bins = int(match.group(3))
        return table_type, method, bins
    return None, None, None

def plot_by_table_type(df, output_dir):
    """Plot mean F1 score by table type (t1, t2, t3)"""
    # Group by table type and calculate mean F1
    grouped = df.groupby('table_type')['f1_score'].mean().sort_index()
    
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(grouped))
    bars = plt.bar(x_pos, grouped.values, alpha=0.8, edgecolor='black', linewidth=1.5, color='steelblue')
    
    # Add value labels on bars
    for i, (x, y) in enumerate(zip(x_pos, grouped.values)):
        plt.text(x, y + 0.005, f'{y:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Table Type', fontsize=13, fontweight='bold')
    plt.ylabel('Mean F1 Score', fontsize=13, fontweight='bold')
    plt.title('Mean F1 Score by Table Type', fontsize=15, fontweight='bold', pad=20)
    plt.xticks(x_pos, grouped.index, fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim(0, max(grouped.values) * 1.15)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'f1_by_table_type.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_by_discretization(df, output_dir):
    """Plot mean F1 score by discretization method (EFD, EWD, KMeans)"""
    # Group by method and calculate mean F1
    grouped = df.groupby('method')['f1_score'].mean().sort_index()
    
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(grouped))
    
    # Different colors for each method
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = plt.bar(x_pos, grouped.values, alpha=0.8, edgecolor='black', linewidth=1.5, color=colors)
    
    # Add value labels on bars
    for i, (x, y) in enumerate(zip(x_pos, grouped.values)):
        plt.text(x, y + 0.005, f'{y:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Discretization Method', fontsize=13, fontweight='bold')
    plt.ylabel('Mean F1 Score', fontsize=13, fontweight='bold')
    plt.title('Mean F1 Score by Discretization Method', fontsize=15, fontweight='bold', pad=20)
    plt.xticks(x_pos, grouped.index, fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim(0, max(grouped.values) * 1.15)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'f1_by_discretization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_by_bins(df, output_dir):
    """Plot mean F1 score by number of bins (4, 5, 6, 7)"""
    # Group by bins and calculate mean F1
    grouped = df.groupby('bins')['f1_score'].mean().sort_index()
    
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(grouped))
    bars = plt.bar(x_pos, grouped.values, alpha=0.8, edgecolor='black', linewidth=1.5, color='coral')
    
    # Add value labels on bars
    for i, (x, y) in enumerate(zip(x_pos, grouped.values)):
        plt.text(x, y + 0.005, f'{y:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Number of Bins', fontsize=13, fontweight='bold')
    plt.ylabel('Mean F1 Score', fontsize=13, fontweight='bold')
    plt.title('Mean F1 Score by Number of Bins', fontsize=15, fontweight='bold', pad=20)
    plt.xticks(x_pos, grouped.index.astype(int), fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim(0, max(grouped.values) * 1.15)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'f1_by_bins.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def main():
    # Read the CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_dir, 'final_results.csv')    

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return
    
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} records from {csv_file}")
    
    # Extract features from test folder names
    print("\nExtracting features from test folder names...")
    df['table_type'], df['method'], df['bins'] = zip(*df['test_folder'].apply(extract_features))
    
    # Remove any rows where extraction failed
    df = df.dropna(subset=['table_type', 'method', 'bins'])
    print(f"Successfully extracted features for {len(df)} records")
    
    # Create output directory
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCreating plots in '{output_dir}' directory...")
    
    # Generate the three plots
    plot_by_table_type(df, output_dir)
    plot_by_discretization(df, output_dir)
    plot_by_bins(df, output_dir)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print("\nMean F1 Score by Table Type:")
    table_stats = df.groupby('table_type')['f1_score'].agg(['mean', 'std', 'count'])
    print(table_stats)
    
    print("\nMean F1 Score by Discretization Method:")
    method_stats = df.groupby('method')['f1_score'].agg(['mean', 'std', 'count'])
    print(method_stats)
    
    print("\nMean F1 Score by Number of Bins:")
    bins_stats = df.groupby('bins')['f1_score'].agg(['mean', 'std', 'count'])
    print(bins_stats)
    
    print("\n✓ All plots generated successfully!")

if __name__ == "__main__":
    main()