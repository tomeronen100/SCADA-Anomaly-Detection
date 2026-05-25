import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re


def parse_statistics_file(stats_file_path):
    """
    Parse the statistics text file and extract relevant information
    
    Args:
        stats_file_path: Path to the statistics .txt file
    
    Returns:
        Dictionary containing parsed statistics
    """
    with open(stats_file_path, 'r') as f:
        content = f.read()
    
    stats = {
        'method': None,
        'n_bins': None,
        'train_file': None,
        'test_file': None,
        'train_shape': None,
        'test_shape': None,
        'columns': []
    }
    
    # Extract basic info
    method_match = re.search(r'Discretization Method: (\w+)', content)
    if method_match:
        stats['method'] = method_match.group(1)
    
    bins_match = re.search(r'Number of Bins: (\d+)', content)
    if bins_match:
        stats['n_bins'] = int(bins_match.group(1))
    
    train_file_match = re.search(r'Train Output File: (.+)', content)
    if train_file_match:
        stats['train_file'] = train_file_match.group(1).strip()
    
    test_file_match = re.search(r'Test Output File: (.+)', content)
    if test_file_match:
        stats['test_file'] = test_file_match.group(1).strip()
    
    # Extract shapes
    train_shape_match = re.search(r'TRAIN DATA STATISTICS.*?Shape: \((\d+), (\d+)\)', content, re.DOTALL)
    if train_shape_match:
        stats['train_shape'] = (int(train_shape_match.group(1)), int(train_shape_match.group(2)))
    
    test_shape_match = re.search(r'TEST DATA STATISTICS.*?Shape: \((\d+), (\d+)\)', content, re.DOTALL)
    if test_shape_match:
        stats['test_shape'] = (int(test_shape_match.group(1)), int(test_shape_match.group(2)))
    
    return stats


def create_distribution_plots(train_csv_path, test_csv_path, output_path, method, n_bins):
    """
    Create comprehensive distribution plots comparing train and test data
    
    Args:
        train_csv_path: Path to processed train CSV
        test_csv_path: Path to processed test CSV
        output_path: Path to save the plot (PNG)
        method: Discretization method name
        n_bins: Number of bins used
    """
    # Load data
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    columns = train_df.columns.tolist()
    n_cols = len(columns)
    
    # Create figure with subplots - 2 rows per column (one for histogram, one for box plot)
    fig = plt.figure(figsize=(16, 4 * n_cols))
    
    # Set style
    sns.set_style("whitegrid")
    colors_train = '#2E86AB'  # Blue for train
    colors_test = '#A23B72'   # Purple/pink for test
    
    for idx, col in enumerate(columns):
        # Histogram subplot
        ax1 = plt.subplot(n_cols, 2, idx * 2 + 1)
        
        # Plot histograms
        train_vals = train_df[col].values
        test_vals = test_df[col].values
        
        # Determine bins for histogram
        all_vals = np.concatenate([train_vals, test_vals])
        bins = min(50, int(np.sqrt(len(all_vals))))
        
        ax1.hist(train_vals, bins=bins, alpha=0.6, label='Train', color=colors_train, density=True)
        ax1.hist(test_vals, bins=bins, alpha=0.6, label='Test', color=colors_test, density=True)
        
        ax1.set_xlabel('Value', fontsize=10)
        ax1.set_ylabel('Density', fontsize=10)
        ax1.set_title(f'{col} - Distribution Comparison', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        train_mean = train_vals.mean()
        train_std = train_vals.std()
        test_mean = test_vals.mean()
        test_std = test_vals.std()
        
        stats_text = f'Train: μ={train_mean:.4f}, σ={train_std:.4f}\nTest: μ={test_mean:.4f}, σ={test_std:.4f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Box plot subplot
        ax2 = plt.subplot(n_cols, 2, idx * 2 + 2)
        
        # Create box plots
        data_to_plot = [train_vals, test_vals]
        bp = ax2.boxplot(data_to_plot, labels=['Train', 'Test'], patch_artist=True,
                        widths=0.6, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
        
        # Color the boxes
        bp['boxes'][0].set_facecolor(colors_train)
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor(colors_test)
        bp['boxes'][1].set_alpha(0.6)
        
        ax2.set_ylabel('Value', fontsize=10)
        ax2.set_title(f'{col} - Box Plot Comparison', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add quartile information
        train_q1, train_median, train_q3 = np.percentile(train_vals, [25, 50, 75])
        test_q1, test_median, test_q3 = np.percentile(test_vals, [25, 50, 75])
        
        quartile_text = f'Train Q1/Med/Q3: {train_q1:.3f}/{train_median:.3f}/{train_q3:.3f}\n'
        quartile_text += f'Test Q1/Med/Q3: {test_q1:.3f}/{test_median:.3f}/{test_q3:.3f}'
        
        ax2.text(0.02, 0.98, quartile_text, transform=ax2.transAxes, 
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Overall title
    fig.suptitle(f'Train vs Test Data Comparison - {method} Method ({n_bins} bins)\n' +
                 f'Train: {len(train_df)} samples | Test: {len(test_df)} samples',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    plt.close()


def create_combined_comparison_plot(train_csv_path, test_csv_path, output_path, method, n_bins):
    """
    Create a single comprehensive plot with all columns side by side
    
    Args:
        train_csv_path: Path to processed train CSV
        test_csv_path: Path to processed test CSV
        output_path: Path to save the plot (PNG)
        method: Discretization method name
        n_bins: Number of bins used
    """
    # Load data
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    columns = train_df.columns.tolist()
    n_cols = len(columns)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
    
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Set style
    sns.set_style("whitegrid")
    colors_train = '#2E86AB'  # Blue for train
    colors_test = '#A23B72'   # Purple/pink for test
    
    for idx, col in enumerate(columns):
        # Top row: Histograms
        ax_hist = axes[0, idx]
        
        train_vals = train_df[col].values
        test_vals = test_df[col].values
        
        # Determine bins for histogram
        all_vals = np.concatenate([train_vals, test_vals])
        bins = min(50, int(np.sqrt(len(all_vals))))
        
        ax_hist.hist(train_vals, bins=bins, alpha=0.6, label='Train', color=colors_train, density=True)
        ax_hist.hist(test_vals, bins=bins, alpha=0.6, label='Test', color=colors_test, density=True)
        
        ax_hist.set_xlabel('Value', fontsize=11)
        ax_hist.set_ylabel('Density', fontsize=11)
        ax_hist.set_title(f'{col}', fontsize=12, fontweight='bold')
        ax_hist.legend(loc='upper right', fontsize=9)
        ax_hist.grid(True, alpha=0.3)
        
        # Add statistics
        train_mean = train_vals.mean()
        train_std = train_vals.std()
        test_mean = test_vals.mean()
        test_std = test_vals.std()
        
        stats_text = f'Train: μ={train_mean:.3f}, σ={train_std:.3f}\nTest: μ={test_mean:.3f}, σ={test_std:.3f}'
        ax_hist.text(0.02, 0.98, stats_text, transform=ax_hist.transAxes, 
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Bottom row: Box plots
        ax_box = axes[1, idx]
        
        data_to_plot = [train_vals, test_vals]
        bp = ax_box.boxplot(data_to_plot, labels=['Train', 'Test'], patch_artist=True,
                           widths=0.6, showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        # Color the boxes
        bp['boxes'][0].set_facecolor(colors_train)
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(colors_test)
        bp['boxes'][1].set_alpha(0.7)
        
        ax_box.set_ylabel('Value', fontsize=11)
        ax_box.set_title(f'{col} - Box Plot', fontsize=11, fontweight='bold')
        ax_box.grid(True, alpha=0.3, axis='y')
        
        # Add range information
        train_min, train_max = train_vals.min(), train_vals.max()
        test_min, test_max = test_vals.min(), test_vals.max()
        
        range_text = f'Train: [{train_min:.3f}, {train_max:.3f}]\nTest: [{test_min:.3f}, {test_max:.3f}]'
        ax_box.text(0.02, 0.98, range_text, transform=ax_box.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Overall title
    fig.suptitle(f'Train vs Test Comparison - {method} Discretization ({n_bins} bins)\n' +
                 f'Train samples: {len(train_df):,} | Test samples: {len(test_df):,}',
                 fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to: {output_path}")
    
    plt.close()


def process_statistics_file(stats_file_path):
    """
    Main function to process a statistics file and generate plots
    
    Args:
        stats_file_path: Path to the statistics .txt file
    """
    print(f"\nProcessing: {stats_file_path}")
    
    # Parse statistics file
    stats = parse_statistics_file(stats_file_path)
    
    if not stats['train_file'] or not stats['test_file']:
        print(f"  ERROR: Could not find train/test file paths in statistics file")
        return
    
    # Check if CSV files exist
    if not Path(stats['train_file']).exists():
        print(f"  ERROR: Train file not found: {stats['train_file']}")
        return
    
    if not Path(stats['test_file']).exists():
        print(f"  ERROR: Test file not found: {stats['test_file']}")
        return
    
    # Generate output filename (replace .txt with .png)
    output_path = str(stats_file_path).replace('_statistics.txt', '_plot.png')
    
    # Create plots
    print(f"  Creating visualization...")
    create_combined_comparison_plot(
        train_csv_path=stats['train_file'],
        test_csv_path=stats['test_file'],
        output_path=output_path,
        method=stats['method'],
        n_bins=stats['n_bins']
    )
    
    print(f"  ✓ Complete!")


def main():
    """
    Main execution function - processes all statistics files in results directory
    """
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("ERROR: 'results' directory not found!")
        return
    
    # Find all statistics files
    stats_files = list(results_dir.glob("*_statistics.txt"))
    
    if not stats_files:
        print("No statistics files found in 'results' directory")
        return
    
    print(f"Found {len(stats_files)} statistics file(s)")
    print("="*80)
    
    # Process each file
    for stats_file in sorted(stats_files):
        process_statistics_file(stats_file)
    
    print("\n" + "="*80)
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()