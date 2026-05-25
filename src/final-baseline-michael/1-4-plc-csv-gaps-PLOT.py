#!/usr/bin/env python3
"""
PLC Inter-Arrival Time Analyzer
Analyzes and plots inter-arrival times from cleaned PLC CSV data
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt


def load_inter_arrival_times(csv_file: str) -> tuple:
    """Load inter-arrival times from CSV file."""
    packet_numbers = []
    inter_arrival_times = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        packet_num = 1
        
        for row in reader:
            try:
                inter_arrival = float(row['inter_arrival_time'])
                packet_numbers.append(packet_num)
                inter_arrival_times.append(inter_arrival)
                packet_num += 1
            except (KeyError, ValueError) as e:
                print(f"Warning: Skipping row due to error: {e}")
                continue
    
    return packet_numbers, inter_arrival_times


def plot_inter_arrival_times(packet_numbers: list, inter_arrival_times: list, output_file: str):
    """Create and save plot of inter-arrival times."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Full view
    ax1.plot(packet_numbers, inter_arrival_times, marker='o', linestyle='-', 
             linewidth=1, markersize=4, color='blue', alpha=0.7)
    
    ax1.set_xlabel('Packet Number', fontsize=12)
    ax1.set_ylabel('Inter-Arrival Time (seconds)', fontsize=12)
    ax1.set_title('PLC Packet Inter-Arrival Times - Full View', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add average line to full view
    if len(inter_arrival_times) > 1:
        avg = sum(inter_arrival_times[1:]) / len(inter_arrival_times[1:])
        ax1.axhline(y=avg, color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {avg:.4f}s', alpha=0.7)
        ax1.legend()
    
    # Plot 2: Zoomed view (exclude outliers above 95th percentile)
    times_sorted = sorted(inter_arrival_times[1:])  # Exclude first 0
    if len(times_sorted) > 0:
        p95_idx = int(len(times_sorted) * 0.95)
        y_max = times_sorted[p95_idx] if p95_idx < len(times_sorted) else times_sorted[-1]
        y_max = y_max * 1.2  # Add 20% padding
    else:
        y_max = max(inter_arrival_times) if inter_arrival_times else 1
    
    ax2.plot(packet_numbers, inter_arrival_times, marker='o', linestyle='-', 
             linewidth=1, markersize=4, color='blue', alpha=0.7)
    
    ax2.set_xlabel('Packet Number', fontsize=12)
    ax2.set_ylabel('Inter-Arrival Time (seconds)', fontsize=12)
    ax2.set_title('PLC Packet Inter-Arrival Times - Detailed View (95th percentile)', 
                  fontsize=14, fontweight='bold')
    ax2.set_ylim(0, y_max)
    ax2.grid(True, alpha=0.3)
    
    # Add average line to zoomed view
    if len(inter_arrival_times) > 1:
        ax2.axhline(y=avg, color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {avg:.4f}s', alpha=0.7)
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_file}")
    print(f"  - Full view shows all data points")
    print(f"  - Detailed view zoomed to {y_max:.4f}s (excludes extreme outliers)")


def calculate_statistics(inter_arrival_times: list) -> dict:
    """Calculate comprehensive statistics about inter-arrival times."""
    stats = {
        'total_packets': len(inter_arrival_times)
    }
    
    if len(inter_arrival_times) > 1:
        # Exclude first value (which is 0 by definition)
        times = inter_arrival_times[1:]
        
        # Basic statistics
        stats['mean'] = sum(times) / len(times)
        stats['min'] = min(times)
        stats['max'] = max(times)
        stats['range'] = stats['max'] - stats['min']
        
        # Median
        sorted_times = sorted(times)
        n = len(sorted_times)
        if n % 2 == 0:
            stats['median'] = (sorted_times[n//2 - 1] + sorted_times[n//2]) / 2
        else:
            stats['median'] = sorted_times[n//2]
        
        # Mode (most common value)
        from collections import Counter
        counts = Counter(times)
        stats['mode'] = counts.most_common(1)[0][0]
        stats['mode_count'] = counts.most_common(1)[0][1]
        stats['mode_percentage'] = (stats['mode_count'] / len(times)) * 100
        
        # Standard deviation
        variance = sum((x - stats['mean']) ** 2 for x in times) / len(times)
        stats['std_dev'] = variance ** 0.5
        
        # Quartiles
        q1_idx = n // 4
        q3_idx = (3 * n) // 4
        stats['q1'] = sorted_times[q1_idx]
        stats['q3'] = sorted_times[q3_idx]
        stats['iqr'] = stats['q3'] - stats['q1']
        
        # Sum
        stats['sum'] = sum(times)
        
        # Unique values
        stats['unique_values'] = len(set(times))
        stats['value_counts'] = dict(counts.most_common())
    
    return stats


def print_statistics(stats: dict):
    """Print statistics about inter-arrival times."""
    print("\n" + "=" * 70)
    print("INTER-ARRIVAL TIME STATISTICS")
    print("=" * 70)
    
    print(f"Total packets: {stats['total_packets']}")
    
    if 'mean' in stats:
        print(f"\nExcluding first packet (which has inter-arrival time = 0):")
        print(f"  Mean:                  {stats['mean']:.6f} seconds")
        print(f"  Median:                {stats['median']:.6f} seconds")
        print(f"  Mode (most common):    {stats['mode']:.6f} seconds ({stats['mode_count']} occurrences, {stats['mode_percentage']:.1f}%)")
        print(f"  Standard Deviation:    {stats['std_dev']:.6f} seconds")
        print(f"  Minimum:               {stats['min']:.6f} seconds")
        print(f"  Maximum:               {stats['max']:.6f} seconds")
        print(f"  Range:                 {stats['range']:.6f} seconds")
        print(f"  Q1 (25th percentile):  {stats['q1']:.6f} seconds")
        print(f"  Q3 (75th percentile):  {stats['q3']:.6f} seconds")
        print(f"  IQR:                   {stats['iqr']:.6f} seconds")
        print(f"  Sum:                   {stats['sum']:.6f} seconds")
        print(f"  Unique values:         {stats['unique_values']}")
    
    print("=" * 70)


def save_statistics_to_file(stats: dict, output_file: str):
    """Save comprehensive statistics to text file."""
    with open(output_file, 'w') as f:
        f.write("PLC INTER-ARRIVAL TIME ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total Packets Analyzed: {stats['total_packets']}\n\n")
        
        if 'mean' in stats:
            f.write("=" * 70 + "\n")
            f.write("SUMMARY STATISTICS (excluding first packet)\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("Central Tendency:\n")
            f.write(f"  Mean:                  {stats['mean']:.6f} seconds\n")
            f.write(f"  Median:                {stats['median']:.6f} seconds\n")
            f.write(f"  Mode (most common):    {stats['mode']:.6f} seconds\n")
            f.write(f"    Occurrences:         {stats['mode_count']} ({stats['mode_percentage']:.1f}%)\n\n")
            
            f.write("Spread/Dispersion:\n")
            f.write(f"  Standard Deviation:    {stats['std_dev']:.6f} seconds\n")
            f.write(f"  Range:                 {stats['range']:.6f} seconds\n")
            f.write(f"  IQR (Interquartile):   {stats['iqr']:.6f} seconds\n\n")
            
            f.write("Range:\n")
            f.write(f"  Minimum:               {stats['min']:.6f} seconds\n")
            f.write(f"  Maximum:               {stats['max']:.6f} seconds\n\n")
            
            f.write("Quartiles:\n")
            f.write(f"  Q1 (25th percentile):  {stats['q1']:.6f} seconds\n")
            f.write(f"  Q2 (50th percentile):  {stats['median']:.6f} seconds\n")
            f.write(f"  Q3 (75th percentile):  {stats['q3']:.6f} seconds\n\n")
            
            f.write("Other:\n")
            f.write(f"  Sum:                   {stats['sum']:.6f} seconds\n")
            f.write(f"  Unique values:         {stats['unique_values']}\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("VALUE FREQUENCY DISTRIBUTION\n")
            f.write("=" * 70 + "\n\n")
            
            total_values = sum(stats['value_counts'].values())
            for value, count in sorted(stats['value_counts'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_values) * 100
                f.write(f"  {value:.6f} seconds: {count} occurrences ({percentage:.2f}%)\n")
    
    print(f"Statistics saved to: {output_file}")


def main():

    # Configuration
    input_csv = "data/clean_226_plc_table.csv"
    output_plot = "graphs/inter_arrival_time_226.png"
    output_stats = "graphs/inter_arrival_time_226_stats.txt"


    # input_csv = "data/clean_161_plc_table.csv"
    # output_plot = "graphs/inter_arrival_time_161.png"
    # output_stats = "graphs/inter_arrival_time_161_stats.txt"
    
    # Create graphs directory if it doesn't exist
    Path("graphs").mkdir(exist_ok=True)
    
    print("PLC Inter-Arrival Time Analyzer")
    print("=" * 70)
    print(f"Input CSV: {input_csv}")
    print(f"Output Plot: {output_plot}")
    print(f"Output Stats: {output_stats}")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    packet_numbers, inter_arrival_times = load_inter_arrival_times(input_csv)
    print(f"Loaded {len(inter_arrival_times)} packets")
    
    # Calculate statistics
    print("\nCalculating statistics...")
    stats = calculate_statistics(inter_arrival_times)
    
    # Print statistics
    print_statistics(stats)
    
    # Save statistics to file
    print("\nSaving statistics...")
    save_statistics_to_file(stats, output_stats)
    
    # Create plot
    print("\nCreating plot...")
    plot_inter_arrival_times(packet_numbers, inter_arrival_times, output_plot)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
