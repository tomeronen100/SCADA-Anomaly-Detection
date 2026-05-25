#!/usr/bin/env python3
"""
PLC Gap Remover - Simple Version
Removes records where inter-arrival time exceeds a threshold
"""

import csv
from pathlib import Path


def remove_gaps_simple(input_file: str, output_file: str, threshold: float, log_file: str):
    """
    Remove records where inter-arrival time exceeds threshold.
    Keep all other records as-is without recalculating anything.
    """
    # Read input CSV
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        all_rows = list(reader)
    
    # Filter rows
    kept_rows = []
    removed_rows = []
    
    for i, row in enumerate(all_rows, start=1):
        try:
            inter_arrival = float(row['inter_arrival_time'])
            
            if inter_arrival > threshold:
                # Remove this record
                removed_rows.append({
                    'packet_num': i,
                    'timestamp': row['Timestamp'],
                    'inter_arrival': inter_arrival
                })
            else:
                # Keep this record
                kept_rows.append(row)
        except (KeyError, ValueError):
            # If can't read inter_arrival_time, keep the record
            kept_rows.append(row)
    
    # Write output CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(kept_rows)
    
    # Write log
    with open(log_file, 'w') as f:
        f.write("PLC GAP REMOVAL LOG\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Output file: {output_file}\n")
        f.write(f"Threshold: {threshold} seconds\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("REMOVAL SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Original records: {len(all_rows)}\n")
        f.write(f"Records removed: {len(removed_rows)}\n")
        f.write(f"Records kept: {len(kept_rows)}\n")
        f.write(f"Removal rate: {len(removed_rows) / len(all_rows) * 100:.2f}%\n\n")
        
        if removed_rows:
            f.write("=" * 70 + "\n")
            f.write("REMOVED RECORDS\n")
            f.write("=" * 70 + "\n\n")
            
            for rec in removed_rows:
                f.write(f"Packet #{rec['packet_num']}: "
                       f"Timestamp={rec['timestamp']}, "
                       f"Inter-arrival={rec['inter_arrival']:.6f}s (> {threshold}s)\n")
        else:
            f.write("No records removed - all inter-arrival times below threshold.\n")
    
    return len(all_rows), len(kept_rows), len(removed_rows)


def main():
    # Configuration
    # input_csv = "data/clean_226_plc_table.csv"
    # threshold = 113.0  # Threshold in seconds
    # output_csv = "data/clean_226_plc_table_no_gaps.csv"
    # output_log = "results/clean_226_plc_table_no_gaps_stats.txt"
    
    input_csv = "data/clean_161_plc_table.csv"
    # threshold = 500  ################################## Threshold in seconds - inter-arrival times above this are considered gaps
    threshold = 3700  ################################## Threshold in seconds - inter-arrival times above this are considered gaps
    output_csv = "data/clean_161_plc_table_no_gaps.csv"
    output_log = "results/clean_161_plc_table_no_gaps_stats.txt"
    output_plot = "graphs/inter_arrival_time_161_no_gaps.png"

    # Create directories if needed
    Path("data").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    print("PLC Gap Remover - Simple Version")
    print("=" * 70)
    print(f"Input: {input_csv}")
    print(f"Threshold: {threshold} seconds")
    print(f"Output CSV: {output_csv}")
    print(f"Output Log: {output_log}")
    print("=" * 70)
    
    # Remove gaps
    print("\nProcessing...")
    original_count, kept_count, removed_count = remove_gaps_simple(
        input_csv, output_csv, threshold, output_log
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Original records: {original_count}")
    print(f"Records removed: {removed_count}")
    print(f"Records kept: {kept_count}")
    print(f"Threshold: {threshold} seconds")
    print("=" * 70)
    print(f"\nOutput saved to: {output_csv}")
    print(f"Log saved to: {output_log}")


if __name__ == "__main__":
    main()    
