#!/usr/bin/env python3
"""
PLC CSV Cleaner
Cleans PLC register CSV data by removing null records, filling missing values, and calculating inter-arrival times
"""

import csv
import os
from pathlib import Path
from typing import List, Dict, Any


def load_csv(input_file: str) -> tuple:
    """Load CSV file and return header and rows."""
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def is_all_registers_null(row: List[str], start_col: int = 1) -> bool:
    """Check if all register columns (excluding timestamp) are empty/null."""
    register_values = row[start_col:]
    return all(val == '' or val is None for val in register_values)


def remove_null_records(rows: List[List[str]], log: List[str]) -> List[List[str]]:
    """
    Remove records where all register values are null and not followed by a non-null record.
    Pattern: null-nonnull-null-nonnull-...
    """
    log.append("\n" + "=" * 70)
    log.append("STEP 1: REMOVING NULL RECORDS")
    log.append("=" * 70)
    
    if not rows:
        log.append("No rows to process.")
        return rows
    
    cleaned_rows = []
    removed_count = 0
    
    i = 0
    while i < len(rows):
        current_is_null = is_all_registers_null(rows[i])
        
        # Look ahead to see if next record is non-null
        has_next_nonnull = False
        if i + 1 < len(rows):
            has_next_nonnull = not is_all_registers_null(rows[i + 1])
        
        # Keep the record if:
        # 1. It has at least one non-null register value, OR
        # 2. It's all null BUT followed by a non-null record
        if not current_is_null or has_next_nonnull:
            cleaned_rows.append(rows[i])
        else:
            removed_count += 1
            log.append(f"Removed record at timestamp {rows[i][0]} (all nulls, not followed by non-null)")
        
        i += 1
    
    log.append(f"\nTotal records before: {len(rows)}")
    log.append(f"Total records removed: {removed_count}")
    log.append(f"Total records after: {len(cleaned_rows)}")
    log.append(f"Removal rate: {removed_count / len(rows) * 100:.2f}%")
    
    return cleaned_rows


def forward_fill_nulls(rows: List[List[str]], header: List[str], log: List[str]) -> List[List[str]]:
    """
    Fill null values with the last known value for each register.
    For the first rows, use the first known value.
    """
    log.append("\n" + "=" * 70)
    log.append("STEP 2: FILLING NULL VALUES")
    log.append("=" * 70)
    
    if not rows:
        log.append("No rows to process.")
        return rows
    
    num_registers = len(header) - 1  # Exclude timestamp column
    filled_rows = [row[:] for row in rows]  # Deep copy
    
    # Track statistics
    fills_per_column = {col_idx: 0 for col_idx in range(1, len(header))}
    
    # First, find the first known value for each column
    first_known_values = {}
    for col_idx in range(1, len(header)):
        for row in filled_rows:
            if row[col_idx] != '':
                first_known_values[col_idx] = row[col_idx]
                break
    
    # Forward fill for each column
    for col_idx in range(1, len(header)):
        last_value = first_known_values.get(col_idx, '')
        
        for row_idx in range(len(filled_rows)):
            if filled_rows[row_idx][col_idx] == '':
                # Fill with last known value
                filled_rows[row_idx][col_idx] = last_value
                fills_per_column[col_idx] += 1
            else:
                # Update last known value
                last_value = filled_rows[row_idx][col_idx]
    
    # Log statistics
    log.append(f"\nTotal rows processed: {len(filled_rows)}")
    log.append(f"\nFills per register column:")
    
    total_fills = 0
    for col_idx in range(1, len(header)):
        fills = fills_per_column[col_idx]
        total_fills += fills
        log.append(f"  {header[col_idx]}: {fills} values filled")
    
    total_cells = len(filled_rows) * num_registers
    log.append(f"\nTotal cells: {total_cells}")
    log.append(f"Total fills: {total_fills}")
    log.append(f"Fill rate: {total_fills / total_cells * 100:.2f}%")
    
    return filled_rows


def add_inter_arrival_time(rows: List[List[str]], header: List[str], log: List[str]) -> tuple:
    """
    Add inter-arrival time column calculated as the difference in timestamps.
    First record gets 0.
    """
    log.append("\n" + "=" * 70)
    log.append("STEP 3: ADDING INTER-ARRIVAL TIME")
    log.append("=" * 70)
    
    if not rows:
        log.append("No rows to process.")
        return header, rows
    
    # Add new column to header
    new_header = header + ['inter_arrival_time']
    
    # Calculate inter-arrival times
    new_rows = []
    for i, row in enumerate(rows):
        new_row = row[:]
        
        if i == 0:
            # First record gets 0
            inter_arrival = 0.0
        else:
            # Calculate difference from previous timestamp
            try:
                current_ts = float(rows[i][0])
                previous_ts = float(rows[i-1][0])
                inter_arrival = current_ts - previous_ts
            except (ValueError, IndexError):
                inter_arrival = 0.0
        
        new_row.append(str(inter_arrival))
        new_rows.append(new_row)
    
    # Calculate statistics
    inter_arrival_times = []
    for i in range(1, len(new_rows)):
        try:
            inter_arrival_times.append(float(new_rows[i][-1]))
        except ValueError:
            pass
    
    log.append(f"\nTotal records: {len(new_rows)}")
    log.append(f"First record inter-arrival time: 0.0 (by definition)")
    
    if inter_arrival_times:
        avg_inter_arrival = sum(inter_arrival_times) / len(inter_arrival_times)
        min_inter_arrival = min(inter_arrival_times)
        max_inter_arrival = max(inter_arrival_times)
        
        log.append(f"\nInter-arrival time statistics (excluding first record):")
        log.append(f"  Average: {avg_inter_arrival:.6f} seconds")
        log.append(f"  Minimum: {min_inter_arrival:.6f} seconds")
        log.append(f"  Maximum: {max_inter_arrival:.6f} seconds")
    
    return new_header, new_rows


def save_csv(header: List[str], rows: List[List[str]], output_file: str, log: List[str]):
    """Save cleaned data to CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    log.append(f"\n\nSaved cleaned CSV to: {output_file}")


def save_log(log: List[str], output_file: str):
    """Save log to text file."""
    with open(output_file, 'w') as f:
        f.write("PLC CSV CLEANING LOG\n")
        f.write("=" * 70 + "\n")
        for line in log:
            f.write(line + "\n")
    
    print(f"Saved cleaning log to: {output_file}")


def main():
    # Configuration
    input_csv = "data/one_plc_226.csv"  # Input CSV file
    output_csv = "data/clean_226_plc_table.csv"  # Output cleaned CSV
    output_log = "results/clean_226_plc_table.txt"  # Output log file

    # input_csv = "data/one_plc_161.csv"  # Input CSV file
    # output_csv = "data/clean_161_plc_table.csv"  # Output cleaned CSV
    # output_log = "results/clean_161_plc_table.txt"  # Output log file

    
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    # Initialize log
    log = []
    log.append(f"Input file: {input_csv}")
    log.append(f"Output CSV: {output_csv}")
    log.append(f"Output log: {output_log}")
    
    print("PLC CSV Cleaner")
    print("=" * 70)
    print(f"Input: {input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Output Log: {output_log}")
    print("=" * 70 + "\n")
    
    # Load CSV
    print("Loading CSV...")
    header, rows = load_csv(input_csv)
    log.append(f"\nOriginal data: {len(rows)} records, {len(header)} columns")
    print(f"Loaded {len(rows)} records with {len(header)} columns\n")
    
    # Step 1: Remove null records
    print("Step 1: Removing null records...")
    cleaned_rows = remove_null_records(rows, log)
    print(f"  Removed {len(rows) - len(cleaned_rows)} records\n")
    
    # Step 2: Forward fill nulls
    print("Step 2: Filling null values with forward fill...")
    filled_rows = forward_fill_nulls(cleaned_rows, header, log)
    print(f"  Filled null values in {len(filled_rows)} records\n")
    
    # Step 3: Add inter-arrival time
    print("Step 3: Adding inter-arrival time column...")
    final_header, final_rows = add_inter_arrival_time(filled_rows, header, log)
    print(f"  Added inter-arrival time column\n")
    
    # Save results
    print("Saving results...")
    save_csv(final_header, final_rows, output_csv, log)
    save_log(log, output_log)
    
    # Print summary
    print("\n" + "=" * 70)
    print("CLEANING SUMMARY")
    print("=" * 70)
    print(f"Original records: {len(rows)}")
    print(f"Records removed: {len(rows) - len(cleaned_rows)}")
    print(f"Final records: {len(final_rows)}")
    print(f"Columns added: 1 (inter_arrival_time)")
    print(f"Final columns: {len(final_header)}")
    print("=" * 70)
    print(f"\nCleaned CSV saved to: {output_csv}")
    print(f"Cleaning log saved to: {output_log}")


if __name__ == "__main__":
    main()