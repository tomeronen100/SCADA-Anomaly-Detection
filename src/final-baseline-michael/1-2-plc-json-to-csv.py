#!/usr/bin/env python3
"""
PLC Register Extractor
Extracts register values from filtered PLC packets and creates CSV with statistics
"""

import json
import csv
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Set


def load_plc_packets(json_file: str) -> List[Dict[Any, Any]]:
    """Load PLC packets from JSON file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
            # Handle both single packet and array of packets
            if isinstance(data, list):
                return data
            else:
                return [data]
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []


def filter_packets_by_func_code(packets: List[Dict]) -> List[Dict]:
    """Filter packets to only include func_code 3 or 4."""
    filtered = []
    for packet in packets:
        func_code = packet.get("func_code")
        if func_code in ["3", "4", 3, 4]:  # Handle both string and int formats
            filtered.append(packet)
    return filtered


def extract_register_value(packet: Dict, register: int) -> Any:
    """Extract register value from a packet."""
    # Try to find the register field
    register_key = f"register_{register}_(uint16)"
    
    if register_key in packet:
        register_data = packet[register_key]
        if isinstance(register_data, dict) and "regval_uint16" in register_data:
            return register_data["regval_uint16"]
    
    return None


def create_csv_from_packets(packets: List[Dict], registers: List[int], output_file: str):
    """Create CSV file with timestamp and register values."""
    
    # Prepare header
    header = ["Timestamp"] + [f"Register_{reg}" for reg in registers]
    
    # Prepare rows
    rows = []
    for packet in packets:
        timestamp = packet.get("Timestamp", "")
        row = [timestamp]
        
        for register in registers:
            value = extract_register_value(packet, register)
            row.append(value if value is not None else "")
        
        rows.append(row)
    
    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"Saved CSV to '{output_file}'")
    return rows


def analyze_columns(rows: List[List], registers: List[int]) -> Dict:
    """Analyze CSV data for statistics."""
    stats = {}
    
    # Analyze each register column (skip timestamp column at index 0)
    for idx, register in enumerate(registers, start=1):
        col_name = f"Register_{register}"
        values = [row[idx] for row in rows]
        
        # Count nulls (empty strings)
        null_count = sum(1 for v in values if v == "")
        non_null_values = [v for v in values if v != ""]
        
        # Value counts
        value_counts = Counter(non_null_values)
        
        stats[col_name] = {
            'register': register,
            'total_packets': len(values),
            'null_count': null_count,
            'non_null_count': len(non_null_values),
            'null_percentage': (null_count / len(values) * 100) if values else 0,
            'unique_values': len(value_counts),
            'value_counts': dict(value_counts.most_common())
        }
    
    return stats


def save_statistics(stats: Dict, total_packets: int, registers: List[int], output_file: str):
    """Save detailed statistics to text file."""
    with open(output_file, 'w') as f:
        f.write("PLC Register Analysis Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total Packets Analyzed: {total_packets}\n")
        f.write(f"Registers Tracked: {len(registers)}\n")
        f.write(f"Register List: {', '.join(map(str, registers))}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("TABLE DESCRIPTION\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Columns:\n")
        f.write("  - Column 0: Timestamp (packet timestamp)\n")
        for idx, register in enumerate(registers, start=1):
            f.write(f"  - Column {idx}: Register_{register} (register {register} value)\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("COLUMN STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        
        for col_name, col_stats in stats.items():
            f.write(f"\n{col_name} (Register {col_stats['register']})\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Total Packets:        {col_stats['total_packets']}\n")
            f.write(f"  Null Count:           {col_stats['null_count']} ({col_stats['null_percentage']:.1f}%)\n")
            f.write(f"  Non-Null Count:       {col_stats['non_null_count']} ({100-col_stats['null_percentage']:.1f}%)\n")
            f.write(f"  Unique Values:        {col_stats['unique_values']}\n")
            
            if col_stats['value_counts']:
                f.write(f"\n  Value Distribution:\n")
                for value, count in col_stats['value_counts'].items():
                    percentage = (count / col_stats['non_null_count'] * 100) if col_stats['non_null_count'] > 0 else 0
                    f.write(f"    {value}: {count} occurrences ({percentage:.1f}%)\n")
            else:
                f.write(f"\n  No values found (all null)\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        # Overall statistics
        total_cells = total_packets * len(registers)
        total_nulls = sum(s['null_count'] for s in stats.values())
        
        f.write(f"Total Data Cells:     {total_cells}\n")
        f.write(f"Total Null Cells:     {total_nulls} ({total_nulls/total_cells*100:.1f}%)\n")
        f.write(f"Total Non-Null Cells: {total_cells - total_nulls} ({(total_cells-total_nulls)/total_cells*100:.1f}%)\n")
    
    print(f"Saved statistics to '{output_file}'")


def main():
    # Configuration
    # input_json_file = "data/one_plc_json_226.json"  # Input JSON file with PLC packets
    # registers = [13262, 13264, 13266]  # Registers to track
    
    input_json_file = "data/one_plc_json_161.json"  # Input JSON file with PLC packets
    registers = [3,5,7,9,11, 43,45,46,48,50]  # Registers to track

    
    # Extract base filename and create output paths
    input_path = Path(input_json_file)
    base_name = input_path.stem  # Get filename without extension
    
    output_csv = f"data/{base_name}.csv"  # Output CSV file in data dir
    output_stats = f"results/{base_name}_stats.txt"  # Output statistics file in results dir
    
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    print("PLC Register Extractor")
    print("=" * 70)
    print(f"Input File: {input_json_file}")
    print(f"Registers: {', '.join(map(str, registers))}")
    print(f"Output CSV: {output_csv}")
    print(f"Output Stats: {output_stats}")
    print("=" * 70 + "\n")
    
    # Load packets
    print("Loading PLC packets...")
    packets = load_plc_packets(input_json_file)
    
    if not packets:
        print("No packets loaded. Exiting.")
        return
    
    print(f"Loaded {len(packets)} packets")
    
    # Filter packets by func_code 3 or 4
    print("Filtering packets by func_code (3 or 4)...")
    packets = filter_packets_by_func_code(packets)
    
    if not packets:
        print("No packets with func_code 3 or 4 found. Exiting.")
        return
    
    print(f"Filtered to {len(packets)} packets with func_code 3 or 4\n")
    
    # Create CSV
    print("Extracting register values and creating CSV...")
    rows = create_csv_from_packets(packets, registers, output_csv)
    
    # Analyze data
    print("Analyzing data...")
    stats = analyze_columns(rows, registers)
    
    # Save statistics
    save_statistics(stats, len(packets), registers, output_stats)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total packets processed: {len(packets)}")
    print(f"Registers tracked: {len(registers)}")
    print(f"\nNull counts per register:")
    for col_name, col_stats in stats.items():
        print(f"  {col_name}: {col_stats['null_count']} nulls ({col_stats['null_percentage']:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    main()