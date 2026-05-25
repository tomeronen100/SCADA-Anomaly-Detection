#!/usr/bin/env python3
"""
PLC Packet Analyzer
Filters Modbus packets by PLC IP address and generates statistics
"""

import json
import os
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any
import sys

def load_json_files(data_dir: str) -> List[Dict[Any, Any]]:
    """Load all JSON files from the data directory and extract packets."""
    all_packets = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Directory '{data_dir}' does not exist")
        return []
    
    json_files = list(data_path.glob("*.json"))
    
    if not json_files:
        print(f"Warning: No JSON files found in '{data_dir}'")
        return []
    
    print(f"Found {len(json_files)} JSON file(s)")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Handle both single packet and array of packets
                if isinstance(data, list):
                    all_packets.extend(data)
                else:
                    all_packets.append(data)
                    
            print(f"Loaded {json_file.name}")
        except json.JSONDecodeError as e:
            print(f"Error decoding {json_file.name}: {e}")
        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
    
    return all_packets


def filter_packets_by_plc(packets: List[Dict], plc_ip: str) -> tuple:
    """
    Filter packets containing the specified PLC IP address.
    Returns (filtered_packets, source_count, dest_count)
    """
    filtered = []
    source_count = 0
    dest_count = 0
    
    for packet in packets:
        source_ip = packet.get("Source_IP", "")
        dest_ip = packet.get("Destination_IP", "")
        
        is_source = source_ip == plc_ip
        is_dest = dest_ip == plc_ip
        
        if is_source or is_dest:
            filtered.append(packet)
            if is_source:
                source_count += 1
            if is_dest:
                dest_count += 1
    
    return filtered, source_count, dest_count


def analyze_function_codes(packets: List[Dict]) -> Counter:
    """Extract and count function codes from packets."""
    func_codes = Counter()
    
    for packet in packets:
        func_code = packet.get("func_code")
        if func_code is not None:
            func_codes[func_code] += 1
    
    return func_codes


def save_filtered_packets(packets: List[Dict], output_file: str):
    """Save filtered packets to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(packets, f, indent=4)
    print(f"\nSaved {len(packets)} packets to '{output_file}'")


def save_statistics(stats: Dict, output_file: str):
    """Save statistics to a text file."""
    with open(output_file, 'w') as f:
        f.write("PLC Packet Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"PLC IP Address: {stats['plc_ip']}\n")
        f.write(f"Total Packets: {stats['total_packets']}\n\n")
        
        f.write(f"Packets where PLC is Source: {stats['source_count']}\n")
        f.write(f"Packets where PLC is Destination: {stats['dest_count']}\n\n")
        
        f.write("Function Code Distribution:\n")
        f.write("-" * 50 + "\n")
        
        if stats['func_codes']:
            for func_code, count in sorted(stats['func_codes'].items()):
                percentage = (count / stats['total_packets']) * 100
                f.write(f"  Function Code {func_code}: {count} packets ({percentage:.1f}%)\n")
        else:
            f.write("  No function codes found\n")
    
    print(f"Saved statistics to '{output_file}'")


def main():
    # Configuration
    # plc_ip = "132.72.32.226"  # PLC IP address to filter
    plc_ip = "132.72.35.161"  # PLC IP address to filter
    data_dir = "data/new/modbus"  # Directory containing JSON files
    output_prefix = "plc_filtered"  # Prefix for output files
    
    print(f"\nPLC Packet Analyzer")
    print("=" * 50)
    print(f"Data Directory: {data_dir}")
    print(f"PLC IP Address: {plc_ip}")
    print(f"Output Prefix: {output_prefix}")
    print("=" * 50 + "\n")
    
    # Load all packets
    print("Loading JSON files...")
    all_packets = load_json_files(data_dir)
    
    if not all_packets:
        print("No packets loaded. Exiting.")
        sys.exit(1)
    
    print(f"Total packets loaded: {len(all_packets)}\n")
    
    # Filter packets by PLC IP
    print(f"Filtering packets for PLC IP: {plc_ip}...")
    filtered_packets, source_count, dest_count = filter_packets_by_plc(all_packets, plc_ip)
    
    if not filtered_packets:
        print(f"No packets found for PLC IP: {plc_ip}")
        sys.exit(0)
    
    print(f"Found {len(filtered_packets)} packets")
    
    # Analyze function codes
    func_codes = analyze_function_codes(filtered_packets)
    
    # Prepare statistics
    stats = {
        'plc_ip': plc_ip,
        'total_packets': len(filtered_packets),
        'source_count': source_count,
        'dest_count': dest_count,
        'func_codes': dict(func_codes)
    }
    
    # Save results
    output_json = f"{output_prefix}.json"
    output_txt = f"{output_prefix}_stats.txt"
    
    save_filtered_packets(filtered_packets, output_json)
    save_statistics(stats, output_txt)
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total packets: {stats['total_packets']}")
    print(f"PLC as source: {stats['source_count']}")
    print(f"PLC as destination: {stats['dest_count']}")
    print(f"\nFunction codes found: {len(func_codes)}")
    for func_code, count in sorted(func_codes.items()):
        print(f"  Code {func_code}: {count} packets")
    print("=" * 50)


if __name__ == "__main__":
    main()