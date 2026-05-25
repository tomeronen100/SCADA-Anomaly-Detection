import json
from pathlib import Path
from collections import defaultdict, Counter

# ==========================
# HELPERS
# ==========================
def iter_packets(obj):
    if isinstance(obj, dict):
        yield obj
    elif isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                yield x


def s(x):
    return "" if x is None else str(x).strip()


# ==========================
# REGISTER EXTRACTION
# ==========================
def extract_registers(pkt):
    """
    Extract register addresses and their values from a packet.
    Returns list of tuples: [(register_addr, value), ...]
    """
    registers = []
    
    # Look for register fields in the packet (e.g., "register_13262_(uint16)")
    for key, val in pkt.items():
        if key.startswith("register_") and isinstance(val, dict):
            # Extract regnum16 and regval_uint16
            reg_num = val.get("regnum16")
            reg_val = val.get("regval_uint16")
            
            if reg_num is not None and reg_val is not None:
                try:
                    reg_addr = int(reg_num)
                    registers.append((reg_addr, reg_val))
                except (ValueError, TypeError):
                    pass
    
    # Fallback: try older format with start_addr and values
    if not registers:
        start_addr = pkt.get("start_addr") or pkt.get("starting_address") or pkt.get("address") or pkt.get("reference_num")
        if start_addr is not None:
            try:
                start_addr = int(start_addr)
            except (ValueError, TypeError):
                start_addr = None
        
        values = pkt.get("register_values") or pkt.get("values") or pkt.get("data")
        
        if start_addr is not None and values is not None:
            if isinstance(values, list):
                for i, val in enumerate(values):
                    registers.append((start_addr + i, val))
            else:
                registers.append((start_addr, values))
    
    # Also check for single register operations (func_code 6)
    if not registers:
        reg_addr = pkt.get("register_addr") or pkt.get("reg_addr")
        reg_value = pkt.get("register_value") or pkt.get("reg_value") or pkt.get("value")
        
        if reg_addr is not None and reg_value is not None:
            try:
                registers.append((int(reg_addr), reg_value))
            except (ValueError, TypeError):
                pass
    
    return registers


# ==========================
# CORE LOGIC
# ==========================
def analyze_plc_registers(modbus_dir, plc_ip):
    json_files = list(modbus_dir.rglob("*.json"))
    
    total_plc_packets = 0
    packets_with_registers = 0
    
    # register_addr -> list of (value, packet_index)
    register_data = defaultdict(list)
    
    errors = []
    packet_counter = 0
    
    for fp in json_files:
        try:
            data = json.loads(fp.read_text())
        except Exception as e:
            errors.append(f"{fp}: {e}")
            continue
        
        for pkt in iter_packets(data):
            src = s(pkt.get("Source_IP"))
            dst = s(pkt.get("Destination_IP"))
            
            # Filter: PLC must be involved
            if src != plc_ip and dst != plc_ip:
                continue
            
            total_plc_packets += 1
            
            # Filter: func_code must be 3, 4, or 6
            fc = s(pkt.get("func_code"))
            if fc not in ["3", "4", "6"]:
                continue
            
            # Extract registers
            registers = extract_registers(pkt)
            
            if registers:
                packets_with_registers += 1
                packet_counter += 1
                
                for reg_addr, value in registers:
                    register_data[reg_addr].append((value, packet_counter))
    
    return {
        "files": len(json_files),
        "total_plc_packets": total_plc_packets,
        "packets_with_registers": packets_with_registers,
        "register_data": register_data,
        "errors": errors,
    }


# ==========================
# STATISTICS
# ==========================
def calculate_register_stats(values_list):
    """
    Given list of (value, packet_index), calculate statistics.
    Returns dict with value counts, changes, and variation info.
    """
    if not values_list:
        return {}
    
    # Extract just the values
    values = [v for v, _ in values_list]
    
    # Count occurrences of each value
    value_counts = Counter(values)
    
    # Count changes (transitions from one value to another)
    changes = 0
    for i in range(1, len(values)):
        if values[i] != values[i-1]:
            changes += 1
    
    # Calculate variation
    # Try to convert to numeric for range calculation
    numeric_values = []
    for v in values:
        try:
            numeric_values.append(float(v))
        except (ValueError, TypeError):
            pass
    
    variation_info = {}
    if numeric_values:
        variation_info["min"] = min(numeric_values)
        variation_info["max"] = max(numeric_values)
        variation_info["range"] = max(numeric_values) - min(numeric_values)
        variation_info["avg"] = sum(numeric_values) / len(numeric_values)
    else:
        variation_info["note"] = "Non-numeric values"
    
    return {
        "total_occurrences": len(values),
        "unique_values": len(value_counts),
        "value_counts": value_counts,
        "changes": changes,
        "variation": variation_info,
    }


# ==========================
# REPORT
# ==========================
def write_register_report(stats, plc_ip, out_path):
    lines = []
    lines.append("PLC REGISTER STATISTICS")
    lines.append("=" * 80)
    lines.append(f"PLC IP: {plc_ip}")
    lines.append(f"JSON files scanned: {stats['files']}")
    lines.append("")
    
    lines.append("SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Total packets involving PLC IP: {stats['total_plc_packets']}")
    lines.append(f"Packets with register values (func_code 3/4/6): {stats['packets_with_registers']}")
    lines.append(f"Total unique registers found: {len(stats['register_data'])}")
    lines.append("")
    
    if not stats['register_data']:
        lines.append("No register data found.")
        lines.append("")
    else:
        lines.append("REGISTER DETAILS")
        lines.append("=" * 80)
        lines.append("")
        
        # Sort registers by address
        for reg_addr in sorted(stats['register_data'].keys()):
            values_list = stats['register_data'][reg_addr]
            reg_stats = calculate_register_stats(values_list)
            
            lines.append(f"Register Address: {reg_addr}")
            lines.append("-" * 80)
            lines.append(f"Total occurrences: {reg_stats['total_occurrences']}")
            lines.append(f"Unique values: {reg_stats['unique_values']}")
            lines.append(f"Number of changes: {reg_stats['changes']}")
            lines.append("")
            
            # Variation info
            if "note" in reg_stats['variation']:
                lines.append(f"Variation: {reg_stats['variation']['note']}")
            else:
                var = reg_stats['variation']
                lines.append(f"Variation (numeric):")
                lines.append(f"  Min: {var['min']}")
                lines.append(f"  Max: {var['max']}")
                lines.append(f"  Range: {var['range']}")
                lines.append(f"  Average: {var['avg']:.2f}")
            lines.append("")
            
            # Value frequency
            lines.append("Value frequency:")
            for val, count in reg_stats['value_counts'].most_common():
                lines.append(f"  {val}: {count} times")
            lines.append("")
            lines.append("")
    
    if stats['errors']:
        lines.append("ERRORS")
        lines.append("-" * 80)
        for e in stats['errors']:
            lines.append(e)
        lines.append("")
    
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    PLC_IPS = [
        "132.72.249.116",
        "132.72.249.44",
        "132.72.32.226",
        "132.72.35.161",
    ]
    
    MODBUS_DIR = Path("data/new/modbus")
    
    OUT_DIR = Path("plc_register_reports")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for plc_ip in PLC_IPS:
        output_txt = OUT_DIR / f"plc_registers_{plc_ip.replace('.', '_')}.txt"
        stats = analyze_plc_registers(MODBUS_DIR, plc_ip)
        write_register_report(stats, plc_ip, output_txt)
        print(f"Saved: {output_txt}")