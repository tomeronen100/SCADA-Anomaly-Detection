import re
from pathlib import Path
import matplotlib.pyplot as plt

# ==========================
# PARSING
# ==========================
def parse_register_report(txt_path):
    """
    Parse a PLC register report text file.
    Returns dict: {reg_addr: {'occurrences': int, 'unique_values': int, 'changes': int, 'variance': float}}
    """
    text = Path(txt_path).read_text(encoding="utf-8")
    
    registers = {}
    
    # Split by register sections
    # Pattern: "Register Address: <number>"
    sections = re.split(r'Register Address: (\d+)', text)
    
    # sections[0] is header, then alternating (reg_num, content)
    for i in range(1, len(sections), 2):
        if i + 1 >= len(sections):
            break
            
        reg_addr = int(sections[i])
        content = sections[i + 1]
        
        # Extract statistics
        occurrences = None
        unique_values = None
        changes = None
        variance = None
        
        # Total occurrences
        match = re.search(r'Total occurrences:\s*(\d+)', content)
        if match:
            occurrences = int(match.group(1))
        
        # Unique values
        match = re.search(r'Unique values:\s*(\d+)', content)
        if match:
            unique_values = int(match.group(1))
        
        # Number of changes
        match = re.search(r'Number of changes:\s*(\d+)', content)
        if match:
            changes = int(match.group(1))
        
        # Variance (Range)
        match = re.search(r'Range:\s*([\d.]+)', content)
        if match:
            variance = float(match.group(1))
        
        if occurrences is not None:
            registers[reg_addr] = {
                'occurrences': occurrences,
                'unique_values': unique_values if unique_values is not None else 0,
                'changes': changes if changes is not None else 0,
                'variance': variance if variance is not None else 0.0,
            }
    
    return registers


# ==========================
# PLOTTING
# ==========================
def create_plots(registers, plc_ip, output_dir):
    """
    Create 3 plots for a PLC:
    1. Register Number vs Number of Values (unique values)
    2. Register Number vs Number of Changes
    3. Register Number vs Variance (Range)
    """
    if not registers:
        print(f"No register data to plot for {plc_ip}")
        return
    
    # Sort by register address
    sorted_regs = sorted(registers.items())
    reg_addrs = [r[0] for r in sorted_regs]
    
    occurrences = [r[1]['occurrences'] for r in sorted_regs]
    unique_values = [r[1]['unique_values'] for r in sorted_regs]
    changes = [r[1]['changes'] for r in sorted_regs]
    variances = [r[1]['variance'] for r in sorted_regs]
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plc_name = plc_ip.replace('.', '_')
    
    # Plot 1: Register Number vs Number of Unique Values
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(reg_addrs)), unique_values, color='steelblue', alpha=0.7)
    plt.xlabel('Register Address', fontsize=12)
    plt.ylabel('Number of Unique Values', fontsize=12)
    plt.title(f'PLC {plc_ip} - Unique Values per Register', fontsize=14, fontweight='bold')
    plt.xticks(range(len(reg_addrs)), reg_addrs, rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"{plc_name}_unique_values.png", dpi=150)
    plt.close()
    
    # Plot 2: Register Number vs Number of Changes
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(reg_addrs)), changes, color='coral', alpha=0.7)
    plt.xlabel('Register Address', fontsize=12)
    plt.ylabel('Number of Changes', fontsize=12)
    plt.title(f'PLC {plc_ip} - Changes per Register', fontsize=14, fontweight='bold')
    plt.xticks(range(len(reg_addrs)), reg_addrs, rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"{plc_name}_changes.png", dpi=150)
    plt.close()
    
    # Plot 3: Register Number vs Variance (Range)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(reg_addrs)), variances, color='mediumseagreen', alpha=0.7)
    plt.xlabel('Register Address', fontsize=12)
    plt.ylabel('Variance (Range)', fontsize=12)
    plt.title(f'PLC {plc_ip} - Variance per Register', fontsize=14, fontweight='bold')
    plt.xticks(range(len(reg_addrs)), reg_addrs, rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"{plc_name}_variance.png", dpi=150)
    plt.close()
    
    print(f"Created 3 plots for {plc_ip}")


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    REPORT_DIR = Path("plc_register_reports")
    GRAPH_DIR = Path("graphs")
    
    # Get all PLC report files
    report_files = list(REPORT_DIR.glob("plc_registers_*.txt"))
    
    if not report_files:
        print(f"No report files found in {REPORT_DIR}")
    
    for report_file in report_files:
        # Extract PLC IP from filename
        # Format: plc_registers_132_72_32_226.txt
        filename = report_file.stem  # plc_registers_132_72_32_226
        plc_ip = filename.replace("plc_registers_", "").replace("_", ".")
        
        print(f"Processing {report_file.name}...")
        
        # Parse the report
        registers = parse_register_report(report_file)
        
        # Create plots
        create_plots(registers, plc_ip, GRAPH_DIR)
    
    print(f"\nAll plots saved to {GRAPH_DIR}/")