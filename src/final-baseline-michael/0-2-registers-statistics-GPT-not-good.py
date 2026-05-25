import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from statistics import pvariance

# ==========================
# CONFIG
# ==========================
FUNC_CODES_KEEP = {3, 4, 6}

# Match keys like:
#   register_12_(uint16)
#   register_12
#   reg_12
#   holding_register_12
REGEX_KEY_ADDR = re.compile(r"(?i)\b(?:holding_)?(?:reg|register)\D*(\d+)\b")

# ==========================
# HELPERS
# ==========================
def iter_packets(obj):
    """Iterate packet dicts whether JSON root is dict or list of dicts."""
    if isinstance(obj, dict):
        yield obj
    elif isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                yield x

def s(x):
    return "" if x is None else str(x).strip()

def try_int(x):
    try:
        if x is None:
            return None
        # handle "3", "3.0", 3, etc.
        return int(float(str(x).strip()))
    except Exception:
        return None

def try_float(x):
    try:
        if x is None:
            return None
        return float(str(x).strip())
    except Exception:
        return None

def extract_registers(pkt):
    """
    Return {addr(int): value(float)} extracted from packet.
    Tries multiple schemas:
      1) pkt["registers"] as dict {addr: value}
      2) pkt["registers"] as list of dicts with addr/value fields
      3) key-per-register fields like "register_12_(uint16)" -> value
      4) func 6 style fields like (address, value) pairs
    """
    regs = {}

    # ---- (1) registers dict ----
    r = pkt.get("registers")
    if isinstance(r, dict):
        for k, v in r.items():
            addr = try_int(k)
            val = try_float(v)
            if addr is not None and val is not None:
                regs[addr] = val
        if regs:
            return regs

    # ---- (2) registers list ----
    if isinstance(r, list):
        for item in r:
            if not isinstance(item, dict):
                continue
            # common names
            addr = (try_int(item.get("address")) or try_int(item.get("addr")) or
                    try_int(item.get("register")) or try_int(item.get("reg")))
            val = (try_float(item.get("value")) or try_float(item.get("val")))
            if addr is not None and val is not None:
                regs[addr] = val
        if regs:
            return regs

    # ---- (3) key-per-register fields ----
    # e.g., "register_12_(uint16)": 345
    for k, v in pkt.items():
        if not isinstance(k, str):
            continue
        m = REGEX_KEY_ADDR.search(k)
        if not m:
            continue
        addr = try_int(m.group(1))
        val = try_float(v)
        if addr is not None and val is not None:
            regs[addr] = val

    if regs:
        return regs

    # ---- (4) func-6 single register write common patterns ----
    # Try a few typical fields:
    #   "register_address" / "register_value"
    #   "address" / "value"
    #   "reg" / "val"
    addr_candidates = ["register_address", "register_addr", "address", "addr", "reg", "register"]
    val_candidates = ["register_value", "value", "val"]

    addr = None
    for ak in addr_candidates:
        addr = try_int(pkt.get(ak))
        if addr is not None:
            break

    val = None
    for vk in val_candidates:
        val = try_float(pkt.get(vk))
        if val is not None:
            break

    if addr is not None and val is not None:
        regs[addr] = val

    return regs

# ==========================
# CORE LOGIC
# ==========================
def scan_plc_register_stats(modbus_dir: Path, plc_ip: str):
    json_files = list(modbus_dir.rglob("*.json"))

    # per-register tracking
    values_by_reg = defaultdict(list)     # reg -> [v1, v2, ...]
    changes_by_reg = Counter()            # reg -> num changes
    last_value_by_reg = {}                # reg -> last value observed

    total_filtered_packets = 0
    filtered_packets_with_regs = 0
    func_counter = Counter()
    errors = []

    for fp in json_files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"{fp}: {e}")
            continue

        for pkt in iter_packets(data):
            src = s(pkt.get("Source_IP"))
            dst = s(pkt.get("Destination_IP"))
            if src != plc_ip and dst != plc_ip:
                continue

            fc = try_int(pkt.get("func_code"))
            if fc not in FUNC_CODES_KEEP:
                continue

            total_filtered_packets += 1
            func_counter[str(fc)] += 1

            regs = extract_registers(pkt)
            if not regs:
                continue

            filtered_packets_with_regs += 1

            # update per-reg stats
            for reg, val in regs.items():
                values_by_reg[reg].append(val)

                if reg in last_value_by_reg:
                    if val != last_value_by_reg[reg]:
                        changes_by_reg[reg] += 1
                last_value_by_reg[reg] = val

    # build final metrics
    reg_stats = {}
    for reg, vals in values_by_reg.items():
        reg_stats[reg] = {
            "n_values": len(vals),
            "n_changes": int(changes_by_reg.get(reg, 0)),
            "variance": float(pvariance(vals)) if len(vals) >= 2 else 0.0,
            "min": float(min(vals)) if vals else None,
            "max": float(max(vals)) if vals else None,
        }

    return {
        "files": len(json_files),
        "plc_ip": plc_ip,
        "total_filtered_packets": total_filtered_packets,
        "filtered_packets_with_regs": filtered_packets_with_regs,
        "unique_registers": len(values_by_reg),
        "func_counter": func_counter,
        "reg_stats": reg_stats,
        "errors": errors,
    }

# ==========================
# REPORT
# ==========================
def write_register_report(stats, out_path: Path):
    plc_ip = stats["plc_ip"]

    def sort_int(x):
        try:
            return int(x)
        except Exception:
            return 10**9

    lines = []
    lines.append("PLC REGISTER STATISTICS (func_code in {3,4,6})")
    lines.append("=" * 80)
    lines.append(f"PLC IP: {plc_ip}")
    lines.append(f"JSON files scanned: {stats['files']}")
    lines.append("")
    lines.append(f"Packets after IP+func filter: {stats['total_filtered_packets']}")
    lines.append(f"Packets with extracted registers: {stats['filtered_packets_with_regs']}")
    lines.append(f"Unique registers seen: {stats['unique_registers']}")
    lines.append("")

    # function code breakdown
    lines.append("Func code breakdown (after filter)")
    lines.append("-" * 80)
    for fc in sorted(stats["func_counter"], key=sort_int):
        lines.append(f"{fc}\t{stats['func_counter'][fc]}")
    lines.append("")

    # per-register table
    lines.append("Per-register stats")
    lines.append("-" * 80)
    lines.append("reg\t#values\t#changes\tvariance\tmin\tmax")
    reg_stats = stats["reg_stats"]
    for reg in sorted(reg_stats.keys()):
        r = reg_stats[reg]
        lines.append(
            f"{reg}\t{r['n_values']}\t{r['n_changes']}\t{r['variance']:.6f}\t{r['min']}\t{r['max']}"
        )
    lines.append("")

    if stats["errors"]:
        lines.append("Errors while reading JSON files:")
        lines.append("-" * 80)
        for e in stats["errors"]:
            lines.append(e)
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")

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

    OUT_DIR = Path("plc_reg_reports")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for plc_ip in PLC_IPS:
        out_txt = OUT_DIR / f"plc_regs_{plc_ip.replace('.', '_')}.txt"
        stats = scan_plc_register_stats(MODBUS_DIR, plc_ip)
        write_register_report(stats, out_txt)
        print("Saved:", out_txt)
