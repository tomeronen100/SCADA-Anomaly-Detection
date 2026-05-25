import json
from pathlib import Path
from collections import Counter

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
# CORE LOGIC
# ==========================
def scan_plc_packets(modbus_dir, plc_ip):
    json_files = list(modbus_dir.rglob("*.json"))

    total = 0
    as_src = 0
    as_dst = 0

    func_all = Counter()
    func_src = Counter()
    func_dst = Counter()

    errors = []

    for fp in json_files:
        try:
            data = json.loads(fp.read_text())
        except Exception as e:
            errors.append(f"{fp}: {e}")
            continue

        for pkt in iter_packets(data):
            src = s(pkt.get("Source_IP"))
            dst = s(pkt.get("Destination_IP"))

            if src != plc_ip and dst != plc_ip:
                continue

            total += 1
            fc = s(pkt.get("func_code"))

            if fc:
                func_all[fc] += 1

            if src == plc_ip:
                as_src += 1
                if fc:
                    func_src[fc] += 1

            if dst == plc_ip:
                as_dst += 1
                if fc:
                    func_dst[fc] += 1

    return {
        "files": len(json_files),
        "total": total,
        "as_src": as_src,
        "as_dst": as_dst,
        "func_all": func_all,
        "func_src": func_src,
        "func_dst": func_dst,
        "errors": errors,
    }


# ==========================
# REPORT
# ==========================
def write_report(stats, plc_ip, out_path):
    def sort_fc(x):
        try:
            return int(x)
        except:
            return 999

    lines = []
    lines.append("PLC MODBUS STATISTICS")
    lines.append("=" * 60)
    lines.append(f"PLC IP: {plc_ip}")
    lines.append(f"JSON files scanned: {stats['files']}")
    lines.append("")
    lines.append(f"Total packets involving PLC: {stats['total']}")
    lines.append(f"Packets as Source_IP       : {stats['as_src']}")
    lines.append(f"Packets as Destination_IP  : {stats['as_dst']}")
    lines.append("")

    def dump(title, counter):
        lines.append(title)
        lines.append("-" * 60)
        if not counter:
            lines.append("N/A\n")
            return
        for k in sorted(counter, key=sort_fc):
            lines.append(f"{k}\t{counter[k]}")
        lines.append("")

    dump("Func codes (all packets)", stats["func_all"])
    dump("Func codes when PLC is Source", stats["func_src"])
    dump("Func codes when PLC is Destination", stats["func_dst"])

    if stats["errors"]:
        lines.append("Errors:")
        for e in stats["errors"]:
            lines.append(e)

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    PLC_IPS = [
        # "132.72.249.116",
        # "132.72.249.44",
        # "132.72.32.226",
        # "132.72.35.161",
        "132.72.249.110"
    ]

    MODBUS_DIR = Path("data/new/modbus")

    # optional: keep all outputs in one folder
    OUT_DIR = Path("plc_reports")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for plc_ip in PLC_IPS:
        output_txt = OUT_DIR / f"plc_stats_{plc_ip.replace('.', '_')}.txt"
        stats = scan_plc_packets(MODBUS_DIR, plc_ip)
        write_report(stats, plc_ip, output_txt)
        print("Saved:", output_txt)







