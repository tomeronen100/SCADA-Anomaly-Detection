import json
from collections import Counter
from pathlib import Path
from typing import Any

INPUT_DIR = Path(r"d:\recording\1-recording\data\new\sbus")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
TOP_IPS = 15


def format_counter(counter: Counter, title: str, top_n: int | None = None) -> list[str]:
    lines = ["", title, "" if counter else "  (no values)"]
    if counter:
        width = max(len(str(key)) for key in counter)
        items = counter.most_common(top_n) if top_n is not None else counter.most_common()
        for key, count in items:
            lines.append(f"  {str(key):<{width}}  {count}")
    return lines


def normalize_ip(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def list_json_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    return sorted([path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json"])


def analyze_file(input_path: Path) -> dict:
    with input_path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)

    if not isinstance(records, list):
        raise ValueError("Expected top-level JSON array")

    src_ips = Counter()
    dst_ips = Counter()
    att_values = Counter()
    cmd_values = Counter()
    value_null = 0
    value_present = 0
    value_missing = 0
    first_record = None

    for record in records:
        if not isinstance(record, dict):
            continue

        if first_record is None:
            first_record = record

        src = normalize_ip(record.get("Source_IP"))
        dst = normalize_ip(record.get("Destination_IP"))
        if src:
            src_ips[src] += 1
        if dst:
            dst_ips[dst] += 1

        if "att" in record:
            att_values[str(record.get("att"))] += 1
        else:
            att_values["<missing>"] += 1

        if "cmd" in record:
            cmd_values[str(record.get("cmd"))] += 1
        else:
            cmd_values["<missing>"] += 1

        if "value" in record:
            if record["value"] is None:
                value_null += 1
            else:
                value_present += 1
        else:
            value_missing += 1

    unique_ips = set(src_ips) | set(dst_ips)
    return {
        "input_file": input_path,
        "total_packets": len(records),
        "src_ips": src_ips,
        "dst_ips": dst_ips,
        "att_values": att_values,
        "cmd_values": cmd_values,
        "value_null": value_null,
        "value_present": value_present,
        "value_missing": value_missing,
        "unique_ips": len(unique_ips),
        "unique_src_ips": len(src_ips),
        "unique_dst_ips": len(dst_ips),
        "first_record_fields": sorted(first_record.keys()) if first_record is not None else [],
    }


def report_file(data: dict) -> list[str]:
    lines = [
        "=" * 80,
        f"File: {data['input_file'].name}",
        f"Path: {data['input_file']}",
        "=" * 80,
        f"Total packets: {data['total_packets']}",
        f"Unique IPs: {data['unique_ips']}",
        f"Unique source IPs: {data['unique_src_ips']}",
        f"Unique destination IPs: {data['unique_dst_ips']}",
        "",
        "Value presence:",
        f"  Packets with value = null: {data['value_null']}",
        f"  Packets with value present: {data['value_present']}",
        f"  Packets with value missing: {data['value_missing']}",
    ]
    lines += format_counter(data['src_ips'], f"Top {TOP_IPS} source IP counts:", top_n=TOP_IPS)
    lines += format_counter(data['dst_ips'], f"Top {TOP_IPS} destination IP counts:", top_n=TOP_IPS)
    lines += format_counter(data['att_values'], "att value counts:")
    lines += format_counter(data['cmd_values'], "cmd value counts:")
    if data['first_record_fields']:
        lines += ["", "Sample record fields:"]
        lines += [f"  {field}" for field in data['first_record_fields']]
    lines.append("")
    return lines


def main() -> None:
    output_dir = build_output_dir()
    files = list_json_files(INPUT_DIR)
    if not files:
        raise FileNotFoundError(f"No JSON files found in {INPUT_DIR}")

    for input_file in files:
        data = analyze_file(input_file)
        report_lines = [
            f"SBUS analysis report for file: {input_file.name}",
            "",
        ]
        report_lines += report_file(data)

        output_file = output_dir / f"{input_file.stem}_analysis.txt"
        output_file.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"Analysis written to: {output_file}")


if __name__ == "__main__":
    main()
