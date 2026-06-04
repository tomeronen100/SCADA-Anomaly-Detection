import argparse
import os
import re
import statistics
from datetime import datetime, timedelta

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(
        "matplotlib is required to run this script. Install it with `pip install matplotlib`."
    )

FILENAME_RE = re.compile(r"capture_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.pcap$", re.IGNORECASE)
EXPECTED_SECONDS = 2 * 60 * 60
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_start_time(filename):
    match = FILENAME_RE.match(filename)
    if not match:
        return None
    date_part, time_part = match.groups()
    timestamp = f"{date_part} {time_part.replace('-', ':')}"
    return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")


def load_pcap_records(pcap_dir):
    records = []
    for entry in os.listdir(pcap_dir):
        if not entry.lower().endswith(".pcap"):
            continue
        start = parse_start_time(entry)
        if not start:
            continue
        path = os.path.join(pcap_dir, entry)
        if not os.path.isfile(path):
            continue
        size = os.path.getsize(path)
        if size <= 0:
            continue
        records.append({"name": entry, "path": path, "start": start, "size": size})
    records.sort(key=lambda record: record["start"])
    return records


def estimate_rate(records):
    if not records:
        return None
    rates = [record["size"] / EXPECTED_SECONDS for record in records]
    return statistics.median(rates)


def merge_intervals(intervals):
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda item: item[0])
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def format_duration(seconds):
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{hours}h {minutes}m {sec}s"


def make_plots(records, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    bars = [
        (mdates.date2num(record["start"]), record["duration_sec"] / 86400.0)
        for record in records
    ]
    axes[0].broken_barh(bars, (0, 5), facecolors="tab:blue")
    axes[0].set_ylim(0, 6)
    axes[0].set_yticks([])
    axes[0].set_ylabel("Coverage")
    axes[0].set_title("Estimated recording coverage timeline")
    axes[0].grid(axis="x", linestyle="--", alpha=0.4)

    axes[1].plot([record["start"] for record in records], [record["duration_sec"] / 3600.0 for record in records], marker="o", linestyle="-", color="tab:orange")
    axes[1].set_ylabel("Estimated duration (hours)")
    axes[1].set_title("Estimated file duration")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(formatter)

    fig.autofmt_xdate()
    fig.tight_layout()

    coverage_path = os.path.join(output_dir, "recording_coverage.png")
    fig.savefig(coverage_path, dpi=150)
    plt.close(fig)
    return coverage_path


def write_summary(records, output_dir, rate, pcap_dir):
    os.makedirs(output_dir, exist_ok=True)
    summary_csv = os.path.join(output_dir, "recording_summary.csv")
    with open(summary_csv, "w", encoding="utf-8") as out:
        out.write("name,start,estimated_end,estimated_duration_s,size_bytes,estimated_duration_h\n")
        for record in records:
            out.write(
                f"{record['name']},{record['start'].isoformat()},{record['end'].isoformat()},{record['duration_sec']:.1f},{record['size']},{record['duration_sec']/3600.0:.3f}\n"
            )

    info_text = os.path.join(output_dir, "recording_info.txt")
    merged = merge_intervals([(record["start"], record["end"]) for record in records])
    covered_seconds = sum((end - start).total_seconds() for start, end in merged)
    span_seconds = (records[-1]["end"] - records[0]["start"]).total_seconds()
    with open(info_text, "w", encoding="utf-8") as out:
        out.write(f"PCAP directory: {os.path.abspath(pcap_dir)}\n")
        out.write(f"Files analyzed: {len(records)}\n")
        out.write(f"Estimated bytes/sec rate: {rate:.1f}\n")
        out.write(f"Estimated total recorded time: {format_duration(sum(r['duration_sec'] for r in records))}\n")
        out.write(f"Time span from first start to last end: {format_duration(span_seconds)}\n")
        out.write(f"Estimated merged coverage time: {format_duration(covered_seconds)}\n")
        out.write(f"Coverage fraction across span: {covered_seconds/span_seconds*100:.2f}%\n")
        out.write(f"Plot saved to: {os.path.join(output_dir, 'recording_coverage.png')}\n")
        out.write(f"CSV summary saved to: {summary_csv}\n")
    return summary_csv


def main():
    parser = argparse.ArgumentParser(
        description="Analyze .pcap recordings and build a time coverage plot."
    )
    parser.add_argument(
        "--pcap-dir",
        default=r"D:\recording\1-recording\data\new",
        help="Folder containing .pcap capture files.",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Folder to save plots and summary files. Defaults to src/sbus-packets/plots.",
    )
    args = parser.parse_args()

    pcap_dir = os.path.abspath(args.pcap_dir)
    plots_dir = os.path.abspath(args.plots_dir) if args.plots_dir else os.path.join(SCRIPT_DIR, "plots")

    records = load_pcap_records(pcap_dir)
    if not records:
        raise SystemExit(f"No valid .pcap files found in {pcap_dir}")

    rate = estimate_rate(records)
    if rate is None or rate <= 0:
        raise SystemExit("Unable to estimate bytes-per-second rate from file sizes.")

    for record in records:
        record["duration_sec"] = record["size"] / rate
        record["end"] = record["start"] + timedelta(seconds=record["duration_sec"])

    coverage_path = make_plots(records, plots_dir)
    summary_csv = write_summary(records, plots_dir, rate, pcap_dir)

    print(f"Analyzed {len(records)} PCAP files.")
    print(f"Coverage plot saved to: {coverage_path}")
    print(f"Summary CSV saved to: {summary_csv}")


if __name__ == "__main__":
    main()
