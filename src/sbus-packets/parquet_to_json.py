"""Convert a Parquet S-Bus output file into JSON for easy reading."""

import argparse
import os
import pandas as pd


def convert_parquet_to_json(parquet_path, json_path, orient='records', indent=4):
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    df.to_json(json_path, orient=orient, indent=indent, force_ascii=False)
    return json_path


def main():
    parser = argparse.ArgumentParser(
        description='Read a Parquet file and export it to JSON for easy inspection.'
    )
    parser.add_argument(
        '--parquet',
        default=r'd:\tomer\SCADA-Anomaly-Detection\src\sbus-packets\data\capture_2026-03-26_13-50-13.sbus.parquet',
        help='Path to the input Parquet file.',
    )
    parser.add_argument(
        '--json',
        default=r'd:\tomer\SCADA-Anomaly-Detection\src\sbus-packets\data\capture_2026-03-26_13-50-13.sbus.json',
        help='Path to the output JSON file.',
    )
    parser.add_argument(
        '--orient',
        default='records',
        choices=['split', 'records', 'index', 'columns', 'values', 'table'],
        help='JSON orientation for pandas.to_json.',
    )
    parser.add_argument(
        '--indent',
        type=int,
        default=4,
        help='Indentation level for the JSON output.',
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    output_path = convert_parquet_to_json(args.parquet, args.json, orient=args.orient, indent=args.indent)
    print(f'Wrote JSON to: {output_path}')


if __name__ == '__main__':
    main()
