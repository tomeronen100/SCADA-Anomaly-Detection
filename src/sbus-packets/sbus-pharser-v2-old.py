"""
svus-pharser-v2.py

Parse SAIA S-Bus packets from a PCAP using pyshark and export a tabular Parquet file
suitable for LSTM/HTM modeling.

Usage:
    python svus-pharser-v2.py --pcap "D:\recording\1-recording\data\new\capture_2026-03-26_13-50-13.pcap" \
        --outdir "d:\tomer\SCADA-Anomaly-Detection\src\sbus-packets\data"

Dependencies:
    pyshark, pandas, pyarrow (or fastparquet)

"""
import argparse
import os
import re
import json
from datetime import datetime

import pyshark
import asyncio
import pandas as pd


def safe_int_from_str(s):
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return int(s)
    s = str(s).strip()
    if s == "":
        return None
    # Try direct int
    try:
        return int(s, 0)
    except Exception:
        pass
    # Try to extract hex or decimal digits
    m = re.search(r"[-]?[0-9A-Fa-f]+", s)
    if m:
        token = m.group(0)
        # if contains letters A-F assume hex
        if re.search(r"[A-Fa-f]", token):
            try:
                return int(token, 16)
            except Exception:
                pass
        try:
            return int(token, 10)
        except Exception:
            pass
    return None


def parse_numeric_vector(value):
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []

    if ':' in text and re.fullmatch(r'(?:[0-9A-Fa-f]{2}:)+[0-9A-Fa-f]{2}', text):
        return [int(byte, 16) for byte in text.split(':')]

    result = []
    for token in re.findall(r'0x[0-9A-Fa-f]+|[-]?\d+', text):
        try:
            result.append(int(token, 0))
        except Exception:
            try:
                result.append(int(token, 16))
            except Exception:
                continue
    return result


def get_raw_payload_vector(packet):
    raw = None
    if hasattr(packet, 'udp') and hasattr(packet.udp, 'payload'):
        raw = packet.udp.payload
    elif hasattr(packet, 'tcp') and hasattr(packet.tcp, 'payload'):
        raw = packet.tcp.payload
    elif hasattr(packet, 'data') and hasattr(packet.data, 'data'):
        raw = packet.data.data
    return parse_numeric_vector(raw)


def parse_sbus_pcap(pcap_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pcap_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}.sbus.parquet")

    # pyshark requires an asyncio event loop; create one for FileCapture
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    except Exception:
        loop = None
    cap = pyshark.FileCapture(pcap_path, display_filter='sbus', use_json=True, keep_packets=False, eventloop=loop)

    rows = []
    parsed_index = 0

    for packet in cap:
        try:
            # Prepare base fields
            # pckt_id: try packet.number then frame_info.number
            pckt_id = None
            if hasattr(packet, 'number'):
                try:
                    pckt_id = int(packet.number)
                except Exception:
                    pckt_id = safe_int_from_str(getattr(packet, 'number', None))
            if pckt_id is None and hasattr(packet, 'frame_info'):
                try:
                    pckt_id = int(packet.frame_info.number)
                except Exception:
                    pckt_id = safe_int_from_str(getattr(getattr(packet, 'frame_info', None), 'number', None))

            # timestamp as float epoch
            timestamp = None
            if hasattr(packet, 'sniff_timestamp'):
                try:
                    timestamp = float(packet.sniff_timestamp)
                except Exception:
                    # fallback parsing
                    timestamp = float(pd.Timestamp(packet.sniff_timestamp).timestamp())

            # IP layer
            source_ip = packet.ip.src if hasattr(packet, 'ip') else None
            destination_ip = packet.ip.dst if hasattr(packet, 'ip') else None

            # Transport ports
            source_port = None
            destination_port = None
            transport_proto = None
            if hasattr(packet, 'udp'):
                transport_proto = 'UDP'
                try:
                    source_port = int(packet.udp.srcport)
                    destination_port = int(packet.udp.dstport)
                except Exception:
                    source_port = safe_int_from_str(getattr(packet.udp, 'srcport', None))
                    destination_port = safe_int_from_str(getattr(packet.udp, 'dstport', None))
            elif hasattr(packet, 'tcp'):
                transport_proto = 'TCP'
                try:
                    source_port = int(packet.tcp.srcport)
                    destination_port = int(packet.tcp.dstport)
                except Exception:
                    source_port = safe_int_from_str(getattr(packet.tcp, 'srcport', None))
                    destination_port = safe_int_from_str(getattr(packet.tcp, 'dstport', None))

            # S-Bus application layer
            if not hasattr(packet, 'sbus'):
                # skip non-sbus packets (display_filter should ensure this rarely happens)
                continue

            sbus = packet.sbus

            # Extract header fields (map to actual dissector names observed)
            sequence = safe_int_from_str(getattr(sbus, 'sequence', None))
            # dissector exposes attribute as 'att'
            att_raw = getattr(sbus, 'att', None) or getattr(sbus, 'telegram_attribute', None)
            att = safe_int_from_str(att_raw)

            cmd = None
            response_time_ms = None
            request_in_frame = None

            # cmd present for request (att == 0x00)
            if att == 0x00:
                cmd = getattr(sbus, 'cmd', None)
                # base address may be named addr_RTC or addr_IOF or destination
                base_address = safe_int_from_str(getattr(sbus, 'addr_RTC', None) or getattr(sbus, 'addr_IOF', None) or getattr(sbus, 'destination', None))
                r_count = safe_int_from_str(getattr(sbus, 'rcount', None) or getattr(sbus, 'fio_count', None) or getattr(sbus, 'wcount', None))
                data_vector = []

                raw = getattr(sbus, 'data_32', None) or getattr(sbus, 'value', None) or getattr(sbus, 'data', None)
                if raw is not None:
                    if isinstance(raw, list):
                        candidates = raw
                    else:
                        candidates = [raw]
                    for item in candidates:
                        if item is None:
                            continue
                        data_vector.extend(parse_numeric_vector(item))
                if not data_vector:
                    data_vector = get_raw_payload_vector(packet)

            # response (att == 0x01)
            elif att == 0x01:
                cmd = None
                base_address = None
                r_count = None
                # response_time is in seconds (float) in this dissector; convert to ms
                rt = getattr(sbus, 'response_time', None)
                try:
                    response_time_ms = float(rt) * 1000.0 if rt is not None else None
                except Exception:
                    response_time_ms = safe_int_from_str(rt)
                # request frame id field may be named 'response_to' or 'request_in'
                request_in_frame = safe_int_from_str(getattr(sbus, 'response_to', None) or getattr(sbus, 'request_in', None))

                # collect all data_32/value/data instances into a list
                data_vector = []
                raw = getattr(sbus, 'data_32', None) or getattr(sbus, 'value', None) or getattr(sbus, 'data', None)
                if raw is not None:
                    if isinstance(raw, list):
                        candidates = raw
                    else:
                        candidates = [raw]
                    for item in candidates:
                        if item is None:
                            continue
                        data_vector.extend(parse_numeric_vector(item))
                if not data_vector:
                    data_vector = get_raw_payload_vector(packet)

            else:
                # ACK/NAK or other att values: skip detailed extraction but record header
                cmd = getattr(sbus, 'cmd', None)
                base_address = None
                r_count = None
                data_vector = []

            # Only count as parsed if we at least have an attribute (att)
            if att is None:
                continue

            parsed_index += 1

            row = {
                'pckt_id': pckt_id,
                'parsed_index': parsed_index,
                'timestamp': timestamp,
                'source_ip': source_ip,
                'destination_ip': destination_ip,
                'transport_protocol': transport_proto,
                'source_port': source_port,
                'destination_port': destination_port,
                'sequence': sequence,
                'att': att,
                'cmd': cmd,
                'response_time_ms': response_time_ms,
                'request_in_frame': request_in_frame,
                'base_address': base_address,
                'r_count': r_count,
                'data_vector': data_vector,
                'pcap_file': os.path.abspath(pcap_path)
            }

            rows.append(row)

        except Exception as e:
            # skip malformed packet but continue processing
            print(f"Skipping packet due to parse error: {e}")
            continue

    cap.close()

    # Convert to DataFrame and write to Parquet
    if not rows:
        print("No S-Bus packets parsed.")
        return None

    df = pd.DataFrame.from_records(rows)

    # Ensure data_vector is stored as list (object) or explode into columns if needed later
    # Write parquet using pyarrow if available
    try:
        df.to_parquet(out_path, index=False)
    except Exception as e:
        raise RuntimeError("Failed to write Parquet. Ensure pyarrow or fastparquet is installed. Original error: " + str(e))

    print(f"Wrote {len(df)} parsed rows to: {out_path}")
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse S-Bus packets from a PCAP and export Parquet rows')
    parser.add_argument('--pcap', default=r"D:\recording\1-recording\data\new\capture_2026-03-26_13-50-13.pcap")
    parser.add_argument('--outdir', default=r"d:\tomer\SCADA-Anomaly-Detection\src\sbus-packets\data")
    args = parser.parse_args()

    out = parse_sbus_pcap(args.pcap, args.outdir)
    if out:
        print('Done.')
