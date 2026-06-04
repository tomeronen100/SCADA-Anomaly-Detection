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

# ── Telegram attribute labels (att field) ────────────────────────────────────
_ATT_LABELS = {
    0x00: 'Request',
    0x01: 'Response',
    0x02: 'ACK/NAK',
}

# ── SBUS command labels (cmd field) ──────────────────────────────────────────
# Source: SAIA Burgess S-Bus Ethernet specification / Wireshark sbus dissector
_CMD_LABELS = {
    0x00: 'Read flag(s)',
    0x01: 'Read input(s)',
    0x02: 'Read flag(s)',
    0x03: 'Read output(s)',
    0x04: 'Read register(s)',
    0x05: 'Read register(s)',
    0x06: 'Read counter(s)',
    0x07: 'Read display register',
    0x08: 'Write flag(s)',
    0x09: 'Write output(s)',
    0x0a: 'Write counter flag(s)',
    0x0b: 'Write flag(s)',
    0x0c: 'Write counter(s)',
    0x0d: 'Write display register',
    0x0e: 'Write register(s)',
    0x13: 'Read register(s)',
    0x14: 'Read timer(s)',
    0x17: 'Write timer(s)',
}


def _fmt_att(att_int):
    """Format att as '0x00 Request', '0x01 Response', etc."""
    if att_int is None:
        return None
    label = _ATT_LABELS.get(att_int, 'Unknown')
    return f'0x{att_int:02x} {label}'


def _fmt_cmd(cmd_raw):
    """Format cmd as '0x02 Read flag(s)', '0x0e Write register(s)', etc."""
    if cmd_raw is None:
        return None
    cmd_int = safe_int_from_str(cmd_raw)
    if cmd_int is None:
        return str(cmd_raw)
    label = _CMD_LABELS.get(cmd_int, 'Unknown')
    return f'0x{cmd_int:02x} {label}'


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


def _get_udp_bytes(packet):
    """Return UDP/TCP payload as a flat list of ints (one per byte), or None."""
    raw = None
    if hasattr(packet, 'udp') and hasattr(packet.udp, 'payload'):
        raw = packet.udp.payload
    elif hasattr(packet, 'tcp') and hasattr(packet.tcp, 'payload'):
        raw = packet.tcp.payload
    elif hasattr(packet, 'data') and hasattr(packet.data, 'data'):
        raw = packet.data.data
    if raw is None:
        return None
    b = parse_numeric_vector(raw)
    return b if b else None


def _data_from_sbus_layer(sbus):
    """Try all known Wireshark SBUS dissector field names for data values."""
    for fname in ('val32', 'bin_data', 'data_32', 'byte_value', 'data_byte', 'value', 'data'):
        raw = _sbus_get(sbus, fname)
        if raw is None:
            continue
        vals = []
        for item in (raw if isinstance(raw, list) else [raw]):
            vals.extend(parse_numeric_vector(item))
        if vals:
            return vals
    return []


def _search_nested(d, names):
    """
    Recursively search a tshark field dict (possibly nested) for a field
    whose dotted-name suffix matches any of `names`.

    tshark JSON nests fields that live inside a Wireshark sub-tree under a
    plain-string key (e.g. "Ether-S-Bus header" → dict of sbus.* fields).
    A flat search by suffix therefore misses them; recursion finds them.
    """
    for key, val in d.items():
        if isinstance(val, dict):
            result = _search_nested(val, names)
            if result is not None:
                return result
        elif val is not None:
            suffix = key.split('.')[-1]
            if suffix in names:
                return val
    return None


def _sbus_get(sbus, *names):
    """
    Robustly read a field from a pyshark sbus layer.
    Tries direct attribute access first (fast path), then falls back to a
    recursive search of _all_fields so fields inside Wireshark sub-trees
    (e.g. 'sbus.sequence' inside 'Ether-S-Bus header') are still found.
    """
    for name in names:
        v = getattr(sbus, name, None)
        if v is not None:
            return v
    raw = getattr(sbus, '_all_fields', None) or {}
    return _search_nested(raw, frozenset(names))


# Commands that write binary flags (not 32-bit registers)
_FLAG_WRITE_CMDS = {0x0b, 0x0a}  # Write flag(s), Write counter flag(s)


def _data_from_raw_request(packet, n_values, cmd_raw=None):
    """
    Fallback: slice raw UDP bytes for a request packet (att=0x00).

    Write register (0x0e) layout:
      [0:4] len | [4] ver | [5] type | [6:8] seq | [8] att | [9] dst | [10] cmd
      [11] wcount_raw | [12:14] base_addr | [14 : 14+n*4] 32-bit values | [-2:] checksum

    Write flag (0x0b) layout:
      same header ... | [12:14] base_addr_IOF | [14] FIO_count | [15 : -2] binary bytes | [-2:] checksum
    """
    b = _get_udp_bytes(packet)
    if b is None or len(b) < 12:
        return []

    cmd_int = safe_int_from_str(cmd_raw)

    if cmd_int in _FLAG_WRITE_CMDS:
        # Binary flag write: data starts after FIO_count byte at offset 14
        data_bytes = b[15:len(b) - 2]
        return list(data_bytes) if data_bytes else []

    # Default: 32-bit register write
    if not n_values or n_values <= 0:
        return []
    result = []
    for i in range(n_values):
        start = 14 + i * 4
        end = start + 4
        if end > len(b) - 2:
            break
        val = (b[start] << 24) | (b[start+1] << 16) | (b[start+2] << 8) | b[start+3]
        result.append(val)
    return result


def _data_from_raw_response(packet):
    """
    Fallback: slice raw UDP bytes for a response packet (att=0x01).
    Ether-S-Bus response layout in the UDP payload:
      [0:4]  length  [4] version  [5] type  [6:8] sequence
      [8] att=0x01
      [9 : total_len-2]  data bytes (32-bit values or binary flags)
      [-2:]  checksum
    Data is interpreted as 32-bit big-endian values when byte count is a multiple of 4,
    otherwise returned as raw bytes (e.g. for 1-byte binary flag responses).
    """
    b = _get_udp_bytes(packet)
    if b is None or len(b) < 12:
        return []
    data_bytes = b[9:len(b) - 2]
    if not data_bytes:
        return []
    if len(data_bytes) % 4 == 0:
        result = []
        for i in range(0, len(data_bytes), 4):
            val = (data_bytes[i] << 24) | (data_bytes[i+1] << 16) | (data_bytes[i+2] << 8) | data_bytes[i+3]
            result.append(val)
        return result
    # binary / flag bytes — return as-is
    return list(data_bytes)


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
    # -2 enables tshark two-pass analysis so that forward-reference "generated"
    # fields (e.g. "Response in frame nr." on a request packet) are populated.
    # Without -2, tshark fills in only backward references (response→request)
    # because it hasn't seen the response yet when processing the request.
    cap = pyshark.FileCapture(
        pcap_path,
        display_filter='sbus',
        use_json=True,
        keep_packets=False,
        eventloop=loop,
        custom_parameters=['-2'],
    )

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

            # All field reads go through _sbus_get which searches both pyshark
            # attributes AND _all_fields, catching fields nested inside sub-trees
            # (e.g. "Ether-S-Bus header") that plain getattr misses.
            att = safe_int_from_str(
                _sbus_get(sbus, 'att', 'telegram_attribute', 'attribut')
            )

            cmd_raw = None
            response_time_ms = None
            request_in_frame = None       # (responses only) frame of the matching request
            response_in_frame = None      # (requests only)  frame of the matching response
            first_request_in_frame = None # (repeated requests) frame of the original attempt

            # ── Request (att == 0x00) ────────────────────────────────────────
            if att == 0x00:
                cmd_raw = _sbus_get(sbus, 'cmd', 'command')
                base_address = safe_int_from_str(
                    _sbus_get(sbus, 'addr_RTC', 'addr_IOF', 'destination', 'addr')
                )
                r_count = safe_int_from_str(
                    _sbus_get(sbus, 'wcount_calculated', 'rcount_calculated',
                              'rcount', 'fio_count', 'wcount')
                )
                response_in_frame = safe_int_from_str(
                    _sbus_get(sbus, 'response_in', 'response_frame', 'resp_frame')
                )
                first_request_in_frame = safe_int_from_str(
                    _sbus_get(sbus, 'first_request_in', 'first_request_frame',
                              'first_req_frame', 'firstrespin')
                )

                data_vector = _data_from_sbus_layer(sbus)
                if not data_vector:
                    data_vector = _data_from_raw_request(packet, r_count, cmd_raw=cmd_raw)

            # ── Response (att == 0x01) ───────────────────────────────────────
            elif att == 0x01:
                base_address = None
                r_count = None
                rt = _sbus_get(sbus, 'response_time', 'resp_time')
                try:
                    response_time_ms = float(rt) * 1000.0 if rt is not None else None
                except Exception:
                    response_time_ms = safe_int_from_str(rt)
                request_in_frame = safe_int_from_str(
                    _sbus_get(sbus, 'response_to', 'request_in', 'req_frame')
                )

                data_vector = _data_from_sbus_layer(sbus)
                if not data_vector:
                    data_vector = _data_from_raw_response(packet)

            # ── ACK / NAK or unknown ─────────────────────────────────────────
            else:
                cmd_raw = _sbus_get(sbus, 'cmd', 'command')
                base_address = None
                r_count = None
                data_vector = []

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
                # att: telegram direction — '0x00 Request', '0x01 Response', '0x02 ACK/NAK'
                'att': _fmt_att(att),
                # cmd: SBUS command with label — '0x02 Read flag(s)', '0x0e Write register(s)', …
                'cmd': _fmt_cmd(cmd_raw),
                # response_time_ms: PLC reply latency — only set for response packets
                'response_time_ms': response_time_ms,
                # request_in_frame: (responses) pckt_id of the matching request
                'request_in_frame': request_in_frame,
                # response_in_frame: (requests) pckt_id of the matching response
                'response_in_frame': response_in_frame,
                # first_request_in_frame: (repeated requests) pckt_id of the original attempt
                'first_request_in_frame': first_request_in_frame,
                # base_address: PLC register / flag / I-O base address accessed
                'base_address': base_address,
                # r_count: number of 32-bit register values or flags being accessed
                'r_count': r_count,
                'data_vector': data_vector,
            }

            rows.append(row)

        except Exception as e:
            # skip malformed packet but continue processing
            print(f"Skipping packet due to parse error: {e}")
            continue

    cap.close()
    import gc; gc.collect()

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
    import gc
    import glob

    parser = argparse.ArgumentParser(description='Parse S-Bus packets from a PCAP directory and export one Parquet per file')
    parser.add_argument('--pcapdir', default=r"D:\recording\1-recording\data\new",
                        help='Directory containing .pcap files to process')
    parser.add_argument('--outdir', default=r"d:\tomer\SCADA-Anomaly-Detection\src\sbus-packets\data")
    args = parser.parse_args()

    pcap_files = sorted(glob.glob(os.path.join(args.pcapdir, '*.pcap')))
    if not pcap_files:
        print(f"No .pcap files found in: {args.pcapdir}")
    else:
        print(f"Found {len(pcap_files)} pcap file(s). Starting...")
        for i, pcap_path in enumerate(pcap_files, 1):
            print(f"[{i}/{len(pcap_files)}] {os.path.basename(pcap_path)}")
            try:
                parse_sbus_pcap(pcap_path, args.outdir)
            except Exception as e:
                print(f"  ERROR: {e}")
            gc.collect()
        print('All done.')
