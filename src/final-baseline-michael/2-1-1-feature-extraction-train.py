# feature_extract_tables.py

from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime


def _ensure_numeric_timestamp(ts: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(ts):
        return ts.astype(float)

    dt = pd.to_datetime(ts, errors="coerce", utc=True)
    if dt.isna().any():
        raise ValueError("Timestamp column contains unparsable values.")
    return (dt.view("int64") / 1e9).astype(float)


def compute_stats(df: pd.DataFrame, cols: list[str]) -> dict:
    stats = {}
    for c in cols:
        s = df[c].astype(float)
        stats[c] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
        }
    return stats


def build_tables(
    in_csv: str,
    out_train_dir: str = "train_data",
    out_results_dir: str = "results",
):
    in_path = Path(in_csv)
    prefix = in_path.stem  # <<< prefix from input file name

    train_dir = Path(out_train_dir)
    results_dir = Path(out_results_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / f"{prefix}__feature_extract_log.txt"

    logs = []

    def log(msg: str):
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{t}] {msg}"
        logs.append(line)
        print(line)

    log("=== Feature extraction started ===")
    log(f"Input CSV: {in_path.resolve()}")

    df = pd.read_csv(in_path)
    log(f"Loaded dataframe shape: {df.shape}")

    # Column detection
    ts_col = "Timestamp" if "Timestamp" in df.columns else df.columns[0]
    inter_col = "inter_arrival_time" if "inter_arrival_time" in df.columns else df.columns[-1]
    reg_cols = [c for c in df.columns if c.startswith("Register_")]
    if not reg_cols:
        reg_cols = list(df.columns[1:-1])

    log(f"Timestamp column: {ts_col}")
    log(f"Inter-arrival column: {inter_col}")
    log(f"Register columns count: {len(reg_cols)}")

    ts = _ensure_numeric_timestamp(df[ts_col])
    dt = ts.diff().fillna(0.0).to_numpy()

    if np.any(dt < 0):
        log("WARNING: timestamps not sorted — sorting by timestamp")
        order = np.argsort(ts.to_numpy())
        df = df.iloc[order].reset_index(drop=True)
        ts = _ensure_numeric_timestamp(df[ts_col])
        dt = ts.diff().fillna(0.0).to_numpy()

    regs = df[reg_cols]
    inter = df[inter_col]

    # --------------------------------------------------
    # TABLE 1 — time + values
    # --------------------------------------------------
    table1 = pd.concat([inter, regs], axis=1)
    p1 = train_dir / f"{prefix}__table_1_time_and_values.csv"
    table1.to_csv(p1, index=False)
    log(f"Saved {p1} | shape={table1.shape}")

    # --------------------------------------------------
    # TABLE 2 — time + per-register duration
    # --------------------------------------------------
    reg_vals = regs.to_numpy()
    n_rows, n_regs = reg_vals.shape

    changed = np.zeros((n_rows, n_regs), dtype=bool)
    changed[1:] = reg_vals[1:] != reg_vals[:-1]
    changed[0] = True

    durations = np.zeros((n_rows, n_regs))
    for i in range(1, n_rows):
        durations[i] = np.where(changed[i], 0.0, durations[i - 1] + dt[i])

    duration_cols = [f"{c}_time" for c in reg_cols]
    table2 = pd.concat(
        [inter, regs, pd.DataFrame(durations, columns=duration_cols)],
        axis=1,
    )
    p2 = train_dir / f"{prefix}__table_2_time_and_duration.csv"
    table2.to_csv(p2, index=False)
    log(f"Saved {p2} | shape={table2.shape}")

    # --------------------------------------------------
    # TABLE 3 — time + global state duration
    # --------------------------------------------------
    any_change = changed.any(axis=1)
    time_in_state = np.zeros(n_rows)
    for i in range(1, n_rows):
        time_in_state[i] = 0.0 if any_change[i] else time_in_state[i - 1] + dt[i]

    table3 = pd.concat(
        [inter, regs, pd.Series(time_in_state, name="time_in_state")],
        axis=1,
    )
    p3 = train_dir / f"{prefix}__table_3_time_and_state.csv"
    table3.to_csv(p3, index=False)
    log(f"Saved {p3} | shape={table3.shape}")


    # --------------------------------------------------
    # STATISTICS (new features only)
    # --------------------------------------------------
    log("=== Feature Statistics (NEW COLUMNS) ===")

    # Table 2: per-register duration stats
    duration_feature_cols = duration_cols
    stats_t2 = compute_stats(table2, duration_feature_cols)

    log("[Table 2] Per-register duration features")
    for col, st in stats_t2.items():
        log(f"{col}: min={st['min']:.6f}, max={st['max']:.6f}, mean={st['mean']:.6f}")

    # Table 3: global time_in_state
    stats_t3 = compute_stats(table3, ["time_in_state"])

    log("[Table 3] Global state feature")
    for col, st in stats_t3.items():
        log(f"{col}: min={st['min']:.6f}, max={st['max']:.6f}, mean={st['mean']:.6f}")


    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------
    log("=== Summary ===")
    log(f"Input shape: {df.shape}")
    log(f"table_1_time_and_values: {table1.shape}")
    log(f"table_2_time_and_duration: {table2.shape}")
    log(f"table_3_time_and_state: {table3.shape}")
    log("=== Feature extraction finished ===")

    log_path.write_text("\n".join(logs) + "\n", encoding="utf-8")
    print(f"\nLog saved to: {log_path.resolve()}")


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    build_tables(in_csv="data/final_226_snd_train.csv")
    # build_tables(in_csv="data/final_161_snd_train.csv")
