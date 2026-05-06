"""
Batch HRV Analysis Script (No SampEn)
--------------------------------------
Runs all HRV features EXCEPT sample entropy in parallel.
Fast and laptop-friendly.

Run add_sampen.py afterwards to add sample entropy to the output CSVs.

Output files: <userID>_<window>hr_hrv.csv
"""

import warnings
import multiprocessing
import pandas as pd
import numpy as np
from pathlib import Path
from hrvanalysis.extract_features import (
    get_time_domain_features,
    get_frequency_domain_features,
    get_geometrical_features,
    get_poincare_plot_features,
    get_csi_cvi_features,
)

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
MIN_BEATS_PER_WINDOW = 30
FREQ_METHOD = "welch"
N_WORKERS = 2  # keep low to avoid crashing laptop
# ─────────────────────────────────────────────────────────────────────────────


def extract_user_id(filename: str) -> str:
    stem = Path(filename).stem
    if stem.endswith("_bbi"):
        stem = stem[:-4]
    return stem


def compute_hrv_for_window(rri: list) -> dict:
    results = {}
    try: results.update(get_time_domain_features(rri))
    except Exception as e: results["time_domain_error"] = str(e)
    try: results.update(get_frequency_domain_features(rri, method=FREQ_METHOD))
    except Exception as e: results["freq_domain_error"] = str(e)
    try: results.update(get_geometrical_features(rri))
    except Exception as e: results["geometrical_error"] = str(e)
    try: results.update(get_poincare_plot_features(rri))
    except Exception as e: results["poincare_error"] = str(e)
    try: results.update(get_csi_cvi_features(rri))
    except Exception as e: results["csi_cvi_error"] = str(e)
    return results


def get_valid_windows(df: pd.DataFrame, delta: pd.Timedelta) -> list:
    timestamps = df["timestamp"].values
    delta_ns = np.timedelta64(int(delta.total_seconds() * 1e9), 'ns')
    windows = []
    i = 0
    while i < len(timestamps):
        win_start = timestamps[i]
        win_end = win_start + delta_ns
        j = np.searchsorted(timestamps, win_end, side='left')
        n_beats = j - i
        if n_beats >= MIN_BEATS_PER_WINDOW:
            windows.append((pd.Timestamp(win_start), pd.Timestamp(win_end), i, j))
        elif n_beats > 0:
            print(f"      ⚠ window at {pd.Timestamp(win_start).date()} skipped "
                  f"({n_beats} beats < {MIN_BEATS_PER_WINDOW} minimum)", flush=True)
        i = j if j > i else i + 1
    return windows


def process_file(args):
    csv_path, output_dir, hours = args
    warnings.filterwarnings("ignore")

    user_id = extract_user_id(csv_path.name)
    print(f"\n  → Processing: {csv_path.name}  (ID: {user_id})", flush=True)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "timestamp" not in df.columns or "bbi" not in df.columns:
        print(f"    ✗ Skipping — expected 'Timestamp' and 'bbi' columns, "
              f"got: {list(df.columns)}", flush=True)
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["bbi"] = pd.to_numeric(df["bbi"], errors="coerce")
    df = df.dropna(subset=["bbi"])

    if df.empty:
        print(f"    ✗ Skipping — no valid beats found after cleaning", flush=True)
        return

    print(f"    Loaded {len(df):,} beats  |  "
          f"{df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}", flush=True)

    window_label = f"{hours}hr"
    delta = pd.Timedelta(hours=hours)
    windows = get_valid_windows(df, delta)

    if not windows:
        print(f"    ✗ {window_label}: no valid windows found", flush=True)
        return

    rows = []
    for wn, (win_start, win_end, i, j) in enumerate(windows):
        chunk = df["bbi"].iloc[i:j].tolist()
        hrv = compute_hrv_for_window(chunk)
        hrv["window_start"] = win_start.isoformat()
        hrv["window_end"] = win_end.isoformat()
        hrv["window_number"] = wn
        hrv["n_beats"] = len(chunk)
        rows.append(hrv)

    out_df = pd.DataFrame(rows)
    front_cols = ["window_start", "window_end", "window_number", "n_beats"]
    other_cols = [c for c in out_df.columns if c not in front_cols]
    out_df = out_df[front_cols + other_cols]

    out_filename = f"{user_id}_{window_label}_hrv.csv"
    out_path = Path(output_dir) / out_filename
    out_df.to_csv(out_path, index=False)
    print(f"    ✓ {window_label}: {len(rows)} windows → {out_filename}", flush=True)


def main():
    print("=" * 60)
    print("   Batch HRV Analysis — No SampEn (Fast Mode)")
    print("=" * 60)

    input_folder = input("\nEnter path to input folder (BBI CSVs): ").strip()
    output_folder = input("Enter path to output folder: ").strip()

    while True:
        try:
            hours = int(input("Enter time window in hours (e.g. 1, 6, 12, 24): ").strip())
            if hours > 0:
                break
        except ValueError:
            pass
        print("  ✗ Please enter a positive whole number of hours.")

    input_dir = Path(input_folder)
    output_dir = Path(output_folder)

    if not input_dir.exists():
        print(f"\n✗ Input folder not found: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        print(f"\n✗ No CSV files found in: {input_dir}")
        return

    print(f"\nFound {len(csv_files)} CSV file(s) in {input_dir}")
    print(f"Window size: {hours} hour(s)")
    print(f"Parallel workers: {N_WORKERS}")
    print(f"Output → {output_dir}\n")
    print("NOTE: SampEn not included — run add_sampen.py afterwards!\n")

    args_list = [(csv_path, str(output_dir), hours) for csv_path in csv_files]

    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        pool.map(process_file, args_list)

    print("\n" + "=" * 60)
    print("  Done! Run add_sampen.py to add sample entropy.")
    print("=" * 60)


if __name__ == "__main__":
    main()