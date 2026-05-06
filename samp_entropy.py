"""
Sample Entropy Script — NeuroKit2
-----------------------------------
Computes sample entropy using NeuroKit2 (faster than hrvanalysis).
Runs sequentially to keep your laptop alive.

Saves separate sampen CSVs: <userID>_<window>hr_sampen.csv
Each CSV has: window_start, window_end, window_number, n_beats, sampen

Run this AFTER batch_hrv_analysis.py.

Usage:
    python add_sampen_nk2.py
"""

import warnings
import pandas as pd
import numpy as np
from pathlib import Path
import neurokit2 as nk

warnings.filterwarnings("ignore")

MIN_BEATS_PER_WINDOW = 30


def extract_user_id(filename: str) -> str:
    stem = Path(filename).stem
    # Handle names with spaces like '7B4042832_244164fa bbi-calgary-baseline'
    stem = stem.split(" ")[0]
    if stem.endswith("_bbi"):
        stem = stem[:-4]
    parts = stem.split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else parts[0]


def get_valid_windows(df: pd.DataFrame, delta: pd.Timedelta) -> list:
    """Smart gap skipping — only visits windows that have data."""
    timestamps = df["timestamp"].values
    delta_ns = np.timedelta64(int(delta.total_seconds() * 1e9), 'ns')
    windows = []
    i = 0
    while i < len(timestamps):
        win_start = timestamps[i]
        win_end = win_start + delta_ns
        j = np.searchsorted(timestamps, win_end, side='left')
        if j - i >= MIN_BEATS_PER_WINDOW:
            windows.append((pd.Timestamp(win_start), pd.Timestamp(win_end), i, j))
        i = j if j > i else i + 1
    return windows


def compute_sampen(rri: list) -> float:
    """Compute sample entropy using NeuroKit2."""
    sampen_val, _ = nk.entropy_sample(rri, dimension=2, delay=1)
    return sampen_val


def main():
    print("=" * 60)
    print("   Sample Entropy (NeuroKit2) — Separate Output Files")
    print("=" * 60)

    bbi_folder = input("\nEnter path to BBI input folder: ").strip()
    output_folder = input("Enter path to output folder (for sampen CSVs): ").strip()

    while True:
        try:
            hours = int(input("Enter time window in hours (must match what you used before): ").strip())
            if hours > 0:
                break
        except ValueError:
            pass
        print("  ✗ Please enter a positive whole number of hours.")

    bbi_dir = Path(bbi_folder)
    output_dir = Path(output_folder)

    if not bbi_dir.exists():
        print(f"\n✗ BBI folder not found: {bbi_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Search recursively in case CSVs are inside subfolders
    bbi_files = sorted(bbi_dir.rglob("*.csv"))
    if not bbi_files:
        print(f"\n✗ No CSV files found in: {bbi_dir}")
        return

    print(f"\nFound {len(bbi_files)} BBI file(s)")
    print(f"Window size: {hours} hour(s)")
    print(f"Using NeuroKit2 for sample entropy")

    window_label = f"{hours}hr"
    delta = pd.Timedelta(hours=hours)

    for bbi_path in bbi_files:
        user_id = extract_user_id(bbi_path.name)
        print(f"\n  → {bbi_path.name}", flush=True)

        # Load BBI
        df = pd.read_csv(bbi_path)
        df.columns = [c.strip().lower() for c in df.columns]

        if "timestamp" not in df.columns or "bbi" not in df.columns:
            print(f"    ✗ Skipping — unexpected columns: {list(df.columns)}")
            continue

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["bbi"] = pd.to_numeric(df["bbi"], errors="coerce")
        df = df.dropna(subset=["bbi"])

        if df.empty:
            print(f"    ✗ Skipping — no valid beats")
            continue

        print(f"    Loaded {len(df):,} beats  |  "
              f"{df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}",
              flush=True)

        windows = get_valid_windows(df, delta)

        if not windows:
            print(f"    ✗ No valid windows found")
            continue

        rows = []
        for idx, (win_start, win_end, i, j) in enumerate(windows):
            chunk = df["bbi"].iloc[i:j].tolist()
            row = {
                "window_start": win_start.isoformat(),
                "window_end": win_end.isoformat(),
                "window_number": idx,
                "n_beats": len(chunk),
                "sampen": None
            }
            try:
                row["sampen"] = compute_sampen(chunk)
            except Exception as e:
                row["sampen_error"] = str(e)

            rows.append(row)

            if (idx + 1) % 10 == 0:
                print(f"    {idx + 1}/{len(windows)} windows done...", flush=True)

        out_df = pd.DataFrame(rows)
        out_filename = f"{user_id}_{window_label}_sampen.csv"
        out_path = output_dir / out_filename
        out_df.to_csv(out_path, index=False)
        print(f"    ✓ {len(rows)} windows → {out_filename}", flush=True)

    print("\n" + "=" * 60)
    print("  Done! All sampen CSVs saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()