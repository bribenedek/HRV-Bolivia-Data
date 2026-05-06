"""
Batch HRV Analysis Script
--------------------------
For each BBI CSV in the input folder, runs HRV analysis across
1h, 6h, 12h, and 24h windows and saves results to the output folder.

Output files: <userID>_<window>hr_hrv.csv
Each row = one time window, columns = all HRV parameters + window timestamps.

Usage:
    python batch_hrv_analysis.py
    (will prompt for input and output folder paths)
"""

import os
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from hrvanalysis.extract_features import (
    get_time_domain_features,
    get_frequency_domain_features,
    get_geometrical_features,
    get_poincare_plot_features,
    get_csi_cvi_features,
    get_sampen,
)

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW_HOURS = [1, 6, 12, 24]
MIN_BEATS_PER_WINDOW = 30          # skip windows with too few beats
FREQ_METHOD = "welch"              # welch | lomb | fft
# ─────────────────────────────────────────────────────────────────────────────


def extract_user_id(filename: str) -> str:
    """Strip '_bbi.csv' suffix to get the participant ID stem."""
    stem = Path(filename).stem          # e.g. '7B4042832_244164fa_bbi'
    if stem.endswith("_bbi"):
        stem = stem[:-4]                # → '7B4042832_244164fa'
    return stem


def compute_hrv_for_window(rri: list) -> dict:
    """Run all HRV feature groups on a list of RR intervals (ms)."""
    results = {}

    try:
        results.update(get_time_domain_features(rri))
    except Exception as e:
        results["time_domain_error"] = str(e)

    try:
        results.update(get_frequency_domain_features(rri, method=FREQ_METHOD))
    except Exception as e:
        results["freq_domain_error"] = str(e)

    try:
        results.update(get_geometrical_features(rri))
    except Exception as e:
        results["geometrical_error"] = str(e)

    try:
        results.update(get_poincare_plot_features(rri))
    except Exception as e:
        results["poincare_error"] = str(e)

    try:
        results.update(get_csi_cvi_features(rri))
    except Exception as e:
        results["csi_cvi_error"] = str(e)

    try:
        sampen = get_sampen(rri)
        results.update(sampen)
    except Exception as e:
        results["sampen_error"] = str(e)

    return results


def process_file(csv_path: Path, output_dir: Path):
    """Load one BBI CSV, run HRV for all window sizes, save results."""
    user_id = extract_user_id(csv_path.name)
    print(f"\n  → Processing: {csv_path.name}  (ID: {user_id})")

    # Load and parse
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "timestamp" not in df.columns or "bbi" not in df.columns:
        print(f"    ✗ Skipping — expected 'Timestamp' and 'bbi' columns, got: {list(df.columns)}")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["bbi"] = pd.to_numeric(df["bbi"], errors="coerce")
    df = df.dropna(subset=["bbi"])

    print(f"    Loaded {len(df):,} beats  |  "
          f"{df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")

    for hours in WINDOW_HOURS:
        window_label = f"{hours}hr"
        delta = pd.Timedelta(hours=hours)

        rows = []
        start = df["timestamp"].iloc[0]
        end_of_data = df["timestamp"].iloc[-1]

        window_num = 0
        while start + delta <= end_of_data:
            end = start + delta
            mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
            chunk = df.loc[mask, "bbi"].tolist()

            if len(chunk) >= MIN_BEATS_PER_WINDOW:
                hrv = compute_hrv_for_window(chunk)
                hrv["window_start"] = start.isoformat()
                hrv["window_end"] = end.isoformat()
                hrv["window_number"] = window_num
                hrv["n_beats"] = len(chunk)
                rows.append(hrv)
            else:
                print(f"    ⚠ {window_label} window {window_num} skipped "
                      f"({len(chunk)} beats < {MIN_BEATS_PER_WINDOW} minimum)")

            start = end
            window_num += 1

        if not rows:
            print(f"    ✗ {window_label}: no valid windows found")
            continue

        out_df = pd.DataFrame(rows)

        # Put identifying/timing cols first
        front_cols = ["window_start", "window_end", "window_number", "n_beats"]
        other_cols = [c for c in out_df.columns if c not in front_cols]
        out_df = out_df[front_cols + other_cols]

        out_filename = f"{user_id}_{window_label}_hrv.csv"
        out_path = output_dir / out_filename
        out_df.to_csv(out_path, index=False)
        print(f"    ✓ {window_label}: {len(rows)} windows → {out_filename}")


def main():
    print("=" * 60)
    print("        Batch HRV Analysis — All Windows")
    print("=" * 60)

    input_folder = input("\nEnter path to input folder (BBI CSVs): ").strip()
    output_folder = input("Enter path to output folder: ").strip()

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
    print(f"Window sizes: {WINDOW_HOURS} hours")
    print(f"Frequency method: {FREQ_METHOD}")
    print(f"Output → {output_dir}\n")

    for csv_path in csv_files:
        try:
            process_file(csv_path, output_dir)
        except Exception as e:
            print(f"  ✗ ERROR processing {csv_path.name}: {e}")

    print("\n" + "=" * 60)
    print("  Done! All files processed.")
    print("=" * 60)


if __name__ == "__main__":
    main()