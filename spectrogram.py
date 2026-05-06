"""
HRV Time-Varying Spectrogram Script
-------------------------------------
For each BBI CSV, generates:
  1. A two-panel PNG (raw BBI signal + spectrogram heatmap) per participant per phase
  2. Three summary CSVs (one per phase) with spectral band power per participant

BBI signal is resampled to 4 Hz (standard for HRV spectral analysis) before
computing the spectrogram using scipy's Short-Time Fourier Transform (STFT).

Frequency bands:
  VLF: 0.00 – 0.04 Hz
  LF:  0.04 – 0.15 Hz
  HF:  0.15 – 0.40 Hz

Usage:
    python spectrogram.py
    (will prompt for input folder and output folder)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy import signal, interpolate

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
RESAMPLE_HZ = 4           # standard HRV resampling rate
NFFT = 256                # FFT window size
OVERLAP = 0.75            # fractional overlap between windows
WINDOW_FN = 'hann'        # window function
CMAP = 'viridis'          # colormap
USE_DB = True             # use dB scale for power

# Frequency bands (Hz)
VLF_BAND = (0.00, 0.04)
LF_BAND  = (0.04, 0.15)
HF_BAND  = (0.15, 0.40)

# Phase folder keywords and labels
PHASES = [
    {"keyword": "baseline", "key": "calgary-baseline", "label": "Calgary Baseline"},
    {"keyword": "la paz",   "key": "lapaz-testing",    "label": "La Paz Testing"},
    {"keyword": "return",   "key": "calgary-return",   "label": "Calgary Return"},
]
# ─────────────────────────────────────────────────────────────────────────────


def extract_participant_id(filename: str) -> str:
    """Extract participant ID — first two underscore-separated parts before any space."""
    stem = Path(filename).stem
    # Handle names like '7B4042832_244164fa bbi-calgary-baseline'
    stem = stem.split(" ")[0]  # take only part before first space
    parts = stem.split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else parts[0]


def find_phase_folder(parent_dir: Path, keyword: str):
    """Find a subfolder matching a keyword (case-insensitive)."""
    for folder in parent_dir.iterdir():
        if folder.is_dir() and keyword.lower() in folder.name.lower():
            return folder
    return None


def load_bbi(csv_path: Path):
    """Load and clean BBI CSV."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "timestamp" not in df.columns or "bbi" not in df.columns:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["bbi"] = pd.to_numeric(df["bbi"], errors="coerce")
    df = df.dropna(subset=["bbi"])
    # Remove physiologically implausible values
    df = df[(df["bbi"] > 300) & (df["bbi"] < 2000)]
    # Average duplicate timestamps
    df = df.groupby("timestamp", as_index=False)["bbi"].mean()
    return df


def resample_bbi(df: pd.DataFrame, fs: float = RESAMPLE_HZ):
    """Resample BBI to regular time grid using cubic interpolation."""
    # Convert timestamps to seconds from start
    t = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().values
    rri = df["bbi"].values / 1000.0  # convert ms to seconds

    # Regular time grid
    t_regular = np.arange(t[0], t[-1], 1.0 / fs)

    # Linear interpolation — more stable than cubic with gappy data
    f_interp = interpolate.interp1d(t, rri, kind='linear',
                                     bounds_error=False, fill_value=np.nan)
    rri_regular = f_interp(t_regular)
    # Fill NaNs from gaps with forward/backward fill
    rri_regular = pd.Series(rri_regular).ffill().bfill().values

    return t_regular, rri_regular


def compute_spectrogram(rri_regular: np.ndarray, fs: float = RESAMPLE_HZ):
    """Compute STFT spectrogram."""
    nperseg = NFFT
    noverlap = int(NFFT * OVERLAP)

    freqs, times, Zxx = signal.spectrogram(
        rri_regular,
        fs=fs,
        window=WINDOW_FN,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density',
    )

    power = np.abs(Zxx) ** 2

    if USE_DB:
        power_display = 10 * np.log10(power + 1e-12)
    else:
        power_display = power

    return freqs, times, power, power_display


def band_power(freqs: np.ndarray, power: np.ndarray, band: tuple) -> float:
    """Average power within a frequency band."""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return np.nan
    return np.mean(power[mask, :])


def peak_frequency(freqs: np.ndarray, power: np.ndarray) -> float:
    """Frequency with highest average power."""
    mean_power = np.mean(power, axis=1)
    return freqs[np.argmax(mean_power)]


def make_spectrogram_plot(participant_id: str, phase_label: str,
                           df: pd.DataFrame, t_regular: np.ndarray,
                           rri_regular: np.ndarray, freqs: np.ndarray,
                           times: np.ndarray, power_display: np.ndarray,
                           out_path: Path):
    """Generate and save two-panel spectrogram figure."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={'height_ratios': [1, 2]})

    fig.suptitle(f"{participant_id} — {phase_label}\nHRV Spectrogram",
                 fontsize=13, fontweight='bold')

    # ── Top panel: raw BBI signal ─────────────────────────────────────────────
    t_minutes = t_regular / 60
    times_minutes = times / 60
    # Clip signal to spectrogram time range for aligned x axes
    mask = (t_minutes >= times_minutes[0]) & (t_minutes <= times_minutes[-1])
    ax1.plot(t_minutes[mask], rri_regular[mask] * 1000, color='steelblue', linewidth=0.6, alpha=0.8)
    ax1.set_ylabel("RR Interval (ms)", fontsize=10)
    ax1.set_xlabel("")
    ax1.set_title("Raw RR Interval Signal", fontsize=11)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── Bottom panel: spectrogram ─────────────────────────────────────────────

    # Limit to HRV-relevant frequencies (0 to 0.5 Hz)
    freq_mask = freqs <= 0.5
    freqs_plot = freqs[freq_mask]
    power_plot = power_display[freq_mask, :]

    im = ax2.pcolormesh(times_minutes, freqs_plot, power_plot,
                         cmap=CMAP, shading='gouraud')

    # Frequency band lines
    ax2.axhline(y=VLF_BAND[1], color='white', linewidth=1, linestyle='--', alpha=0.7)
    ax2.axhline(y=LF_BAND[1], color='white', linewidth=1, linestyle='--', alpha=0.7)

    # Band labels inside the plot on the right side
    ax2.text(0.99, (VLF_BAND[0] + VLF_BAND[1]) / 2,
             'VLF', fontsize=8, color='white', va='center', ha='right',
             transform=ax2.get_yaxis_transform(), fontweight='bold')
    ax2.text(0.99, (LF_BAND[0] + LF_BAND[1]) / 2,
             'LF', fontsize=8, color='white', va='center', ha='right',
             transform=ax2.get_yaxis_transform(), fontweight='bold')
    ax2.text(0.99, (HF_BAND[0] + HF_BAND[1]) / 2,
             'HF', fontsize=8, color='white', va='center', ha='right',
             transform=ax2.get_yaxis_transform(), fontweight='bold')

    # Colorbar at the bottom so x axes align between panels
    cbar = plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.15, shrink=0.6)
    cbar.set_label('Power (dB)' if USE_DB else 'Power', fontsize=9)

    ax2.set_ylabel("Frequency (Hz)", fontsize=10)
    ax2.set_xlabel("Time (minutes)", fontsize=10)
    ax2.set_title("Time-Varying Spectrogram", fontsize=11)
    ax2.set_ylim(0, 0.5)

    # Match x-axis of both panels
    ax1.set_xlim(times_minutes[0], times_minutes[-1])
    ax2.set_xlim(times_minutes[0], times_minutes[-1])

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("     HRV Time-Varying Spectrogram Generator")
    print("=" * 60)

    input_folder = input("\nEnter path to input folder (BBI CSVs): ").strip()
    output_folder = input("Enter path to output folder: ").strip()

    input_dir = Path(input_folder)
    output_dir = Path(output_folder)

    if not input_dir.exists():
        print(f"\n✗ Input folder not found: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    # CSVs go in output_dir (top level), plots go in plots_dir

    print(f"\nLooking for phase subfolders...")
    phase_folders = {}
    for phase in PHASES:
        folder = find_phase_folder(input_dir, phase["keyword"])
        if folder is None:
            print(f"  ✗ Could not find subfolder for '{phase['label']}' "
                  f"(looking for '{phase['keyword']}' in folder names)")
            return
        phase_folders[phase["key"]] = (folder, phase["label"])
        n = len(list(folder.rglob("*.csv")))
        print(f"  ✓ {phase['label']} → {folder.name} ({n} files)")

    print(f"\nResample rate: {RESAMPLE_HZ} Hz | NFFT: {NFFT} | Overlap: {OVERLAP*100:.0f}%")
    print(f"Colormap: {CMAP} | Scale: {'dB' if USE_DB else 'linear'}")
    print(f"Output → {output_dir}\n")

    # Accumulate summary rows per phase
    phase_summaries = {phase["key"]: [] for phase in PHASES}

    for phase_key, (phase_folder, phase_label) in phase_folders.items():
        # Search recursively in case CSVs are inside participant subfolders
        csv_files = sorted(phase_folder.rglob("*.csv"))
        print(f"\n── {phase_label} ({len(csv_files)} files) ──")

        for csv_path in csv_files:
            participant_id = extract_participant_id(csv_path.name)
            print(f"  → {csv_path.name}", flush=True)

            # Skip suspiciously large files (likely full timeline, not phase-specific)
            file_size = csv_path.stat().st_size
            if file_size > 50 * 1024 * 1024:  # skip files over 50MB
                print(f"    ⚠ Skipping — file too large ({file_size/1e6:.0f} MB), likely a full timeline file")
                continue

            # Load BBI
            df = load_bbi(csv_path)
            if df is None or df.empty or len(df) < 100:
                print(f"    ✗ Not enough valid data — skipping")
                continue

            # Resample
            try:
                t_regular, rri_regular = resample_bbi(df)
            except Exception as e:
                print(f"    ✗ Resampling failed: {e}")
                continue

            # Compute spectrogram
            try:
                freqs, times, power, power_display = compute_spectrogram(rri_regular)
            except Exception as e:
                print(f"    ✗ Spectrogram failed: {e}")
                continue

            # Save PNG
            participant_dir = plots_dir / participant_id
            participant_dir.mkdir(exist_ok=True)
            out_png = participant_dir / f"{participant_id}_{phase_key}_spectrogram.png"

            try:
                make_spectrogram_plot(participant_id, phase_label, df,
                                       t_regular, rri_regular,
                                       freqs, times, power_display, out_png)
                print(f"    ✓ Plot saved → {out_png.name}", flush=True)
            except Exception as e:
                print(f"    ✗ Plot failed: {e}")
                continue

            # Compute summary stats
            summary = {
                "participant_id": participant_id,
                "mean_vlf_power": band_power(freqs, power, VLF_BAND),
                "mean_lf_power":  band_power(freqs, power, LF_BAND),
                "mean_hf_power":  band_power(freqs, power, HF_BAND),
                "total_power":    band_power(freqs, power, (0.0, 0.5)),
                "lf_hf_ratio":    band_power(freqs, power, LF_BAND) / band_power(freqs, power, HF_BAND),
                "peak_frequency": peak_frequency(freqs, power),
                "duration_minutes": (t_regular[-1] - t_regular[0]) / 60,
                "n_beats": len(df),
            }
            phase_summaries[phase_key].append(summary)

    # Save summary CSVs
    print("\nSaving summary CSVs...")
    phase_labels_map = {
        "calgary-baseline": "calgary_baseline",
        "lapaz-testing":    "lapaz_testing",
        "calgary-return":   "calgary_return",
    }
    for phase_key, rows in phase_summaries.items():
        if rows:
            out_csv = output_dir / f"{phase_labels_map[phase_key]}_spectrogram_data.csv"  # top level
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"  ✓ {out_csv.name} ({len(rows)} participants)")
        else:
            print(f"  ⚠ No data for {phase_key}")

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()