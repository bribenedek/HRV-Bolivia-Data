"""
Spaghetti Plots — HRV Parameters Over Time by Study Phase
----------------------------------------------------------
Generates one spaghetti plot per HRV parameter with three panels:
Calgary Baseline | La Paz Testing | Calgary Return

Each panel shows one semi-transparent line per participant over time,
with a bold mean line on top. Same y-axis scale across all panels.

Usage:
    python spaghetti_plots.py
"""

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Parameter groups ──────────────────────────────────────────────────────────
TIME_DOMAIN = [
    'mean_nni', 'sdnn', 'sdsd', 'nni_50', 'pnni_50',
    'nni_20', 'pnni_20', 'rmssd', 'median_nni', 'range_nni',
    'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'min_hr', 'std_hr'
]
FREQUENCY_DOMAIN = ['lf', 'hf', 'lf_hf_ratio', 'lfnu', 'hfnu', 'total_power', 'vlf']
GEOMETRICAL = ['triangular_index', 'tinn']
POINCARE = ['sd1', 'sd2', 'ratio_sd2_sd1']
CSI_CVI = ['csi', 'cvi', 'Modified_csi']
ALL_PARAMS = TIME_DOMAIN + FREQUENCY_DOMAIN + GEOMETRICAL + POINCARE + CSI_CVI

# ── Group config ──────────────────────────────────────────────────────────────
GROUPS = [
    {"keyword": "baseline", "label": "Calgary Baseline",  "color": "#2196F3"},
    {"keyword": "la paz",   "label": "La Paz Testing",    "color": "#FF5722"},
    {"keyword": "return",   "label": "Calgary Return",    "color": "#4CAF50"},
]

# ── Plot config ───────────────────────────────────────────────────────────────
DPI = 300
FIGSIZE = (15, 5)
LINE_ALPHA = 0.5
LINE_WIDTH = 1.2
MEAN_LINE_WIDTH = 2.5
OUTLIER_PERCENTILE = 99  # clip y-axis at this percentile to handle outliers
# ─────────────────────────────────────────────────────────────────────────────


def find_group_folder(window_dir: Path, keyword: str):
    for folder in window_dir.iterdir():
        if folder.is_dir() and keyword.lower() in folder.name.lower():
            return folder
    return None


def extract_participant_id(filename: str) -> str:
    parts = Path(filename).stem.split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else parts[0]


def load_group_data(folder: Path) -> pd.DataFrame:
    dfs = []
    for csv_path in sorted(folder.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
            df["participant_id"] = extract_participant_id(csv_path.name)
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠ Could not load {csv_path.name}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def nice_label(param: str) -> str:
    labels = {
        'mean_nni': 'Mean NNI (ms)', 'sdnn': 'SDNN (ms)', 'sdsd': 'SDSD (ms)',
        'nni_50': 'NNI50 (count)', 'pnni_50': 'pNNI50 (%)', 'nni_20': 'NNI20 (count)',
        'pnni_20': 'pNNI20 (%)', 'rmssd': 'RMSSD (ms)', 'median_nni': 'Median NNI (ms)',
        'range_nni': 'Range NNI (ms)', 'cvsd': 'CVSD', 'cvnni': 'CVNNI',
        'mean_hr': 'Mean HR (bpm)', 'max_hr': 'Max HR (bpm)', 'min_hr': 'Min HR (bpm)',
        'std_hr': 'Std HR (bpm)', 'lf': 'LF Power (ms²)', 'hf': 'HF Power (ms²)',
        'lf_hf_ratio': 'LF/HF Ratio', 'lfnu': 'LF (n.u.)', 'hfnu': 'HF (n.u.)',
        'total_power': 'Total Power (ms²)', 'vlf': 'VLF Power (ms²)',
        'triangular_index': 'Triangular Index', 'tinn': 'TINN (ms)',
        'sd1': 'SD1 (ms)', 'sd2': 'SD2 (ms)', 'ratio_sd2_sd1': 'SD2/SD1 Ratio',
        'csi': 'CSI', 'cvi': 'CVI', 'Modified_csi': 'Modified CSI',
    }
    return labels.get(param, param.replace('_', ' ').upper())


def param_group_label(param: str) -> str:
    if param in TIME_DOMAIN: return "Time Domain"
    if param in FREQUENCY_DOMAIN: return "Frequency Domain"
    if param in GEOMETRICAL: return "Geometrical"
    if param in POINCARE: return "Poincaré"
    if param in CSI_CVI: return "CSI/CVI"
    return ""


def get_y_limits(group_data: dict, param: str):
    """Get shared y-axis limits across all groups, clipping extreme outliers."""
    all_vals = []
    for group in GROUPS:
        df = group_data[group["label"]]
        if param in df.columns and not df.empty:
            vals = pd.to_numeric(df[param], errors='coerce').dropna().values
            all_vals.extend(vals)
    if not all_vals:
        return None, None
    all_vals = np.array(all_vals)
    y_min = np.percentile(all_vals, 1)
    y_max = np.percentile(all_vals, OUTLIER_PERCENTILE)
    padding = (y_max - y_min) * 0.1 if y_max != y_min else 1
    return y_min - padding, y_max + padding


def make_spaghetti_plot(param: str, group_data: dict, output_dir: Path):
    """Generate and save one spaghetti plot for a single HRV parameter."""

    # Check if any group has data
    has_data = any(
        param in group_data[g["label"]].columns and not group_data[g["label"]].empty
        for g in GROUPS
    )
    if not has_data:
        print(f"  ⚠ Skipping {param} — no valid data in any group")
        return None

    y_min, y_max = get_y_limits(group_data, param)

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=True)
    fig.suptitle(f"{nice_label(param)} Over Time by Study Phase",
                 fontsize=14, fontweight='bold', y=1.02)

    for ax, group in zip(axes, GROUPS):
        df = group_data[group["label"]]
        color = group["color"]
        label = group["label"]

        ax.set_title(label, fontsize=12, fontweight='bold', color=color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        if param not in df.columns or df.empty:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', color='gray', fontsize=11)
            ax.set_xlabel("Time", fontsize=10)
            continue

        df = df.copy()
        df[param] = pd.to_numeric(df[param], errors='coerce')
        df["window_start"] = pd.to_datetime(df["window_start"], format="ISO8601", utc=True)
        df = df.dropna(subset=[param, "window_start"])

        if df.empty:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', color='gray', fontsize=11)
            continue

        # Average each participant down to one point per day
        df["date"] = df["window_start"].dt.floor("D")
        daily = df.groupby(["participant_id", "date"])[param].mean().reset_index()

        # Plot one line per participant (daily averages)
        for pid, pdata in daily.groupby("participant_id"):
            pdata = pdata.sort_values("date")
            ax.plot(pdata["date"], pdata[param],
                    color=color, alpha=LINE_ALPHA, linewidth=LINE_WIDTH)

        # Mean line across all participants per day
        mean_line = daily.groupby("date")[param].mean().reset_index()
        mean_line = mean_line.sort_values("date")
        ax.plot(mean_line["date"], mean_line[param],
                color=color, linewidth=MEAN_LINE_WIDTH, alpha=0.95,
                label="Mean", zorder=5)

        # Format x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha='right', fontsize=8)
        ax.set_xlabel("Date", fontsize=10)

        # Apply shared y limits
        if y_min is not None:
            ax.set_ylim(y_min, y_max)

        # Participant count
        n = df["participant_id"].nunique()
        ax.text(0.98, 0.98, f"n = {n}", transform=ax.transAxes,
                fontsize=9, color='gray', ha='right', va='top')

    # Shared y label on leftmost panel
    axes[0].set_ylabel(nice_label(param), fontsize=11)

    # Category label
    cat = param_group_label(param)
    fig.text(0.99, 0.99, cat, fontsize=9, color='gray',
             ha='right', va='top', style='italic')

    # Mean line legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=LINE_WIDTH, alpha=LINE_ALPHA, label='Participant'),
        Line2D([0], [0], color='gray', linewidth=MEAN_LINE_WIDTH, label='Mean'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    out_path = output_dir / f"{param}_spaghetti.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    return out_path


def main():
    print("=" * 60)
    print("     HRV Spaghetti Plots — By Study Phase")
    print("=" * 60)

    window_folder = input("\nEnter path to window folder (e.g. your 1hr or 6hr folder): ").strip()
    output_folder = input("Enter path to output folder for plots: ").strip()

    window_dir = Path(window_folder)
    output_dir = Path(output_folder)

    if not window_dir.exists():
        print(f"\n✗ Window folder not found: {window_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLooking for group subfolders...")
    group_data = {}
    for group in GROUPS:
        folder = find_group_folder(window_dir, group["keyword"])
        if folder is None:
            print(f"  ✗ Could not find subfolder for '{group['label']}' "
                  f"(looking for '{group['keyword']}' in folder names)")
            return
        print(f"  ✓ {group['label']} → {folder.name}")
        df = load_group_data(folder)
        n_participants = df["participant_id"].nunique() if not df.empty else 0
        print(f"    Loaded {n_participants} participants, {len(df):,} total windows")
        group_data[group["label"]] = df

    print(f"\nGenerating {len(ALL_PARAMS)} spaghetti plots...")
    print(f"Output → {output_dir}\n")

    success = 0
    for i, param in enumerate(ALL_PARAMS):
        try:
            out_path = make_spaghetti_plot(param, group_data, output_dir)
            if out_path:
                print(f"  ✓ [{i+1}/{len(ALL_PARAMS)}] {param}")
                success += 1
        except Exception as e:
            print(f"  ✗ [{i+1}/{len(ALL_PARAMS)}] {param} failed: {e}")

    print("\n" + "=" * 60)
    print(f"  Done! {success} plots saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()