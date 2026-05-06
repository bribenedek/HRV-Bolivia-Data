"""
Box and Whisker Plots — HRV Parameters by Group
-------------------------------------------------
Generates one box plot per HRV parameter, with three boxes:
Calgary Baseline, La Paz Testing, Calgary Return.

Stats are computed at the PARTICIPANT level (one mean per participant
per group) which is the standard approach for HRV papers.

Features:
- Participant-level averaging before stats
- Kruskal-Wallis overall test
- Pairwise Mann-Whitney U with significance bars (* p<0.05, ** p<0.01, *** p<0.001)
- Clean publication-style plots, no individual dots

Usage:
    python box_whisker_plots.py
"""

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats
from itertools import combinations

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
    {"keyword": "baseline", "label": "Calgary\nBaseline",  "color": "#2196F3"},
    {"keyword": "la paz",   "label": "La Paz\nTesting",    "color": "#FF5722"},
    {"keyword": "return",   "label": "Calgary\nReturn",    "color": "#4CAF50"},
]

# ── Plot config ───────────────────────────────────────────────────────────────
DPI = 300
FIGSIZE = (7, 6)
SIG_LEVEL = 0.05
# ─────────────────────────────────────────────────────────────────────────────


def find_group_folder(window_dir: Path, keyword: str):
    for folder in window_dir.iterdir():
        if folder.is_dir() and keyword.lower() in folder.name.lower():
            return folder
    return None


def extract_participant_id(filename: str) -> str:
    """Extract participant ID from filename (first two underscore-separated parts)."""
    parts = Path(filename).stem.split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else parts[0]


def load_group_data(folder: Path) -> pd.DataFrame:
    """Load all CSVs and tag with participant ID."""
    dfs = []
    for csv_path in sorted(folder.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
            df["participant_id"] = extract_participant_id(csv_path.name)
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠ Could not load {csv_path.name}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def get_participant_means(df: pd.DataFrame, param: str) -> pd.Series:
    """Average each participant down to one value — standard for HRV stats."""
    if param not in df.columns or df.empty:
        return pd.Series(dtype=float)
    df[param] = pd.to_numeric(df[param], errors='coerce')
    return df.groupby("participant_id")[param].mean().dropna()


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


def format_p(p: float) -> str:
    if p < 0.001: return "p < 0.001"
    if p < 0.01:  return "p < 0.01"
    if p < 0.05:  return "p < 0.05"
    return f"p = {p:.3f}"


def star_label(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def add_significance_bars(ax, data_to_plot):
    """Add pairwise Mann-Whitney significance bars above the boxes."""
    all_vals = [v for v in data_to_plot if len(v) > 0]
    if not all_vals:
        return

    y_max = max(np.nanmax(v) for v in all_vals)
    y_min = min(np.nanmin(v) for v in all_vals)
    y_range = y_max - y_min if y_max != y_min else abs(y_max) * 0.1 or 1
    bar_height = y_range * 0.05
    bar_gap = y_range * 0.10
    current_y = y_max + bar_height

    pairs = list(combinations(range(3), 2))
    sig_pairs = []
    for i, j in pairs:
        d1, d2 = data_to_plot[i], data_to_plot[j]
        if len(d1) < 2 or len(d2) < 2:
            continue
        _, p = stats.mannwhitneyu(d1, d2, alternative='two-sided')
        star = star_label(p)
        if star:
            sig_pairs.append((i, j, star))

    for idx, (i, j, star) in enumerate(sig_pairs):
        x1, x2 = i + 1, j + 1
        y = current_y + idx * bar_gap
        ax.plot([x1, x1, x2, x2],
                [y, y + bar_height * 0.5, y + bar_height * 0.5, y],
                color='black', linewidth=1.2)
        ax.text((x1 + x2) / 2, y + bar_height * 0.6, star,
                ha='center', va='bottom', fontsize=14, color='black')

    if sig_pairs:
        n = len(sig_pairs)
        ax.set_ylim(top=current_y + n * bar_gap + bar_height * 3)


def make_box_plot(param: str, group_data: dict, output_dir: Path):
    """Generate and save one box plot for a single HRV parameter."""
    colors = [g["color"] for g in GROUPS]
    labels = [g["label"] for g in GROUPS]

    # Get participant-level means per group
    data_to_plot = []
    for group in GROUPS:
        df = group_data[group["label"]]
        means = get_participant_means(df, param)
        data_to_plot.append(means.values)

    # Skip if all groups empty
    if all(len(v) == 0 for v in data_to_plot):
        print(f"  ⚠ Skipping {param} — no valid data in any group")
        return None

    # Kruskal-Wallis
    valid = [v for v in data_to_plot if len(v) >= 2]
    kw_p = None
    kw_sig = False
    if len(valid) >= 2:
        _, kw_p = stats.kruskal(*valid)
        kw_sig = kw_p < SIG_LEVEL

    fig, ax = plt.subplots(figsize=FIGSIZE)

    bp = ax.boxplot(
        data_to_plot,
        patch_artist=True,
        notch=False,
        widths=0.5,
        medianprops=dict(color='black', linewidth=2.5),
        whiskerprops=dict(linewidth=1.5, linestyle='--'),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='', markersize=0),  # no dots
    )

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Pairwise significance bars (only if KW significant)
    if kw_sig:
        add_significance_bars(ax, data_to_plot)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(nice_label(param), fontsize=12)
    ax.set_title(f"{nice_label(param)}\nby Study Phase", fontsize=13, fontweight='bold')

    # Category label top right
    cat = param_group_label(param)
    ax.text(0.98, 0.98, cat, transform=ax.transAxes,
            fontsize=9, color='gray', ha='right', va='top', style='italic')

    # KW p-value bottom left
    if kw_p is not None:
        ax.text(0.02, 0.01, f"Kruskal-Wallis {format_p(kw_p)}",
                transform=ax.transAxes, fontsize=8, color='gray', va='bottom')

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Significance legend at the bottom
    ax.text(0.5, -0.12, "* p < 0.05     ** p < 0.01     *** p < 0.001",
            transform=ax.transAxes, fontsize=9, color='gray',
            ha='center', va='top')

    plt.tight_layout()
    out_path = output_dir / f"{param}_boxplot.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    return out_path


def main():
    print("=" * 60)
    print("     HRV Box and Whisker Plots — By Study Phase")
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
            print(f"  ✗ Could not find subfolder for '{group['label'].replace(chr(10), ' ')}' "
                  f"(looking for '{group['keyword']}' in folder names)")
            return
        print(f"  ✓ {group['label'].replace(chr(10), ' ')} → {folder.name}")
        df = load_group_data(folder)
        n_participants = df["participant_id"].nunique() if not df.empty else 0
        print(f"    Loaded {n_participants} participants, {len(df):,} total windows")
        group_data[group["label"]] = df

    print(f"\nGenerating {len(ALL_PARAMS)} box plots...")
    print(f"Output → {output_dir}\n")

    success = 0
    for i, param in enumerate(ALL_PARAMS):
        try:
            out_path = make_box_plot(param, group_data, output_dir)
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