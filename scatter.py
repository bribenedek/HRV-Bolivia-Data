"""
Ratio Metric Dot Plots — One dot per participant by Study Phase
---------------------------------------------------------------
Generates one dot plot per ratio HRV parameter with three groups:
Calgary Baseline | La Paz Testing | Calgary Return

Each dot = one participant (averaged across their windows).
A horizontal line shows the group mean.

Usage:
    python ratio_dotplots.py
"""

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Ratio parameters ──────────────────────────────────────────────────────────
RATIO_PARAMS = ['lf_hf_ratio', 'ratio_sd2_sd1', 'lfnu', 'hfnu', 'cvsd', 'cvnni']

# ── Group config ──────────────────────────────────────────────────────────────
GROUPS = [
    {"keyword": "baseline", "label": "Calgary\nBaseline",  "color": "#2196F3"},
    {"keyword": "la paz",   "label": "La Paz\nTesting",    "color": "#FF5722"},
    {"keyword": "return",   "label": "Calgary\nReturn",    "color": "#4CAF50"},
]

# ── Plot config ───────────────────────────────────────────────────────────────
DPI = 300
FIGSIZE = (7, 5)
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


def get_participant_means(df: pd.DataFrame, param: str) -> pd.Series:
    if param not in df.columns or df.empty:
        return pd.Series(dtype=float)
    df[param] = pd.to_numeric(df[param], errors='coerce')
    return df.groupby("participant_id")[param].mean().dropna()


def nice_label(param: str) -> str:
    labels = {
        'lf_hf_ratio': 'LF/HF Ratio',
        'ratio_sd2_sd1': 'SD2/SD1 Ratio',
        'lfnu': 'LF (n.u.)',
        'hfnu': 'HF (n.u.)',
        'cvsd': 'CVSD',
        'cvnni': 'CVNNI',
    }
    return labels.get(param, param.replace('_', ' ').upper())


def make_dot_plot(param: str, group_data: dict, output_dir: Path):
    """Generate and save one dot plot for a single ratio parameter."""

    data_to_plot = []
    for group in GROUPS:
        df = group_data[group["label"]]
        means = get_participant_means(df, param)
        data_to_plot.append(means.values)

    if all(len(v) == 0 for v in data_to_plot):
        print(f"  ⚠ Skipping {param} — no valid data in any group")
        return None

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for i, (group, vals) in enumerate(zip(GROUPS, data_to_plot)):
        x_pos = i + 1
        color = group["color"]

        if len(vals) == 0:
            continue

        # Jittered dots
        jitter = np.random.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(x_pos + jitter, vals,
                   color=color, alpha=0.8, s=40, zorder=3,
                   edgecolors='black', linewidths=0.4)

        # Mean line
        mean_val = np.mean(vals)
        ax.plot([x_pos - 0.3, x_pos + 0.3], [mean_val, mean_val],
                color='black', linewidth=2.5, zorder=4)

        # n label
        ax.text(x_pos, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else min(vals) * 0.95,
                f"n={len(vals)}", ha='center', va='top', fontsize=9, color='gray')

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([g["label"] for g in GROUPS], fontsize=11)
    ax.set_ylabel(nice_label(param), fontsize=12)
    ax.set_title(f"{nice_label(param)}\nby Study Phase", fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend for mean line
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=2.5, label='Group mean'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, frameon=False, loc='upper right')

    plt.tight_layout()
    out_path = output_dir / f"{param}_dotplot.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    return out_path


def main():
    print("=" * 60)
    print("     HRV Ratio Metric Dot Plots — By Study Phase")
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

    print(f"\nGenerating {len(RATIO_PARAMS)} dot plots...")
    print(f"Output → {output_dir}\n")

    success = 0
    for i, param in enumerate(RATIO_PARAMS):
        try:
            out_path = make_dot_plot(param, group_data, output_dir)
            if out_path:
                print(f"  ✓ [{i+1}/{len(RATIO_PARAMS)}] {param}")
                success += 1
        except Exception as e:
            print(f"  ✗ [{i+1}/{len(RATIO_PARAMS)}] {param} failed: {e}")

    print("\n" + "=" * 60)
    print(f"  Done! {success} plots saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()