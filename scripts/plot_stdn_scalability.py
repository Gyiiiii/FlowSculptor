import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCALES = [25, 50, 75, 100]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot HF-STFormer vs STDN scalability curves.")
    parser.add_argument("--dataset", default="PEMS04", choices=["PEMS04"])
    parser.add_argument("--input_csv", default=None, help="Default: scalability_stdn_plot_data.csv if it exists, else scalability_summary.csv.")
    return parser.parse_args()


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_curve(rows, model, scale_type):
    values = []
    for scale in SCALES:
        candidates = [
            row
            for row in rows
            if row["Model"] == model
            and row["Scale Type"] == scale_type
            and int(row["Scale Percent"]) == scale
            and row.get("Inference Time / Test Set (s)") not in (None, "")
        ]
        if not candidates:
            raise ValueError(f"Missing {model} {scale_type} {scale}% in plot data.")
        values.append(float(candidates[0]["Inference Time / Test Set (s)"]))
    return values


def main():
    args = parse_args()
    out_root = PROJECT_ROOT / "experiments" / "scalability"
    preferred = out_root / "scalability_stdn_plot_data.csv"
    summary = out_root / "scalability_summary.csv"
    input_path = Path(args.input_csv) if args.input_csv else (preferred if preferred.exists() else summary)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_path}. Run summarize_scalability.py first.")
    rows = read_rows(input_path)

    x_labels = ["25%", "50%", "75%", "100%"]
    x_indices = np.arange(len(x_labels))

    stdn_dataset = get_curve(rows, "STDN", "data")
    hfstformer_dataset = get_curve(rows, "HF-STFormer", "data")
    stdn_graph = get_curve(rows, "STDN", "graph")
    hfstformer_graph = get_curve(rows, "HF-STFormer", "graph")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.2), dpi=300)

    lw = 1.8
    ms = 9.0
    mew = 1.6
    color_red = "#FF1A1A"
    color_blue = "#2980B9"

    ax1.plot(
        x_indices,
        stdn_dataset,
        color=color_red,
        marker="s",
        markersize=ms,
        markerfacecolor="white",
        markeredgewidth=mew,
        linewidth=lw,
        label="STDN",
    )
    ax1.plot(
        x_indices,
        hfstformer_dataset,
        color=color_blue,
        marker="^",
        markersize=ms,
        markerfacecolor="white",
        markeredgewidth=mew,
        linewidth=lw,
        label="HF-STFormer",
    )
    ax1.set_xlabel("Cardinality of dataset", fontsize=12, labelpad=4)
    ax1.set_ylabel("Inference time (s)", fontsize=12, labelpad=2)

    ax2.plot(
        x_indices,
        stdn_graph,
        color=color_red,
        marker="s",
        markersize=ms,
        markerfacecolor="white",
        markeredgewidth=mew,
        linewidth=lw,
        label="STDN",
    )
    ax2.plot(
        x_indices,
        hfstformer_graph,
        color=color_blue,
        marker="^",
        markersize=ms,
        markerfacecolor="white",
        markeredgewidth=mew,
        linewidth=lw,
        label="HF-STFormer",
    )
    ax2.set_xlabel("Cardinality of graph", fontsize=12, labelpad=4)
    ax2.set_ylabel("Inference time (s)", fontsize=12, labelpad=2)

    ymax = max(stdn_dataset + hfstformer_dataset + stdn_graph + hfstformer_graph)
    y_top = max(1.0, np.ceil(ymax / 5.0) * 5.0)
    yticks = np.arange(0, y_top + 0.1, 5.0)

    for ax in [ax1, ax2]:
        ax.set_xlim(-0.18, 3.18)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylim(0, y_top)
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(int(item)) for item in yticks], fontsize=10)
        ax.tick_params(axis="y", pad=3)
        ax.tick_params(axis="x", pad=5)
        ax.grid(True, linestyle="-", color="#DCDCDC", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(direction="in", top=False, right=False, bottom=True, left=True, length=3.5, width=1.0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
            spine.set_color("black")
        leg = ax.legend(
            loc="upper left",
            frameon=True,
            fontsize=9.5,
            facecolor="white",
            edgecolor="#B0B0B0",
            framealpha=1,
            handlelength=1.5,
            handletextpad=0.4,
            borderpad=0.3,
        )
        leg.get_frame().set_linewidth(0.7)

    plt.subplots_adjust(wspace=0.38, left=0.11, right=0.985, top=0.95, bottom=0.22)
    for suffix in ("png", "pdf"):
        output_path = out_root / f"stdn_scalability_time.{suffix}"
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.10, dpi=300)
        print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
