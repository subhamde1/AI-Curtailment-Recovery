"""Regenerate the published Figure 17 from its immutable timing snapshot."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination PNG; the frozen record is never overwritten.")
    args = parser.parse_args()
    data = json.loads((ROOT / "publication_snapshot/scaling.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3))
    for seed, color in zip((20261011, 20261012, 20261013), ("#173F5F", "#A34A28", "#367C68")):
        records = [q for q in data if q["seed"] == seed]
        axes[0].plot([q["jobs"] for q in records], [q["median_solve_ms"] for q in records],
                     "o-", color=color, label=str(seed))
        axes[1].plot([q["jobs"] for q in records], [q["p95_solve_ms"] for q in records],
                     "s-", color=color)
    axes[0].set(xlabel="Base jobs", ylabel="Median LP solve time (ms)", xticks=(50, 200, 500))
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Base jobs", ylabel="95th percentile LP time (ms)", xticks=(50, 200, 500))
    for axis in axes:
        axis.grid(alpha=.15)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
