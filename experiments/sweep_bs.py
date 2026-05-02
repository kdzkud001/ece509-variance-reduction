"""
sweep_bs.py — Batch Size Sweep Experiment
ECE 509 Term Project: Variance Reduction Methods Comparison

Runs SGD, SAGA, SVRG, SARAH at multiple batch sizes on a chosen dataset,
saves all results, and produces a grid comparison plot.

Usage:
    python sweep_bs.py --dataset a9a --epochs 20
    python sweep_bs.py --dataset a9a --epochs 20 --batch-sizes 32 128 512

Authors:
    Noah Jacobson (SGD, SARAH)
    Advaith Subramanian Sahasranamam (SVRG)
    Kudzaishe Kadzimu (SAGA)
"""

import argparse
import json
import os
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.makedirs("plots/sweep_bs", exist_ok=True)
os.makedirs("results/sweep_bs", exist_ok=True)

METHODS  = ["sgd", "sarah", "svrg", "saga"]
COLORS   = {"sgd": "#2196F3", "sarah": "#FF9800", "svrg": "#4CAF50", "saga": "#F44336"}

# Skip batch_size=1 for SVRG — recomputes full gradient every single step, extremely slow
SKIP = {("svrg", 1)}

DEFAULT_BATCH_SIZES = [32, 128, 512]
DEFAULT_LR = 0.05


def run_method(method, batch_size, lr, dataset, epochs):
    """
    Calls main_v4.py for a single (method, batch_size) combination.
    Returns the loss history list, or None if skipped/failed.
    """
    if (method, batch_size) in SKIP:
        print(f"  [skipped] {method} bs={batch_size} — too slow at this batch size")
        return None

    result_path = f"results/sweep_bs/{method}_bs{batch_size}.json"

    if os.path.exists(result_path):
        print(f"  [cached]  {method} bs={batch_size}")
        with open(result_path) as f:
            return json.load(f)["loss"]

    print(f"  Running {method.upper()} bs={batch_size} on {dataset} ...")

    script = None
    for candidate in ["algorithms/main_v4.py", "main_v4.py"]:
        if os.path.exists(candidate):
            script = candidate
            break

    if script is None:
        print("ERROR: Could not find main_v4.py. Run from project root.")
        sys.exit(1)

    cmd = [
        sys.executable, script,
        "--method",     method,
        "--lr",         str(lr),
        "--dataset",    dataset,
        "--epochs",     str(epochs),
        "--batch-size", str(batch_size),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  WARNING: {method} bs={batch_size} failed.")
        print(result.stderr[-500:])
        return None

    raw_path = f"{method}_results.json"
    if not os.path.exists(raw_path):
        print(f"  WARNING: {raw_path} not found after run.")
        return None

    with open(raw_path) as f:
        data = json.load(f)

    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)

    return data["loss"]


def plot_all(all_results, batch_sizes, dataset, epochs, lr):
    """Produces three plot figures for the batch size sweep."""

    n_methods    = len(METHODS)
    n_batch      = len(batch_sizes)

    # ------------------------------------------------------------------
    # FIGURE 1: Grid — rows = methods, columns = batch sizes
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(4 * n_batch, 3.5 * n_methods))
    fig.suptitle(
        f"Batch Size Sweep — {dataset.upper()} (lr={lr}, {epochs} epochs)",
        fontsize=14, fontweight="bold", y=1.01
    )

    gs = gridspec.GridSpec(n_methods, n_batch, figure=fig,
                           hspace=0.55, wspace=0.35)

    for row, method in enumerate(METHODS):
        for col, bs in enumerate(batch_sizes):
            ax = fig.add_subplot(gs[row, col])
            history = all_results.get((method, bs))

            if history is not None:
                epochs_range = range(1, len(history) + 1)
                ax.plot(epochs_range, history,
                        color=COLORS[method], linewidth=2,
                        marker='o', markersize=3)
                ax.set_title(
                    f"{method.upper()}  bs={bs}\nfinal={history[-1]:.4f}",
                    fontsize=9
                )
            elif (method, bs) in SKIP:
                ax.text(0.5, 0.5, "skipped\n(too slow)",
                        ha="center", va="center", transform=ax.transAxes,
                        color="gray", fontsize=9)
                ax.set_title(f"{method.upper()}  bs={bs}", fontsize=9)
            else:
                ax.text(0.5, 0.5, "run failed",
                        ha="center", va="center", transform=ax.transAxes,
                        color="red", fontsize=9)
                ax.set_title(f"{method.upper()}  bs={bs}", fontsize=9)

            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel("Avg Loss", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.4)

    fig.tight_layout()
    grid_path = f"plots/sweep_bs/grid_{dataset}.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved grid plot      → {grid_path}")
    plt.close(fig)

    # ------------------------------------------------------------------
    # FIGURE 2: Overlay per batch size — all methods on same axes
    # ------------------------------------------------------------------
    fig2, axes = plt.subplots(1, n_batch, figsize=(5 * n_batch, 4), sharey=False)
    if n_batch == 1:
        axes = [axes]

    fig2.suptitle(
        f"All Methods by Batch Size — {dataset.upper()} (lr={lr})",
        fontsize=13, fontweight="bold"
    )

    for col, bs in enumerate(batch_sizes):
        ax = axes[col]
        any_plotted = False
        for method in METHODS:
            history = all_results.get((method, bs))
            if history is not None:
                ax.plot(range(1, len(history) + 1), history,
                        label=method.upper(), color=COLORS[method],
                        linewidth=2, marker='o', markersize=3)
                any_plotted = True
        ax.set_title(f"batch size = {bs}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Average Loss", fontsize=9)
        ax.grid(True, alpha=0.4)
        if any_plotted:
            ax.legend(fontsize=8)

    fig2.tight_layout()
    overlay_path = f"plots/sweep_bs/overlay_{dataset}.png"
    fig2.savefig(overlay_path, dpi=150, bbox_inches="tight")
    print(f"Saved overlay plot   → {overlay_path}")
    plt.close(fig2)

    # ------------------------------------------------------------------
    # FIGURE 3: Per method — all batch sizes on same axes
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))
    fig3.suptitle(
        f"Each Method Across Batch Sizes — {dataset.upper()} (lr={lr})",
        fontsize=13, fontweight="bold"
    )

    bs_colors = plt.cm.viridis(np.linspace(0.1, 0.85, n_batch))

    for col, method in enumerate(METHODS):
        ax = axes3[col]
        for i, bs in enumerate(batch_sizes):
            history = all_results.get((method, bs))
            if history is not None:
                ax.plot(range(1, len(history) + 1), history,
                        label=f"bs={bs}", color=bs_colors[i],
                        linewidth=2, marker='o', markersize=3)
        ax.set_title(method.upper(), fontsize=11,
                     fontweight="bold", color=COLORS[method])
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Average Loss", fontsize=9)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)

    fig3.tight_layout()
    per_method_path = f"plots/sweep_bs/per_method_{dataset}.png"
    fig3.savefig(per_method_path, dpi=150, bbox_inches="tight")
    print(f"Saved per-method plot → {per_method_path}")
    plt.close(fig3)


def print_summary_table(all_results, batch_sizes):
    """Prints terminal table of final losses."""
    col_w = 12
    header = f"{'Method':<10}" + "".join(f"bs={bs}".rjust(col_w) for bs in batch_sizes)
    print("\n" + "=" * len(header))
    print("FINAL LOSS SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for method in METHODS:
        row = f"{method.upper():<10}"
        for bs in batch_sizes:
            history = all_results.get((method, bs))
            if history:
                row += f"{history[-1]:.4f}".rjust(col_w)
            elif (method, bs) in SKIP:
                row += "  skipped".rjust(col_w)
            else:
                row += "  FAILED".rjust(col_w)
        print(row)
    print("=" * len(header) + "\n")


def main(args):
    batch_sizes = args.batch_sizes
    print(f"\n{'='*55}")
    print(f"  Batch Size Sweep")
    print(f"  Dataset      : {args.dataset}")
    print(f"  Batch sizes  : {batch_sizes}")
    print(f"  LR           : {args.lr}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Methods      : {METHODS}")
    print(f"{'='*55}\n")

    all_results = {}

    for method in METHODS:
        print(f"\n--- {method.upper()} ---")
        for bs in batch_sizes:
            history = run_method(
                method, bs,
                lr=args.lr,
                dataset=args.dataset,
                epochs=args.epochs
            )
            all_results[(method, bs)] = history

    print_summary_table(all_results, batch_sizes)
    plot_all(all_results, batch_sizes, args.dataset, args.epochs, args.lr)
    print("All done. Check plots/sweep_bs/ for output figures.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch size sweep for variance reduction methods"
    )
    parser.add_argument(
        "--dataset", type=str, default="a9a",
        help="Dataset: synthetic | phishing | a9a | covtype | cod-rna"
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Epochs per run (default: 20)"
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LR,
        help=f"Fixed learning rate for all runs (default: {DEFAULT_LR})"
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to sweep (default: 32 128 512)"
    )
    args = parser.parse_args()
    main(args)