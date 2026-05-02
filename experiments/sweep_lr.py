"""
sweep_lr.py — Learning Rate Sweep Experiment
ECE 509 Term Project: Variance Reduction Methods Comparison

Runs SGD, SAGA, SVRG, SARAH at multiple learning rates on a chosen dataset,
saves all results, and produces a grid comparison plot.

Usage:
    python sweep_lr.py --dataset a9a --epochs 20
    python sweep_lr.py --dataset phishing --epochs 20 --lrs 0.001 0.01 0.05 0.1

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

os.makedirs("plots/sweep_lr", exist_ok=True)
os.makedirs("results/sweep_lr", exist_ok=True)

METHODS  = ["sgd", "sarah", "svrg", "saga"]
COLORS   = {"sgd": "#2196F3", "sarah": "#FF9800", "svrg": "#4CAF50", "saga": "#F44336"}
DEFAULT_LRS = [0.001, 0.01, 0.05, 0.1]


def run_method(method, lr, dataset, epochs, batch_size):
    """
    Calls main_v4.py for a single (method, lr) combination.
    Saves the result JSON to results/sweep_lr/<method>_lr<lr>.json.
    Returns the loss history list, or None if the run failed.
    """
    result_path = f"results/sweep_lr/{method}_lr{lr}.json"

    # Skip if already computed — avoids re-running expensive experiments
    if os.path.exists(result_path):
        print(f"  [cached] {method} lr={lr}")
        with open(result_path) as f:
            return json.load(f)["loss"]

    print(f"  Running {method.upper()} lr={lr} on {dataset} ...")

    # Locate main_v4.py — checks algorithms/ folder and current directory
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
        print(f"  WARNING: {method} lr={lr} failed.")
        print(result.stderr[-500:])   # print last 500 chars of error
        return None

    # main_v4.py writes <method>_results.json in the working directory
    raw_path = f"{method}_results.json"
    if not os.path.exists(raw_path):
        print(f"  WARNING: {raw_path} not found after run.")
        return None

    with open(raw_path) as f:
        data = json.load(f)

    # Archive to sweep results folder before next run overwrites it
    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)

    return data["loss"]


def plot_grid(all_results, lrs, dataset, epochs):
    """
    Produces a grid plot: one row per method, one column per learning rate.
    Each cell shows the loss curve for that (method, lr) pair.
    Also saves one overlay plot per learning rate (all methods on same axes).
    """

    n_methods = len(METHODS)
    n_lrs     = len(lrs)

    # ------------------------------------------------------------------
    # FIGURE 1: Grid — rows = methods, columns = learning rates
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(4 * n_lrs, 3.5 * n_methods))
    fig.suptitle(
        f"Learning Rate Sweep — {dataset.upper()} ({epochs} epochs)",
        fontsize=14, fontweight="bold", y=1.01
    )

    gs = gridspec.GridSpec(n_methods, n_lrs, figure=fig,
                           hspace=0.55, wspace=0.35)

    for row, method in enumerate(METHODS):
        for col, lr in enumerate(lrs):
            ax = fig.add_subplot(gs[row, col])
            history = all_results.get((method, lr))

            if history is not None:
                epochs_range = range(1, len(history) + 1)
                ax.plot(epochs_range, history,
                        color=COLORS[method], linewidth=2,
                        marker='o', markersize=3)
                final = history[-1]
                ax.set_title(f"{method.upper()}  lr={lr}\nfinal={final:.4f}",
                             fontsize=9)
            else:
                ax.text(0.5, 0.5, "run failed",
                        ha="center", va="center", transform=ax.transAxes,
                        color="red", fontsize=9)
                ax.set_title(f"{method.upper()}  lr={lr}", fontsize=9)

            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel("Avg Loss", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.4)

    fig.tight_layout()
    grid_path = f"plots/sweep_lr/grid_{dataset}.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved grid plot  →  {grid_path}")
    plt.close(fig)

    # ------------------------------------------------------------------
    # FIGURE 2: One overlay per learning rate (all methods on same axes)
    # ------------------------------------------------------------------
    fig2, axes = plt.subplots(1, n_lrs, figsize=(5 * n_lrs, 4), sharey=False)
    if n_lrs == 1:
        axes = [axes]

    fig2.suptitle(
        f"All Methods by Learning Rate — {dataset.upper()}",
        fontsize=13, fontweight="bold"
    )

    for col, lr in enumerate(lrs):
        ax = axes[col]
        any_plotted = False

        for method in METHODS:
            history = all_results.get((method, lr))
            if history is not None:
                epochs_range = range(1, len(history) + 1)
                ax.plot(epochs_range, history,
                        label=method.upper(), color=COLORS[method],
                        linewidth=2, marker='o', markersize=3)
                any_plotted = True

        ax.set_title(f"lr = {lr}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Average Loss", fontsize=9)
        ax.grid(True, alpha=0.4)
        if any_plotted:
            ax.legend(fontsize=8)

    fig2.tight_layout()
    overlay_path = f"plots/sweep_lr/overlay_{dataset}.png"
    fig2.savefig(overlay_path, dpi=150, bbox_inches="tight")
    print(f"Saved overlay plot →  {overlay_path}")
    plt.close(fig2)

    # ------------------------------------------------------------------
    # FIGURE 3: One overlay per method (all learning rates on same axes)
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4),
                                sharey=False)
    fig3.suptitle(
        f"Each Method Across Learning Rates — {dataset.upper()}",
        fontsize=13, fontweight="bold"
    )

    lr_colors = plt.cm.plasma(np.linspace(0.1, 0.85, n_lrs))

    for col, method in enumerate(METHODS):
        ax = axes3[col]
        for i, lr in enumerate(lrs):
            history = all_results.get((method, lr))
            if history is not None:
                epochs_range = range(1, len(history) + 1)
                ax.plot(epochs_range, history,
                        label=f"lr={lr}", color=lr_colors[i],
                        linewidth=2, marker='o', markersize=3)

        ax.set_title(method.upper(), fontsize=11,
                     fontweight="bold", color=COLORS[method])
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Average Loss", fontsize=9)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)

    fig3.tight_layout()
    per_method_path = f"plots/sweep_lr/per_method_{dataset}.png"
    fig3.savefig(per_method_path, dpi=150, bbox_inches="tight")
    print(f"Saved per-method plot → {per_method_path}")
    plt.close(fig3)


def print_summary_table(all_results, lrs):
    """Prints a terminal table of final loss values for quick comparison."""
    col_w = 12
    header = f"{'Method':<10}" + "".join(f"lr={lr}".rjust(col_w) for lr in lrs)
    print("\n" + "=" * len(header))
    print("FINAL LOSS SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for method in METHODS:
        row = f"{method.upper():<10}"
        for lr in lrs:
            history = all_results.get((method, lr))
            if history:
                row += f"{history[-1]:.4f}".rjust(col_w)
            else:
                row += "  FAILED".rjust(col_w)
        print(row)
    print("=" * len(header) + "\n")


def main(args):
    lrs = args.lrs
    print(f"\n{'='*55}")
    print(f"  Learning Rate Sweep")
    print(f"  Dataset : {args.dataset}")
    print(f"  LRs     : {lrs}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Methods : {METHODS}")
    print(f"{'='*55}\n")

    all_results = {}

    for method in METHODS:
        print(f"\n--- {method.upper()} ---")
        for lr in lrs:
            history = run_method(
                method, lr,
                dataset=args.dataset,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            all_results[(method, lr)] = history

    print_summary_table(all_results, lrs)
    plot_grid(all_results, lrs, args.dataset, args.epochs)
    print("\nAll done. Check plots/sweep_lr/ for output figures.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learning rate sweep for variance reduction methods"
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
        "--batch-size", type=int, default=128,
        help="Mini-batch size (default: 128)"
    )
    parser.add_argument(
        "--lrs", type=float, nargs="+",
        default=DEFAULT_LRS,
        help="Learning rates to sweep (default: 0.001 0.01 0.05 0.1)"
    )
    args = parser.parse_args()
    main(args)