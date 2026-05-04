"""
plot_grad_norms.py — Gradient Norm Comparison Plot
ECE 509 Term Project: Variance Reduction Methods Comparison

Reads *_results.json files produced by main_v4.py and plots the
gradient norm per iteration for all methods on the same axes.

This directly visualises the variance reduction mechanism:
  - SGD:               gradient norm stays noisy (bounces around) — the noise floor
  - SARAH/SVRG/SAGA:   norm visibly decreases and stabilises over time

Usage:
    # First run all four methods to generate result JSONs:
    python algorithms/main_v4.py --method sgd   --lr 0.05 --epochs 20 --dataset a9a
    python algorithms/main_v4.py --method sarah --lr 0.05 --epochs 20 --dataset a9a
    python algorithms/main_v4.py --method svrg  --lr 0.05 --epochs 20 --dataset a9a
    python algorithms/main_v4.py --method saga  --lr 0.05 --epochs 20 --dataset a9a

    # Then plot:
    python experiments/plot_grad_norms.py --dataset a9a

Authors:
    Noah Jacobson, Advaith Subramanian Sahasranamam, Kudzaishe Kadzimu
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.makedirs("plots/grad_norms", exist_ok=True)

METHODS = ["sgd", "sarah", "svrg", "saga"]
COLORS = {
    "sgd":   "#2196F3",
    "sarah": "#FF9800",
    "svrg":  "#4CAF50",
    "saga":  "#F44336",
}


def load_norms(method):
    """
    Loads gradient norms from <method>_results.json.
    Returns the list of per-iteration norms, or None if file not found
    or if the file predates grad_norm logging.
    """
    path = f"{method}_results.json"
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping {method.upper()}")
        return None
    with open(path) as f:
        data = json.load(f)
    if "grad_norms" not in data:
        print(
            f"  WARNING: {path} has no 'grad_norms' key.\n"
            f"           Re-run main_v4.py to regenerate results with norm logging."
        )
        return None
    norms = data["grad_norms"]
    if len(norms) == 0:
        print(
            f"  WARNING: {path} has an empty 'grad_norms' list.\n"
            f"           Re-run main_v4.py to regenerate results with norm logging."
        )
        return None
    return norms


def smooth(values, window):
    """
    Rolling average smoothing. Used to show trend clearly without
    hiding the underlying noise in the raw norm signal.

    Args:
        values: list or array of floats
        window: int, rolling window size

    Returns:
        numpy array of same length as values
    """
    values = np.array(values, dtype=float)
    smoothed = np.convolve(values, np.ones(window) / window, mode="same")

    # Fix edge artifacts at boundaries with shrinking windows
    half = window // 2
    for i in range(half):
        smoothed[i] = np.mean(values[:2 * i + 1]) if i > 0 else values[0]
        smoothed[-(i + 1)] = np.mean(values[-(2 * i + 1):]) if i > 0 else values[-1]

    return smoothed


def plot_overlay(all_norms, dataset, smooth_window, max_iters):
    """
    Single plot with all four methods overlaid.
    Raw norms shown faint, smoothed norms shown bold.
    This is the main report figure — directly shows variance reduction.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for method in METHODS:
        norms = all_norms.get(method)
        if norms is None:
            continue

        norms = np.array(norms[:max_iters], dtype=float)
        iters = np.arange(1, len(norms) + 1)
        color = COLORS[method]

        # Raw norm — faint to show true noise level
        ax.plot(iters, norms, color=color, alpha=0.15, linewidth=0.8)

        # Smoothed norm — bold to show trend
        smoothed = smooth(norms, smooth_window)
        ax.plot(iters, smoothed, color=color, linewidth=2.2,
                label=f"{method.upper()} (smoothed, w={smooth_window})")

    ax.set_xlabel("Iteration (mini-batch step)", fontsize=11)
    ax.set_ylabel("Gradient Norm  ||g||2", fontsize=11)
    ax.set_title(
        f"Gradient Norm per Iteration — {dataset.upper()}\n"
        f"Faint = raw norm, Bold = rolling average (window={smooth_window})",
        fontsize=11
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.35)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    fig.tight_layout()
    out = f"plots/grad_norms/overlay_{dataset}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved overlay plot         -> {out}")
    plt.close(fig)


def plot_per_method(all_norms, dataset, smooth_window, max_iters):
    """
    2x2 grid — one panel per method.
    Each panel shows raw norm, smoothed norm, and mean norm as dashed line.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"Gradient Norm per Iteration — {dataset.upper()}  "
        f"(smoothing window = {smooth_window})",
        fontsize=13, fontweight="bold"
    )
    axes = axes.flatten()

    for i, method in enumerate(METHODS):
        ax = axes[i]
        norms = all_norms.get(method)

        if norms is None:
            ax.text(0.5, 0.5, f"{method.upper()}\nno data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color="gray")
            ax.set_title(method.upper(), fontsize=11)
            continue

        norms = np.array(norms[:max_iters], dtype=float)
        iters = np.arange(1, len(norms) + 1)
        color = COLORS[method]

        # Raw norm (faint)
        ax.plot(iters, norms, color=color, alpha=0.2, linewidth=0.8)
        # Smoothed norm (bold)
        smoothed = smooth(norms, smooth_window)
        ax.plot(iters, smoothed, color=color, linewidth=2.2, label="smoothed")
        # Mean norm (dashed)
        mean_norm = np.mean(norms)
        ax.axhline(mean_norm, color=color, linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"mean = {mean_norm:.4f}")

        ax.set_title(method.upper(), fontsize=11,
                     fontweight="bold", color=color)
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel("||g||2", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.35)

    fig.tight_layout()
    out = f"plots/grad_norms/per_method_{dataset}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved per-method plot      -> {out}")
    plt.close(fig)


def plot_variance_comparison(all_norms, dataset, window, max_iters):
    """
    Plots rolling variance and rolling std of gradient norm over iterations.

    This is the most theoretically significant plot:
      - SGD variance stays flat — the noise floor never goes away
      - VR methods: variance trends downward — gradient estimates get cleaner
    This matches exactly what SVRG, SAGA, SARAH papers prove theoretically.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Gradient Norm Variance Over Iterations — {dataset.upper()}\n"
        f"Rolling variance (window={window}) — lower = less noise in gradient estimates",
        fontsize=12, fontweight="bold"
    )

    ax_var = axes[0]
    ax_std = axes[1]

    for method in METHODS:
        norms = all_norms.get(method)
        if norms is None:
            continue

        norms = np.array(norms[:max_iters], dtype=float)
        iters = np.arange(1, len(norms) + 1)
        color = COLORS[method]

        rolling_var = np.array([
            np.var(norms[max(0, i - window):i + 1])
            for i in range(len(norms))
        ])
        rolling_std = np.sqrt(rolling_var)

        ax_var.plot(iters, rolling_var, color=color, linewidth=1.8,
                    label=method.upper(), alpha=0.85)
        ax_std.plot(iters, rolling_std, color=color, linewidth=1.8,
                    label=method.upper(), alpha=0.85)

    for ax, ylabel, title in [
        (ax_var, "Rolling Variance of ||g||2", "Gradient Norm Variance"),
        (ax_std, "Rolling Std of ||g||2",      "Gradient Norm Std Dev"),
    ]:
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.35)

    fig.tight_layout()
    out = f"plots/grad_norms/variance_{dataset}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved variance plot        -> {out}")
    plt.close(fig)


def print_summary(all_norms):
    """Prints terminal table: mean, std, max norm for each method."""
    print("\n" + "=" * 54)
    print("GRADIENT NORM SUMMARY")
    print(f"{'Method':<10} {'Mean ||g||':>12} {'Std ||g||':>12} {'Max ||g||':>12}")
    print("-" * 54)
    for method in METHODS:
        norms = all_norms.get(method)
        if norms is None:
            print(f"{method.upper():<10} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
        else:
            norms = np.array(norms, dtype=float)
            print(
                f"{method.upper():<10} "
                f"{np.mean(norms):>12.4f} "
                f"{np.std(norms):>12.4f} "
                f"{np.max(norms):>12.4f}"
            )
    print("=" * 54 + "\n")


def main(args):
    print(f"\nLoading gradient norm data  (dataset: {args.dataset})")

    all_norms = {method: load_norms(method) for method in METHODS}

    if all(v is None for v in all_norms.values()):
        print(
            "\nERROR: No result files found.\n"
            "Run main_v4.py for all methods first, then retry.\n"
        )
        return

    print_summary(all_norms)

    print("Generating plots ...")
    plot_overlay(all_norms, args.dataset, args.smooth, args.max_iters)
    plot_per_method(all_norms, args.dataset, args.smooth, args.max_iters)
    plot_variance_comparison(all_norms, args.dataset, args.smooth, args.max_iters)

    print(f"\nDone. All plots saved to plots/grad_norms/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot gradient norms from main_v4.py result JSON files"
    )
    parser.add_argument(
        "--dataset", type=str, default="a9a",
        help="Dataset name — used for plot titles only (default: a9a)"
    )
    parser.add_argument(
        "--smooth", type=int, default=50,
        help="Rolling average window size for smoothing (default: 50)"
    )
    parser.add_argument(
        "--max-iters", type=int, default=5000,
        help="Max iterations to plot — caps x-axis for readability (default: 5000)"
    )
    args = parser.parse_args()
    main(args)