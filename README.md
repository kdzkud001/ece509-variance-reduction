# ECE 509 Term Project — Variance Reduction Methods for SGD

**Course**: ECE 509 — Convex Optimization (Spring 2026)  
**Instructor**: Prof. Waheed U. Bajwa, Rutgers University

## Team Members

| Name | Role |
|---|---|
| Noah Jacobson (Point of Contact) | SGD, SARAH implementation |
| Kudzaishe Kadzimu | SVRG implementation |
| Advaith Subramanian Sahasranamam | SAGA implementation |

---

## Project Overview

This project implements and compares four optimization methods for
finite-sum convex objectives:

- **SGD** — Stochastic Gradient Descent (baseline)
- **SVRG** — Stochastic Variance Reduced Gradient (Johnson & Zhang, 2013)
- **SAGA** — Stochastic Average Gradient Accelerated (Defazio et al., 2014)
- **SARAH** — Stochastic Recursive Gradient Algorithm (Nguyen et al., 2017)

All methods are applied to L2-regularized binary logistic regression on
the `a9a` benchmark dataset from LIBSVM.

---

## Repository Structure

```
ece509-variance-reduction/
├── algorithms/
│   └── main_v4.py              # Main training script (all 4 methods)
├── experiments/
│   ├── sweep_lr.py             # Learning rate sweep experiment
│   ├── sweep_bs.py             # Batch size sweep experiment
│   └── plot_grad_norms.py      # Gradient norm visualization script
├── plots/
│   ├── plot_comparison.png     # Main convergence comparison
│   ├── grad_norms/             # Gradient norm and variance plots
│   ├── sweep_lr/               # Learning rate sweep plots
│   └── sweep_bs/               # Batch size sweep plots
├── results/
│   ├── sweep_lr/               # Cached JSON results for lr sweep
│   └── sweep_bs/               # Cached JSON results for bs sweep
├── requirements.txt            # Python dependencies
├── llm-usage.md                # LLM disclosure (required by course policy)
└── README.md                   # This file
```

---

## Setup

### Requirements

- Python 3.9+
- Linux / WSL / macOS

### Installation

```bash
# Clone the repository
git clone https://github.com/kdzkud001/ece509-variance-reduction.git
cd ece509-variance-reduction

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn matplotlib numpy scipy libsvmdata
```

---

## Reproducing the Results

All experiments are run from the **project root directory**.
Results are saved as `<method>_results.json` in the working directory.
Plots are saved to `plots/`.

### Step 1 — Main Convergence Comparison (Figure 1 in report)

```bash
python algorithms/main_v4.py --method sgd   --lr 0.05 --epochs 20 --dataset a9a --weight-decay 0.01
python algorithms/main_v4.py --method sarah --lr 0.05 --epochs 20 --dataset a9a --weight-decay 0.01
python algorithms/main_v4.py --method svrg  --lr 0.05 --epochs 20 --dataset a9a --weight-decay 0.01
python algorithms/main_v4.py --method saga  --lr 0.05 --epochs 20 --dataset a9a --weight-decay 0.01
```

Output: `plots/plot_comparison.png`

---

### Step 2 — Gradient Norm and Variance Plots (Figures 2 and 3 in report)

Run Step 1 first, then:

```bash
python experiments/plot_grad_norms.py --dataset a9a --smooth 50
```

Output: `plots/grad_norms/per_method_a9a.png`, `plots/grad_norms/variance_a9a.png`

---

### Step 3 — Learning Rate Sweep (Figure 4 in report)

```bash
python experiments/sweep_lr.py --dataset a9a --epochs 20 --lrs 0.001 0.01 0.05 0.1 0.5
```

Output: `plots/sweep_lr/overlay_a9a.png`

---

### Step 4 — Batch Size Sweep (Figure 5 in report)

```bash
python experiments/sweep_bs.py --dataset a9a --epochs 20 --lr 0.05
```

Output: `plots/sweep_bs/overlay_a9a.png`

---

### Step 5 — Aggressive Learning Rate Experiment (Figure 6 in report)

```bash
rm -f sgd_results.json sarah_results.json svrg_results.json saga_results.json

python algorithms/main_v4.py --method sgd   --lr 0.5 --epochs 50 --dataset a9a --weight-decay 0.01
python algorithms/main_v4.py --method sarah --lr 0.5 --epochs 50 --dataset a9a --weight-decay 0.01
python algorithms/main_v4.py --method svrg  --lr 0.5 --epochs 50 --dataset a9a --weight-decay 0.01
python algorithms/main_v4.py --method saga  --lr 0.5 --epochs 50 --dataset a9a --weight-decay 0.01

cp plots/plot_comparison.png plots/plot_comparison_aggressive.png
```

Output: `plots/plot_comparison_aggressive.png`

---

## Command Line Arguments

`main_v4.py` accepts the following arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--method` | str | required | `sgd`, `sarah`, `svrg`, or `saga` |
| `--dataset` | str | `synthetic` | `synthetic`, `a9a`, `phishing`, `covtype`, `cod-rna` |
| `--lr` | float | `0.1` | Learning rate $\eta$ |
| `--batch-size` | int | `128` | Mini-batch size |
| `--epochs` | int | `10` | Number of training epochs |
| `--weight-decay` | float | `0.01` | L2 regularization strength $\lambda$ |
| `--inner-loop-size` | int | `None` | SVRG inner loop size $m$ (default: $\lfloor n/b \rfloor$) |

---

## Dataset

The `a9a` dataset is downloaded automatically from LIBSVM on first run
via the `libsvmdata` package. It requires an internet connection for
the first run only; subsequent runs use the cached version.

- **Samples**: 32,561
- **Features**: 123
- **Task**: Binary classification
- **Source**: UCI Adult census dataset (Chang & Lin, 2011)

---

## Key Results

| Method | Mean Gradient Norm | Variance Reduction | Final Loss (η=0.05) |
|---|---|---|---|
| SGD | 0.303 | None (noise floor) | ~0.32 |
| SARAH | 0.303 | Minimal | ~0.32 |
| SVRG | 0.026 | ✅ Collapses after ~300 iters | ~0.32 |
| SAGA | 0.035 | ✅ Collapses after ~500 iters | ~0.32 |

At aggressive learning rate η=0.5: SGD and SARAH diverge; SVRG and SAGA converge.

---

## References

1. R. Johnson and T. Zhang, "Accelerating SGD using predictive variance reduction," NeurIPS 2013.
2. A. Defazio, F. Bach, S. Lacoste-Julien, "SAGA: A fast incremental gradient method," NeurIPS 2014.
3. L. M. Nguyen et al., "SARAH: A novel method for machine learning problems," ICML 2017.
4. C.-C. Chang and C.-J. Lin, "LIBSVM: A library for support vector machines," TIST 2011.

---

## LLM Disclosure

This project used Claude (Anthropic) for implementation assistance and
experiment design. Full disclosure including prompts, outputs, and
verification steps is documented in [`llm-usage.md`](llm-usage.md),
as required by course policy (Section 6.2 of the project specification).