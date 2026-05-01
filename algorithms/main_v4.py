import argparse
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import os
os.makedirs("plots", exist_ok=True)

# Noah Jacobson: SGD, SARAH
# Advaith Subramanian Sahasranamam: SVRG
# Kudzaishe Kadzimu: SAGA (to be added)

# run first:
'''
conda create -n vr_optims python=3.9 pytorch torchvision cpuonly -c pytorch -c conda-forge
conda activate vr_optims
pip install matplotlib tqdm coloredlogs scikit-learn
'''

# example: python main_v4.py --method svrg --lr 0.05 --batch-size 128 --epochs 10 --inner-loop-size 100


# =============================================================================
# BASE CLASS
# =============================================================================

class OptimizerBase:
    """
    Parent class shared by all optimizers.
    Stores the model parameters and learning rate.
    Provides helper methods to clear and save gradients.
    """
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr
        self.grads = None

    def zero_grad(self):
        """Clears stored gradients and PyTorch's internal gradient buffers."""
        self.grads = None
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def store_grads(self):
        """
        Copies gradients out of PyTorch's internal structure and saves them.
        .clone() is critical — without it we'd just hold a reference to memory
        that gets zeroed next iteration.
        """
        self.grads = [p.grad.clone() for p in self.params]


# =============================================================================
# SGD
# =============================================================================

class SGD(OptimizerBase):
    """
    Vanilla Stochastic Gradient Descent.
    Update rule: w = w - lr * g
    where g is the mini-batch gradient.

    Problem: g is noisy — computed on a random subset of data.
    That noise never goes away, creating a 'noise floor' that
    prevents exact convergence regardless of step size.
    """
    def step(self):
        with torch.no_grad():
            for parameter, gradient in zip(self.params, self.grads):
                parameter -= self.lr * gradient


# =============================================================================
# SARAH
# =============================================================================

class SARAH(OptimizerBase):
    """
    Stochastic Recursive Gradient Algorithm (SARAH).
    Nguyen et al., 2017. https://arxiv.org/pdf/1703.00102

    Key idea: Instead of using the raw noisy mini-batch gradient,
    build a RECURSIVE correction:

        v_t = g_t - g_{t-1} + v_{t-1}

    where:
        g_t  = mini-batch gradient at current parameters
        g_{t-1} = mini-batch gradient at PREVIOUS parameters (same batch)
        v_{t-1} = previous corrected gradient

    The noise in g_t and g_{t-1} partially cancels, reducing variance.
    An outer loop (full gradient) resets the estimate each epoch.

    Memory cost: O(d) — just stores previous gradient vector.
    No gradient table needed (unlike SAGA).
    """
    def __init__(self, params, lr):
        super().__init__(params, lr)
        self.v_prev = None       # previous corrected gradient
        self.grads_prev = None   # previous raw mini-batch gradient

    def get_outer_loop(self):
        """
        Called once per epoch BEFORE the inner mini-batch loop.
        Computes the full gradient over all data and takes one step with it.
        This 'resets' the recursive estimate to a low-noise starting point.
        """
        # Save full gradient as starting point for recursive correction
        self.v_prev = [p.grad.clone() for p in self.params]
        self.grads_prev = [p.grad.clone() for p in self.params]

        # Take one gradient step using the full gradient
        with torch.no_grad():
            for p, v0 in zip(self.params, self.v_prev):
                p -= self.lr * v0

    def step(self):
        with torch.no_grad():
            if self.v_prev is None:
                # First iteration: no previous values, use raw gradient
                v = self.grads
            else:
                # Recursive correction: v_t = g_t - g_{t-1} + v_{t-1}
                v = [g - gp + vp for g, gp, vp in
                     zip(self.grads, self.grads_prev, self.v_prev)]

            for parameter, v_i in zip(self.params, v):
                parameter -= self.lr * v_i

            # Save for next iteration
            self.v_prev = v
            self.grads_prev = self.grads


# =============================================================================
# SVRG
# =============================================================================

class SVRG(OptimizerBase):
    """
    Stochastic Variance Reduced Gradient (SVRG).
    Johnson & Zhang, NeurIPS 2013. https://proceedings.neurips.cc/paper/2013/file/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf

    Key idea: Periodically take a SNAPSHOT of the model parameters (w_tilde)
    and compute the full gradient at that snapshot (mu). Then for each
    mini-batch step, use a CORRECTED gradient:

        v = g(w_t) - g_tilde(w_tilde) + mu

    where:
        g(w_t)        = mini-batch gradient at current parameters
        g_tilde(w_tilde) = mini-batch gradient at the SNAPSHOT (same batch, old params)
        mu            = full gradient at snapshot

    Why this works:
        - mu is a low-noise anchor (computed over all data)
        - g(w_t) - g_tilde(w_tilde) corrects for the difference
          between current and snapshot parameters
        - As w_t approaches w_tilde (near convergence), v approaches mu,
          which approaches zero — the noise floor disappears

    Memory cost: O(d) — just stores the snapshot parameters and mu.
    No per-sample table needed (unlike SAGA).

    Downside: Every m inner steps, you pay the cost of one full gradient.
    The inner loop size m controls the trade-off between snapshot cost
    and correction quality.
    """
    def __init__(self, params, lr, n_samples, batch_size, inner_loop_size=None):
        super().__init__(params, lr)

        # Snapshot of parameters at last full gradient computation
        self.snapshot_params = None

        # Full gradient at snapshot: mu = (1/n) sum_{i} grad f_i(w_tilde)
        self.mu = None

        # Gradient at snapshot for current mini-batch: g_tilde(w_tilde)
        # Needed to compute the correction g(w_t) - g_tilde(w_tilde)
        self.snapshot_grads = None

        # How many inner steps between full gradient recomputations.
        # Standard theory sets this to 2n or 5n (n = dataset size).
        # Here we default to one full pass worth of mini-batches.
        if inner_loop_size is None:
            self.inner_loop_size = max(1, n_samples // batch_size)
        else:
            self.inner_loop_size = inner_loop_size

        self.step_count = 0  # tracks inner steps taken since last snapshot

    def update_snapshot(self, model, dataset, loss_fn):
        """
        Recomputes the snapshot: saves current parameters as w_tilde,
        then computes the full gradient mu = (1/n) sum grad f_i(w_tilde).

        Called at the start of each outer loop (every inner_loop_size steps).
        This is the expensive part — costs one full pass over data.
        """
        # Save a copy of current parameters as the new snapshot w_tilde
        self.snapshot_params = [p.data.clone() for p in self.params]

        # Compute full gradient over entire dataset
        model.zero_grad()
        X_full, y_full = dataset[:]
        logits = model(X_full)
        loss = loss_fn(logits, y_full)
        loss.backward()

        # Save full gradient as mu — this is our low-noise anchor
        self.mu = [p.grad.clone() for p in self.params]

        # Reset inner step counter
        self.step_count = 0

    def compute_snapshot_grad(self, X_batch, y_batch, loss_fn):
        """
        Computes g_tilde(w_tilde): the mini-batch gradient evaluated at
        the SNAPSHOT parameters (not the current parameters).

        This is what makes SVRG different from SARAH — SVRG re-evaluates
        the gradient at the old snapshot for every mini-batch step.

        Steps:
            1. Temporarily swap current params with snapshot params
            2. Compute mini-batch gradient
            3. Save it as snapshot_grads
            4. Restore current params
        """
        # Save current parameters so we can restore them after
        current_params = [p.data.clone() for p in self.params]

        # Load snapshot parameters into the model
        with torch.no_grad():
            for p, snap in zip(self.params, self.snapshot_params):
                p.data.copy_(snap)

        # Compute gradient at snapshot using the same mini-batch
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

        logits_snap = torch.nn.functional.linear(
            X_batch,
            self.params[0],   # weight
            self.params[1] if len(self.params) > 1 else None  # bias
        )
        loss_snap = loss_fn(logits_snap, y_batch)
        loss_snap.backward()

        # Save snapshot gradients
        self.snapshot_grads = [p.grad.clone() for p in self.params]

        # Restore current parameters
        with torch.no_grad():
            for p, cur in zip(self.params, current_params):
                p.data.copy_(cur)

    def step(self):
        """
        SVRG corrected gradient update.

        v = g(w_t) - g_tilde(w_tilde) + mu
        w_{t+1} = w_t - lr * v

        The correction g(w_t) - g_tilde(w_tilde) cancels out noise shared
        between both gradients (they use the same mini-batch).
        mu pulls the estimate toward the true full gradient direction.
        """
        with torch.no_grad():
            for p, g, g_snap, mu_i in zip(
                self.params, self.grads, self.snapshot_grads, self.mu
            ):
                # Corrected variance-reduced gradient
                v = g - g_snap + mu_i
                p -= self.lr * v

        self.step_count += 1


# =============================================================================
# HELPER: Full dataset loss (used by SARAH and SVRG outer loops)
# =============================================================================

def full_loss(model, dataset, loss_criterion):
    """
    Computes loss over the entire dataset.
    Used to get the full gradient for SARAH and SVRG outer loops.
    """
    X_full, y_full = dataset[:]
    logits = model(X_full)
    loss = loss_criterion(logits, y_full)
    return loss


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(model, optimizer, loader, epochs, method, dataset):
    """
    Unified training loop for all methods.
    Handles the different outer-loop requirements for SARAH and SVRG.

    Returns:
        history: list of average losses (one per epoch, not per iteration)
                 This gives a smooth curve that is easier to interpret.
    """
    history = []
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):

        # ----------------------------------------------------------
        # SARAH outer loop: full gradient once at start of each epoch
        # ----------------------------------------------------------
        if method == "sarah":
            optimizer.zero_grad()
            loss = full_loss(model, dataset, loss_fn)
            loss.backward()
            optimizer.get_outer_loop()

        # ----------------------------------------------------------
        # SVRG outer loop: initial snapshot before first epoch
        # (subsequent snapshots happen mid-epoch every inner_loop_size steps)
        # ----------------------------------------------------------
        if method == "svrg" and epoch == 0:
            optimizer.update_snapshot(model, dataset, loss_fn)

        # ----------------------------------------------------------
        # Inner loop: mini-batch updates
        # ----------------------------------------------------------
        epoch_losses = []  # collect batch losses, average at end of epoch

        for X, y in loader:
            X, y = X.float(), y.float()

            # SVRG: check if it's time for a new snapshot
            if method == "svrg":
                if optimizer.step_count >= optimizer.inner_loop_size:
                    optimizer.update_snapshot(model, dataset, loss_fn)

                # Compute gradient at snapshot for this mini-batch
                optimizer.compute_snapshot_grad(X, y, loss_fn)

            # Compute gradient at current parameters
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.store_grads()
            optimizer.step()

            epoch_losses.append(loss.item())

        # Average loss for this epoch — one clean data point per epoch
        epoch_avg_loss = np.mean(epoch_losses)
        history.append(epoch_avg_loss)

        print(f"Epoch {epoch + 1}: avg_loss={epoch_avg_loss:.4f}")

        # Evaluation stats at end of epoch
        model.eval()
        with torch.no_grad():
            X_full, y_full = dataset[:]
            raw_scores = model(X_full)
            probs = torch.sigmoid(raw_scores).cpu().numpy().flatten()
            predictions = (probs >= 0.5).astype(int)
            truths = y_full.cpu().numpy().astype(int)

        accuracy  = accuracy_score(truths, predictions)
        precision = precision_score(truths, predictions, zero_division=0)
        recall    = recall_score(truths, predictions, zero_division=0)
        f1        = f1_score(truths, predictions, zero_division=0)
        auc       = roc_auc_score(truths, probs)
        print(
            f"Epoch {epoch + 1} — "
            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}"
        )

        model.train()

    return history


# =============================================================================
# PLOTTING
# =============================================================================

def plot_loss(history, method):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(history) + 1), history, label=method.upper(),
             linewidth=2, marker='o', markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title(f"Loss Curve ({method.upper()})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/plot_{method}_loss.png")
    print(f"Saved plot to plots/plot_{method}_loss.png")


# =============================================================================
# MAIN
# =============================================================================

def main(args):
    # Dataset
    X, y = make_classification(
        n_samples=20000,
        n_features=50,
        n_informative=20,
        n_redundant=5,
        n_repeated=0,
        class_sep=0.5,
        flip_y=0.05,
        random_state=0
    )

    X = StandardScaler().fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape (N, 1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    n_samples = X.shape[0]

    # Model: simple logistic regression (single linear layer)
    model = nn.Linear(X.shape[1], 1)

    # Optimizer selection
    if args.method == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr)

    elif args.method == "sarah":
        optimizer = SARAH(model.parameters(), lr=args.lr)

    elif args.method == "svrg":
        optimizer = SVRG(
            model.parameters(),
            lr=args.lr,
            n_samples=n_samples,
            batch_size=args.batch_size,
            inner_loop_size=args.inner_loop_size
        )

    else:
        raise ValueError("--method must be one of: sgd, sarah, svrg. (saga coming soon)")

    # Train
    history = train(model, optimizer, loader, args.epochs,
                    method=args.method, dataset=dataset)

    # Save results
    with open(f"{args.method}_results.json", "w") as f:
        json.dump({"loss": history}, f, indent=4)

    # Individual loss curve
    plot_loss(history, args.method)

    # Comparison plot — overlays all methods that have been run
    methods = ["sgd", "sarah", "svrg"]
    plt.figure(figsize=(7, 4))
    for m in methods:
        try:
            with open(f"{m}_results.json") as f:
                data = json.load(f)
            epochs_range = range(1, len(data["loss"]) + 1)
            plt.plot(epochs_range, data["loss"], label=m.upper(),
                     linewidth=2, marker='o', markersize=4)
        except FileNotFoundError:
            pass

    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("SGD vs SARAH vs SVRG")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/plot_comparison.png")
    print("Saved comparison plot to plots/plot_comparison.png")


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variance Reduction Methods Comparison")
    parser.add_argument("--method",           type=str,   required=True,
                        help="Optimizer to use: sgd | sarah | svrg")
    parser.add_argument("--lr",               type=float, default=0.1,
                        help="Learning rate (default: 0.1)")
    parser.add_argument("--batch-size",       type=int,   default=128,
                        help="Mini-batch size (default: 128)")
    parser.add_argument("--epochs",           type=int,   default=10,
                        help="Number of epochs (default: 10)")
    parser.add_argument("--inner-loop-size",  type=int,   default=None,
                        help="SVRG inner loop size m (default: n_samples // batch_size)")
    args = parser.parse_args()
    main(args)