import argparse
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import os
os.makedirs("plots", exist_ok=True)

# Noah Jacobson: SGD, SARAH
# Advaith Subramanian Sahasranamam: SAGA
# Kudzaishe Kadzimu: SVRG

# =============================================================================
# BASE CLASS

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
        """
        self.grads = [p.grad.clone() for p in self.params]


# =============================================================================
# SGD

class SGD(OptimizerBase):
    """
    Vanilla Stochastic Gradient Descent.
    Update rule: w = w - lr * g
    where g is the mini-batch gradient.
    """
    def step(self):
        with torch.no_grad():
            for parameter, gradient in zip(self.params, self.grads):
                parameter -= self.lr * gradient
        return float(sum(g.pow(2).sum().item() for g in self.grads) ** 0.5)


# =============================================================================
# SARAH

class SARAH(OptimizerBase):
    def __init__(self, params, lr):
        super().__init__(params, lr)
        self.v_prev = None
        self.grads_prev = None
        self.params_prev = None  # store previous parameters

    def get_outer_loop(self):
        # Full gradient already computed in p.grad
        self.v_prev = [p.grad.clone() for p in self.params]
        self.grads_prev = [p.grad.clone() for p in self.params]

        # Store params BEFORE step
        self.params_prev = [p.clone() for p in self.params]

        with torch.no_grad():
            for p, v0 in zip(self.params, self.v_prev):
                p -= self.lr * v0

    def step(self, closure=None):
        """
        closure: function that recomputes loss + gradients on CURRENT batch
        Must use SAME mini-batch when called twice
        """
        if closure is None:
            raise ValueError("SARAH requires a closure for recomputing gradients")

        # ---- g_t already computed BEFORE calling step() ----
        g_t = [g.clone() for g in self.grads]

        if self.v_prev is None:
            v = g_t
        else:
            # Swap to previous params — no_grad only for the tensor copies
            with torch.no_grad():
                current_params = [p.clone() for p in self.params]
                for p, p_prev in zip(self.params, self.params_prev):
                    p.copy_(p_prev)

            # closure() must run OUTSIDE no_grad so backward() can compute grads
            closure()
            g_prev = [p.grad.clone() for p in self.params]

            with torch.no_grad():
                # Restore current params (w_t)
                for p, p_curr in zip(self.params, current_params):
                    p.copy_(p_curr)

                # SARAH update: v_t = grad(w_t) - grad(w_{t-1}) + v_{t-1}
                v = [g - gp + vp for g, gp, vp in zip(g_t, g_prev, self.v_prev)]

        grad_norm = float(sum(vi.pow(2).sum().item() for vi in v) ** 0.5)

        with torch.no_grad():
            # save w_t (BEFORE stepping) so next iteration can compute grad at w_{t-1}
            self.params_prev = [p.clone() for p in self.params]

            # parameter update
            for p, v_i in zip(self.params, v):
                p -= self.lr * v_i

            # ---- update stored values ----
            self.v_prev = [vi.clone() for vi in v]
            self.grads_prev = [g.clone() for g in g_t]

        return grad_norm

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
        norm_sq = 0.0
        with torch.no_grad():
            for p, g, g_snap, mu_i in zip(
                self.params, self.grads, self.snapshot_grads, self.mu
            ):
                v = g - g_snap + mu_i
                norm_sq += v.pow(2).sum().item()
                p -= self.lr * v

        self.step_count += 1
        return float(norm_sq ** 0.5)



# =============================================================================
# INDEXED DATASET — needed by SAGA
# =============================================================================

class IndexedTensorDataset(torch.utils.data.Dataset):
    """
    Wraps a TensorDataset and also returns the sample index alongside
    the data. SAGA needs to know WHICH sample was picked so it can
    look up and update that sample's stored gradient in the table.

    Normal DataLoader returns: (X_batch, y_batch)
    This returns:              (indices, X_batch, y_batch)
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return idx, self.X[idx], self.y[idx]


# =============================================================================
# SAGA
# =============================================================================

class SAGA(OptimizerBase):
    """
    SAGA — Defazio et al., 2014. https://arxiv.org/abs/1407.0202
    Implementation: Kudzaishe Kadzimu

    Key idea: Maintain a TABLE of the most recent gradient for every
    single training sample. Use these stored gradients to correct
    the stochastic gradient estimate:

        v = g_new(i) - grad_table[i] + grad_avg

    where:
        g_new(i)      = gradient computed on sample i right now
        grad_table[i] = gradient computed on sample i LAST TIME it was picked
        grad_avg      = average of ALL stored gradients in the table

    Why this works:
        - grad_table[i] and g_new(i) share the same data point i,
          so their noise partially cancels
        - grad_avg is a running estimate of the full gradient,
          acting as a low-variance anchor (like mu in SVRG)
        - As training progresses, the table entries get fresher
          and grad_avg approaches the true gradient mean → variance → 0

    Memory cost: O(n × d) — one gradient vector per training sample.
        This is SAGA's main downside vs SVRG/SARAH which only need O(d).
        For 20,000 samples with 50 features: 20000 × 50 = 1,000,000 floats.

    No outer loop needed — SAGA updates the table incrementally,
    one sample at a time, making it simpler to implement than SVRG.
    """
    def __init__(self, params, lr, n_samples, n_features, weight_decay=0.0):
        super().__init__(params, lr)

        self.n            = n_samples
        self.weight_decay = weight_decay

        # Gradient table: shape (n_samples, n_features)
        # grad_table_w[i] = most recent weight gradient for sample i
        # grad_table_b[i] = most recent bias gradient for sample i
        # Initialised to zero — equivalent to assuming zero gradient at start
        self.grad_table_w = np.zeros((n_samples, n_features))
        self.grad_table_b = np.zeros(n_samples)

        # Running average of all stored gradients
        # grad_avg = (1/n) * sum of all rows in grad_table
        # Updated incrementally — no need to sum the whole table each step
        self.grad_avg_w = np.zeros(n_features)
        self.grad_avg_b = 0.0

    def saga_step(self, indices, X_batch, y_batch):
        """
        Performs one SAGA update for a mini-batch.

        For each sample in the batch:
            1. Compute fresh gradient g_new for that sample
            2. Compute corrected gradient: v = g_new - grad_table[i] + grad_avg
            3. Update grad_avg incrementally (avoids full table sum)
            4. Update grad_table[i] with g_new
            5. Take gradient step using average of corrected gradients

        Args:
            indices:  tensor of sample indices for this batch
            X_batch:  feature matrix for this batch
            y_batch:  labels for this batch
        """
        indices = indices.numpy()
        X_np    = X_batch.numpy()
        y_np    = y_batch.numpy().flatten()

        # Accumulate corrected gradients across the batch
        batch_dw = np.zeros_like(self.grad_avg_w)
        batch_db = 0.0

        for j, i in enumerate(indices):
            xi = X_np[j]
            yi = y_np[j]

            # Current model weights and bias (read from PyTorch params)
            w = self.params[0].data.numpy().flatten()  # weight vector
            b = self.params[1].data.numpy().item()     # bias scalar

            # Forward pass: logistic regression probability
            zi  = xi @ w + b
            pi  = 1.0 / (1.0 + np.exp(-zi))   # sigmoid

            # Raw gradient for sample i at current parameters
            g_new_w = (pi - yi) * xi   # shape (d,)
            g_new_b = (pi - yi)        # scalar

            # L2 regularization gradient: d/dw [(lambda/2)||w||^2] = lambda * w
            # Applied only to weights, not bias (standard practice)
            if self.weight_decay > 0.0:
                g_new_w = g_new_w + self.weight_decay * w

            # SAGA corrected gradient: g_new - old_table_entry + running_avg
            dw = g_new_w - self.grad_table_w[i] + self.grad_avg_w
            db = g_new_b - self.grad_table_b[i] + self.grad_avg_b

            batch_dw += dw
            batch_db += db

            # Update running average BEFORE updating the table
            # Formula: avg += (g_new - old_entry) / n
            # This is O(d) — no need to recompute the whole sum
            self.grad_avg_w += (g_new_w - self.grad_table_w[i]) / self.n
            self.grad_avg_b += (g_new_b - self.grad_table_b[i]) / self.n

            # Store fresh gradient in the table for sample i
            self.grad_table_w[i] = g_new_w
            self.grad_table_b[i] = g_new_b

        # Average corrected gradient over the batch
        batch_size = len(indices)
        batch_dw /= batch_size
        batch_db /= batch_size

        grad_norm = float((np.sum(batch_dw ** 2) + batch_db ** 2) ** 0.5)

        with torch.no_grad():
            self.params[0] -= self.lr * torch.tensor(
                batch_dw.reshape(self.params[0].shape), dtype=torch.float32)
            self.params[1] -= self.lr * torch.tensor(
                np.array([batch_db]), dtype=torch.float32)

        return grad_norm



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

def train(model, optimizer, loader, epochs, method, dataset, weight_decay=0.0):
    """
    Unified training loop for all methods.
    Handles the different outer-loop requirements for SARAH and SVRG.
    SAGA uses its own saga_step() and bypasses PyTorch autograd entirely.

    Args:
        weight_decay: L2 regularization strength lambda. Adds (lambda/2)||w||^2
                      to the loss, guaranteeing strong convexity with mu=lambda.
                      Controls condition number kappa = (L+lambda)/lambda.
                      Default 0.0 = no regularization.
    Returns:
        history:    list of average losses (one per epoch)
        grad_norms: list of gradient norms (one per mini-batch iteration)
    """
    history    = []
    grad_norms = []

    def loss_fn(logits, y):
        """
        BCE loss + optional L2 penalty on weights (not bias).
        L2 penalty (lambda/2)||w||^2 guarantees strong convexity,
        enabling linear convergence of SVRG and SAGA.
        """
        bce = nn.BCEWithLogitsLoss()(logits, y)
        if weight_decay > 0.0:
            l2 = (weight_decay / 2.0) * sum(
                p.pow(2).sum()
                for name, p in model.named_parameters()
                if 'bias' not in name
            )
            return bce + l2
        return bce

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

        for batch in loader:

            # SAGA loader returns (indices, X, y); all others return (X, y)
            if method == "saga":
                indices, X, y = batch
                X, y = X.float(), y.float()

                # SAGA bypasses PyTorch autograd entirely — it computes
                # gradients manually using its own sigmoid implementation
                grad_norm = optimizer.saga_step(indices, X, y)
                grad_norms.append(grad_norm)

                # Compute loss separately just for logging (no backward needed)
                with torch.no_grad():
                    logits = model(X)
                    loss = loss_fn(logits, y)

            else:
                X, y = batch
                X, y = X.float(), y.float()

                # SVRG: check if it's time for a new snapshot
                if method == "svrg":
                    if optimizer.step_count >= optimizer.inner_loop_size:
                        optimizer.update_snapshot(model, dataset, loss_fn)
                    optimizer.compute_snapshot_grad(X, y, loss_fn)

                # Compute gradient at current parameters
                optimizer.zero_grad()
                logits = model(X)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.store_grads()

                # ---- SARAH needs closure ----
                if method == "sarah":
                    def closure():
                        optimizer.zero_grad()
                        logits_closure = model(X)   # SAME batch
                        loss_closure = loss_fn(logits_closure, y)
                        loss_closure.backward()
                        return loss_closure

                    grad_norm = optimizer.step(closure)
                else:
                    grad_norm = optimizer.step()

                grad_norms.append(grad_norm)

            epoch_losses.append(loss.item())

        # Average loss for this epoch — one clean data point per epoch
        epoch_avg_loss = np.mean(epoch_losses)
        history.append(epoch_avg_loss)

        print(f"Epoch {epoch + 1}: avg_loss={epoch_avg_loss:.4f}")

        # Evaluation stats at end of epoch
        model.eval()
        with torch.no_grad():
            # IndexedTensorDataset returns (idx, X, y); TensorDataset returns (X, y)
            raw = dataset[:]
            X_full, y_full = (raw[1], raw[2]) if len(raw) == 3 else (raw[0], raw[1])
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

    return history, grad_norms


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
# DATASET LOADER
# =============================================================================

def load_dataset(name):
    """
    Loads and preprocesses a dataset by name.

    Supported options:
        synthetic   — make_classification (20k samples, 50 features)
        a9a         — libsvm, 32561 samples, 123 features. Standard benchmark.

    Returns:
        X: np.ndarray of shape (n_samples, n_features), float32
        y: np.ndarray of shape (n_samples,), float32, values in {0, 1}
    """
    print(f"Loading dataset: {name}")

    if name == "synthetic":
        X, y = make_classification(
            n_samples=20000,
            n_features=500,
            n_informative=10,
            n_redundant=490,
            n_repeated=0,
            class_sep=0.3,
            flip_y=0.1,
            random_state=0
        )

    else:
        # libsvm datasets — requires: pip install libsvmdata
        try:
            from libsvmdata import fetch_libsvm
        except ImportError:
            raise ImportError(
                "libsvmdata is not installed. Run: pip install libsvmdata"
            )

        # Map friendly names to libsvm dataset identifiers
        # phishing: 11055 samples, 68 features — replaces mushrooms as small debug dataset
        # a9a:      32561 samples, 123 features — standard benchmark
        # covtype:  581012 samples, 54 features — large scale
        # cod-rna:  59535 samples, 8 features  — medium scale alternative
        libsvm_names = {
            "phishing": "phishing",
            "a9a":      "a9a",
            "covtype":  "covtype.binary",
            "cod-rna":  "cod-rna",
        }

        if name not in libsvm_names:
            raise ValueError(
                f"Unknown dataset '{name}'. "
                f"Choose from: synthetic | phishing | a9a | covtype | cod-rna"
            )

        try:
            # Use newer API if available (libsvmdata >= 0.5)
            from libsvmdata import fetch_dataset
            X, y = fetch_dataset(libsvm_names[name])
        except ImportError:
            X, y = fetch_libsvm(libsvm_names[name])

        # libsvm data often comes as sparse scipy matrices — convert to dense
        if issparse(X):
            X = X.toarray()

    # Scale features: zero mean, unit variance
    # Critical for gradient methods — prevents features on different scales
    # from causing wildly uneven gradient magnitudes
    X = StandardScaler().fit_transform(X).astype(np.float32)

    # Map labels to {0, 1} — BCEWithLogitsLoss requires binary labels
    # libsvm datasets often use {-1, +1} or {1, 2}
    unique = np.unique(y)
    if set(unique) != {0.0, 1.0}:
        y = (y == unique.max()).astype(np.float32)
    else:
        y = y.astype(np.float32)

    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Class balance: {np.mean(y):.2%} positive")

    return X, y


# =============================================================================
# MAIN
# =============================================================================

def main(args):
    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    X, y = load_dataset(args.dataset)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape (N, 1)

    n_samples  = X.shape[0]
    n_features = X.shape[1]

    # SAGA needs an indexed dataset so it knows which sample index was picked.
    # All other methods use the standard TensorDataset.
    if args.method == "saga":
        dataset = IndexedTensorDataset(X, y)
    else:
        dataset = TensorDataset(X, y)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model: simple logistic regression (single linear layer)
    model = nn.Linear(n_features, 1)

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

    elif args.method == "saga":
        optimizer = SAGA(
            model.parameters(),
            lr=args.lr,
            n_samples=n_samples,
            n_features=n_features,
            weight_decay=args.weight_decay
        )

    else:
        raise ValueError("--method must be one of: sgd | sarah | svrg | saga")

    # Train
    history, grad_norms = train(model, optimizer, loader, args.epochs,
                                method=args.method, dataset=dataset,
                                weight_decay=args.weight_decay)

    # Save results
    with open(f"{args.method}_results.json", "w") as f:
        json.dump({"loss": history, "grad_norms": grad_norms}, f, indent=4)

    # Individual loss curve
    plot_loss(history, args.method)

    # Comparison plot — overlays all methods that have been run
    methods = ["sgd", "sarah", "svrg", "saga"]
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
    plt.title(f"SGD vs SARAH vs SVRG vs SAGA ({args.dataset})")
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
                        help="Optimizer to use: sgd | sarah | svrg | saga")
    parser.add_argument("--dataset",          type=str,   default="synthetic",
                        help="Dataset: synthetic | phishing | a9a | covtype | cod-rna (default: synthetic)")
    parser.add_argument("--lr",               type=float, default=0.1,
                        help="Learning rate (default: 0.1)")
    parser.add_argument("--batch-size",       type=int,   default=128,
                        help="Mini-batch size (default: 128)")
    parser.add_argument("--epochs",           type=int,   default=10,
                        help="Number of epochs (default: 10)")
    parser.add_argument("--inner-loop-size",  type=int,   default=None,
                        help="SVRG inner loop size m (default: n_samples // batch_size)")
    parser.add_argument("--weight-decay",     type=float, default=0.01,
                        help="L2 regularization strength lambda (default: 0.01)")
    args = parser.parse_args()
    main(args)