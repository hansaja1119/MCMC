"""
MCMC-Based Machine Unlearning (MCU)

This script replicates the importance sampling-based unlearning approach from 
the ASIACCS 2022 paper "Markov Chain Monte Carlo-Based Machine Unlearning".

It trains a Bayesian Logistic Regression model, and then unlearns a subset of
data (D_e) from the obtained posterior using importance sampling. It then 
compares the unlearned posterior to the ground-truth "retrained from scratch" 
posterior.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- 1. Dataset Setup ---
def generate_dataset(n_samples=1000, n_features=15, random_state=42):
    """
    Generates a realistic synthetic tabular binary classification dataset
    mimicking the UCI Phishing Websites Dataset.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=2,
        class_sep=1.2,
        random_state=random_state
    )
    return train_test_split(X, y, test_size=0.2, random_state=random_state)

def split_erase_data(X_train, y_train, mode="random", seed=42):
    """
    Splits the training data into an erase set (D_e) and retain set (D_r).
    
    Modes:
    - 'random': Removes 5 random data points.
    - 'cluster': Removes a clustered subset (~10% of the dataset) mimicking
                 a specific type of attack.
    """
    np.random.seed(seed)
    n_train = len(X_train)
    
    if mode == "random":
        erase_indices = np.random.choice(n_train, 5, replace=False)
    elif mode == "cluster":
        # Simulate removing a cluster by selecting points where feature 0 > threshold
        threshold = np.percentile(X_train[:, 0], 90) # Top 10%
        erase_indices = np.where(X_train[:, 0] > threshold)[0]
    else:
        raise ValueError("Unknown mode.")
        
    erase_mask = np.zeros(n_train, dtype=bool)
    erase_mask[erase_indices] = True
    keep_mask = ~erase_mask
    
    D_e = (X_train[erase_mask], y_train[erase_mask])
    D_r = (X_train[keep_mask], y_train[keep_mask])
    
    return D_e, D_r

# --- 2. Model Definition & Base Training ---
def bayesian_logistic_regression(X, y=None):
    """
    Bayesian Logistic Regression Model
    Standard Normal priors are set for weights and bias.
    """
    n_samples, n_features = X.shape
    
    # Priors
    w = numpyro.sample("w", dist.Normal(jnp.zeros(n_features), jnp.ones(n_features)))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0))
    
    # Logits compute
    logits = jnp.matmul(X, w) + b
    
    # Likelihood
    with numpyro.plate("data", n_samples):
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

def train_mcmc(X, y, num_warmup=500, num_samples=1000, rng_key=random.PRNGKey(0)):
    """
    Trains the model using NUTS (No-U-Turn Sampler).
    """
    kernel = NUTS(bayesian_logistic_regression)
    # Using progress_bar=False to keep standard output relatively clean
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=True)
    mcmc.run(rng_key, X=X, y=y)
    return mcmc.get_samples()

# --- 3. MCU Unlearning Algorithm Phase ---
def perform_unlearning(D_e, posterior_samples):
    """
    MCU Unlearning via Importance Sampling.
    
    Calculates weights w_i = 1 / P(D_e | theta_i) for each posterior sample theta_i.
    Returns normalized weights that represent the Unlearned Posterior.
    """
    X_e, y_e = D_e
    
    w = posterior_samples['w'] # Shape: (num_samples, n_features)
    b = posterior_samples['b'] # Shape: (num_samples,)
    
    # Compute logits for erased data for all samples
    # X_e.T shape: (n_features, n_erased)
    # np.dot(w, X_e.T) shape: (num_samples, n_erased)
    logits = jnp.dot(w, X_e.T) + b[:, None]
    
    # Compute log likelihood P(D_e | theta_i)
    bernoulli_dist = dist.Bernoulli(logits=logits)
    log_likelihoods = bernoulli_dist.log_prob(y_e)
    
    # Sum over the erased data points to get total log likelihood per sample
    total_log_likelihood = jnp.sum(log_likelihoods, axis=1) # Shape: (num_samples,)
    
    # The unlearning weight is w_i = 1 / P(D_e | theta_i)
    # log(w_i) = -log P(D_e | theta_i)
    log_weights = -total_log_likelihood
    
    # Normalize weights (softmax) to avoid numerical instability
    log_weights_norm = log_weights - jax.nn.logsumexp(log_weights)
    weights = jnp.exp(log_weights_norm)
    
    '''
    NOTE: ENLARGED CANDIDATE SET (Flattened Prior)
    If the erased dataset D_e is large (like in our 'cluster' case), many weights might 
    become near-zero, causing weight degeneracy where only a few samples dominate. 
    To prevent this, the MCU paper recommends an "Enlarged Candidate Set". 
    
    Implementation concept:
    1. Base MCMC uses a flatter prior P_{flat}(theta) e.g. dist.Normal(0, T) where T > 1.
    2. The unlearning weight formula is adjusted to re-introduce the true standard prior:
       w_i = ( P_{true_prior}(theta_i) / P_{flat}(theta_i) ) * ( 1 / P(D_e | theta_i) )
    This ensures the Candidate Set covers broader regions of the parameter space, 
    allowing some samples to have high likelihood even for significantly shifted posteriors.
    '''
    return weights

# --- Diagnostic Visualizations ---
def plot_all_diagnostics(D_r, D_e, X_test, y_test,
                         base_samples, retrained_samples,
                         unlearning_weights, distances, mode):
    """
    Generates a 2x3 diagnostic figure covering every key aspect of MCU:

    [1] Dataset Split (PCA 2D)       - Shows retain vs erase data geometry
    [2] Importance Weights           - Reveals weight degeneracy; annotates ESS
    [3] Posterior KDE Comparison     - Overlays Base / Unlearned / Retrained
                                       marginals for the top-3 weight dimensions
    [4] Posterior Trace (Bias b)     - MCMC chain health and posterior shift
    [5] Predictive Probability Dist  - P(y=1|x) histograms for all three models
    [6] Per-Dim Wasserstein Distance - Bar chart over all 16 parameter dimensions
    """
    weights_np = np.array(unlearning_weights)
    X_r, y_r   = D_r
    X_e, y_e   = D_e

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"MCU Diagnostic Dashboard  [{mode.upper()} DELETION MODE]",
        fontsize=15, fontweight="bold", y=1.01
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Plot 1: Dataset split via PCA ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    X_all   = np.vstack([X_r, X_e])
    y_all   = np.concatenate([y_r, y_e])
    labels_all = np.array(["Retain"] * len(X_r) + ["Erase"] * len(X_e))

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_all)
    var  = pca.explained_variance_ratio_

    for cls, marker, color in [(0, "o", "#4C72B0"), (1, "s", "#DD8452")]:
        mask = y_all == cls
        # Retain points
        r_mask = mask & (labels_all == "Retain")
        ax1.scatter(X_2d[r_mask, 0], X_2d[r_mask, 1],
                    marker=marker, c=color, alpha=0.35, s=18, label=f"Retain cls={cls}")
        # Erase points
        e_mask = mask & (labels_all == "Erase")
        ax1.scatter(X_2d[e_mask, 0], X_2d[e_mask, 1],
                    marker=marker, c="red", edgecolors="black", linewidths=0.8,
                    alpha=0.9, s=60, label=f"Erase cls={cls}" if e_mask.any() else "_")

    ax1.set_title("[1] Dataset Split (PCA 2D)", fontweight="bold")
    ax1.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax1.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax1.legend(fontsize=7, loc="best")
    ax1.text(0.02, 0.98,
             f"D_r = {len(X_r)}  |  D_e = {len(X_e)}",
             transform=ax1.transAxes, fontsize=8,
             verticalalignment="top", bbox=dict(boxstyle="round", alpha=0.15))

    # ── Plot 2: Importance weights histogram + ESS ────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    N   = len(weights_np)
    ess = 1.0 / np.sum(weights_np ** 2)          # Effective Sample Size
    ess_pct = ess / N * 100

    ax2.hist(weights_np, bins=60, color="steelblue", edgecolor="white", linewidth=0.4)
    ax2.axvline(1.0 / N, color="crimson", linestyle="--", linewidth=1.5,
                label=f"Uniform (1/N={1/N:.5f})")
    ax2.set_title("[2] Importance Weights Distribution", fontweight="bold")
    ax2.set_xlabel("Normalized Weight $\\tilde{w}_i$")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=8)
    ax2.text(0.97, 0.95,
             f"ESS = {ess:.1f} / {N}\n({ess_pct:.1f}%)",
             transform=ax2.transAxes, fontsize=9, ha="right", va="top",
             color="darkred", fontweight="bold",
             bbox=dict(boxstyle="round", alpha=0.15))
    # Annotate degeneracy warning
    if ess_pct < 10:
        ax2.set_facecolor("#fff0f0")
        ax2.text(0.5, 0.5, "⚠ Weight Degeneracy",
                 transform=ax2.transAxes, fontsize=10, ha="center",
                 color="red", alpha=0.4, fontweight="bold", rotation=15)

    # ── Plot 3: Posterior KDE – top-3 weight dimensions ──────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    w_base      = np.array(base_samples["w"])       # (N, 15)
    w_retrained = np.array(retrained_samples["w"])  # (N, 15)

    # Pick the 3 dimensions with the largest mean absolute value (most influential)
    top3 = np.argsort(np.abs(w_base.mean(axis=0)))[-3:][::-1]
    palette = ["#2196F3", "#4CAF50", "#FF9800"]

    for rank, dim in enumerate(top3):
        col = palette[rank]
        x_base = w_base[:, dim]
        x_ret  = w_retrained[:, dim]

        kde_base = scipy.stats.gaussian_kde(x_base)
        kde_ret  = scipy.stats.gaussian_kde(x_ret)
        kde_unl  = scipy.stats.gaussian_kde(x_base, weights=weights_np)

        x_rng = np.linspace(
            min(x_base.min(), x_ret.min()) - 0.3,
            max(x_base.max(), x_ret.max()) + 0.3, 300
        )
        label_sfx = f" (w[{dim}])"
        ax3.plot(x_rng, kde_base(x_rng),  linestyle="--",  color=col, alpha=0.6,
                 label="Base"      + label_sfx if rank == 0 else "_")
        ax3.plot(x_rng, kde_unl(x_rng),   linestyle="-",   color=col, alpha=0.95, linewidth=2,
                 label="Unlearned" + label_sfx if rank == 0 else "_")
        ax3.plot(x_rng, kde_ret(x_rng),   linestyle=":",   color=col, alpha=0.8,
                 label="Retrained" + label_sfx if rank == 0 else "_")

    # Legend with line styles only
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], linestyle="--", color="grey",        label="Base (trained on D)"),
        Line2D([0], [0], linestyle="-",  color="grey", lw=2,  label="Unlearned (IS weighted)"),
        Line2D([0], [0], linestyle=":",  color="grey",        label="Retrained (ground truth)"),
    ] + [
        Line2D([0], [0], color=palette[i], lw=2, label=f"w[{top3[i]}]") for i in range(3)
    ]
    ax3.set_title("[3] Posterior Marginals – Top-3 Weights", fontweight="bold")
    ax3.set_xlabel("Parameter Value")
    ax3.set_ylabel("Density")
    ax3.legend(handles=legend_handles, fontsize=7, loc="best", ncol=2)

    # ── Plot 4: MCMC trace for bias b ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    b_base      = np.array(base_samples["b"])
    b_retrained = np.array(retrained_samples["b"])

    ax4.plot(b_base,      alpha=0.7, color="#4C72B0", linewidth=0.6, label="Base (on D)")
    ax4.plot(b_retrained, alpha=0.7, color="#55A868", linewidth=0.6, label="Retrained (on D_r)")
    ax4.axhline(np.average(b_base,      weights=weights_np), color="#4C72B0",
                linestyle="--", linewidth=1.5, label="Unlearned mean")
    ax4.axhline(b_retrained.mean(), color="#55A868",
                linestyle="--", linewidth=1.5, label="Retrained mean")
    ax4.set_title("[4] MCMC Trace – Bias $b$", fontweight="bold")
    ax4.set_xlabel("Sample Index")
    ax4.set_ylabel("$b$ value")
    ax4.legend(fontsize=8, loc="best")

    # ── Plot 5: Predictive P(y=1|x) distributions ────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    X_test_jnp  = jnp.array(X_test)
    w_b         = np.array(base_samples["w"])
    b_b         = np.array(base_samples["b"])
    w_r         = np.array(retrained_samples["w"])
    b_r         = np.array(retrained_samples["b"])

    logits_base = np.dot(w_b, X_test.T) + b_b[:, None]      # (N, n_test)
    probs_base  = 1 / (1 + np.exp(-logits_base))

    mean_base_probs = probs_base.mean(axis=0)
    mean_unl_probs  = (probs_base * weights_np[:, None]).sum(axis=0)
    mean_ret_probs  = (1 / (1 + np.exp(-(np.dot(w_r, X_test.T) + b_r[:, None])))).mean(axis=0)

    bins = np.linspace(0, 1, 30)
    ax5.hist(mean_base_probs,  bins=bins, alpha=0.55, color="#4C72B0",
             label="Base",      edgecolor="white", linewidth=0.3)
    ax5.hist(mean_unl_probs,   bins=bins, alpha=0.55, color="#DD8452",
             label="Unlearned", edgecolor="white", linewidth=0.3)
    ax5.hist(mean_ret_probs,   bins=bins, alpha=0.55, color="#55A868",
             label="Retrained", edgecolor="white", linewidth=0.3)
    ax5.axvline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Decision (0.5)")
    ax5.set_title("[5] Predictive $P(y{=}1 \\mid x)$ on Test Set", fontweight="bold")
    ax5.set_xlabel("Mean Predicted Probability")
    ax5.set_ylabel("Count (test points)")
    ax5.legend(fontsize=8)

    # ── Plot 6: Per-dimension Wasserstein distances ───────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    n_w_dims   = np.array(base_samples["w"]).shape[1]
    dim_labels = [f"w[{i}]" for i in range(n_w_dims)] + ["b"]
    colors_bar = ["#c0392b" if d > np.mean(distances) else "#2980b9" for d in distances]

    bars = ax6.bar(dim_labels, distances, color=colors_bar, edgecolor="white", linewidth=0.4)
    ax6.axhline(np.mean(distances), color="black", linestyle="--",
                linewidth=1.5, label=f"Mean = {np.mean(distances):.4f}")
    ax6.set_title("[6] Per-Dim Wasserstein Distance", fontweight="bold")
    ax6.set_xlabel("Parameter Dimension")
    ax6.set_ylabel("$W_1$ Distance")
    ax6.set_xticklabels(dim_labels, rotation=60, ha="right", fontsize=7)
    ax6.legend(fontsize=8)
    # Color bar legend
    from matplotlib.patches import Patch
    ax6.legend(handles=[
        bars[0],  # proxy
        Patch(color="#c0392b", label="Above mean"),
        Patch(color="#2980b9", label="Below mean"),
        plt.Line2D([0], [0], color="black", linestyle="--",
                   label=f"Mean = {np.mean(distances):.4f}")
    ], fontsize=7)

    plt.tight_layout()
    fname = f"mcu_diagnostics_{mode}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\n[Diagnostics saved → {fname}]")
    plt.show()


def predict(X, posterior_samples, weights=None):
    """
    Calculates weighted predictive means for classification.
    """
    w = posterior_samples['w']
    b = posterior_samples['b']
    
    logits = jnp.dot(w, X.T) + b[:, None]
    probs = jax.nn.sigmoid(logits) # Shape: (num_samples, n_test_points)
    
    if weights is None:
        mean_probs = jnp.mean(probs, axis=0) # Unweighted Mean for standard MCMC
    else:
        # Weighted mean for the unlearned model
        mean_probs = jnp.sum(probs * weights[:, None], axis=0)
        
    y_pred = (mean_probs > 0.5).astype(int)
    return y_pred

# --- 4. Evaluation & Verification ---
def evaluate(D_r, D_test, D_e, mode="random"):
    X_r, y_r = D_r
    X_test, y_test = D_test
    
    print(f"\n[{mode.upper()} DELETION MODE] Evaluation")
    
    # Stage 1: Base Model Training (Candidate Set)
    print("Training Base Model on full dataset D (D_r + D_e)...")
    X_full = np.vstack((X_r, D_e[0]))
    y_full = np.concatenate((y_r, D_e[1]))
    base_samples = train_mcmc(X_full, y_full, num_warmup=500, num_samples=1500, rng_key=random.PRNGKey(1))
    
    # Stage 2: MCU Unlearning
    print("Applying MCU Importance Sampling for unlearning...")
    unlearning_weights = perform_unlearning(D_e, base_samples)
    
    # Stage 3: Retrain from scratch (Ground Truth)
    print("Retraining Base Model entirely on D_r (Ground Truth)...")
    retrained_samples = train_mcmc(X_r, y_r, num_warmup=500, num_samples=1500, rng_key=random.PRNGKey(2))
    
    # -- Compare Predictives --
    y_pred_base = predict(X_test, base_samples)
    y_pred_unlearned = predict(X_test, base_samples, weights=unlearning_weights)
    y_pred_retrained = predict(X_test, retrained_samples)
    
    acc_base = accuracy_score(y_test, y_pred_base)
    acc_unlearned = accuracy_score(y_test, y_pred_unlearned)
    acc_retrained = accuracy_score(y_test, y_pred_retrained)
    
    print(f"\n--- Predictive Accuracy (Clean Test Set) ---")
    print(f"Base Model Accuracy      : {acc_base:.4f}")
    print(f"Unlearned Model Accuracy : {acc_unlearned:.4f}")
    print(f"Retrained Model Accuracy : {acc_retrained:.4f}")
    
    # -- Statistical Distance (Wasserstein) --
    # Compare 1D Wasserstein distances of marginal parameter distributions
    # between the Unlearned model (weighted original samples) and Retrained model.
    distances = []
    
    for param_name in ['w', 'b']:
        s_unlearned = base_samples[param_name]
        s_retrained = retrained_samples[param_name]
        
        # ensure 2D: (num_samples, num_features)
        if s_unlearned.ndim == 1:
            s_unlearned = s_unlearned.reshape(-1, 1)
            s_retrained = s_retrained.reshape(-1, 1)
            
        n_dims = s_unlearned.shape[1]
        for d in range(n_dims):
            # Compute 1D Wasserstein distance
            # Unlearned uses the importance sampling weights, retrained is uniform
            wd = scipy.stats.wasserstein_distance(
                u_values=np.array(s_unlearned[:, d]), 
                v_values=np.array(s_retrained[:, d]), 
                u_weights=np.array(unlearning_weights)
            )
            distances.append(wd)
            
    avg_wasserstein = np.mean(distances)
    print(f"\n--- Unlearning Fidelity ---")
    print(f"Average Wasserstein Distance (Unlearned vs Retrained): {avg_wasserstein:.4f}")

    # -- Visualisations --
    print("\nGenerating diagnostic plots...")
    plot_all_diagnostics(
        D_r, D_e, X_test, y_test,
        base_samples, retrained_samples,
        unlearning_weights, distances, mode
    )

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    print("Generating Synthetic Dataset...")
    X_train, X_test, y_train, y_test = generate_dataset()
    D_test = (X_test, y_test)
    
    # ==========================================
    # TOGGLE DELETION MODE HERE: 'random' or 'cluster'
    # ==========================================
    # Change to "cluster" to remove ~10% of data (mimicking a specific attack type)
    active_mode = "cluster" 
    
    D_e, D_r = split_erase_data(X_train, y_train, mode=active_mode)
    print(f"Dataset Split: D_r ({len(D_r[0])} samples), D_e ({len(D_e[0])} samples)")
    
    evaluate(D_r, D_test, D_e, mode=active_mode)
