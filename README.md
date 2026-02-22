# MCMC-Based Machine Unlearning (MCU)

A Python implementation of the importance sampling-based machine unlearning algorithm from:

> **"Markov Chain Monte Carlo-Based Machine Unlearning: Unlearning what needs to be forgotten"**  
> Nguyen, Quoc Phong and Oikawa, Ryutaro and Divakaran, Dinil Mon and Chan, Mun Choon and Low, Bryan Kian Hsiang
> _ACM ASIACCS 2022_

---

## Table of Contents

1. [What is Machine Unlearning?](#1-what-is-machine-unlearning)
2. [Dataset](#2-dataset)
3. [Mathematical Foundation](#3-mathematical-foundation)
4. [Algorithm: MCU via Importance Sampling](#4-algorithm-mcu-via-importance-sampling)
5. [Code Walkthrough](#5-code-walkthrough)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Installation & Usage](#7-installation--usage)
8. [Results Interpretation](#8-results-interpretation)

---

## 1. What is Machine Unlearning?

Machine unlearning is the problem of **removing the influence of specific training data points** from an already-trained model, ideally without retraining from scratch. Motivations include:

- **Right to be Forgotten** (GDPR, CCPA compliance): a user requests their data to be deleted
- **Poisoning attack mitigation**: removing adversarially injected training samples
- **Data correction**: fixing mislabeled or corrupted training data

The naive solution — retraining the model from scratch on the remaining data — is correct but computationally prohibitive for large models or large datasets. MCU provides a principled, efficient alternative using **Bayesian posterior manipulation via importance sampling**.

---

## 2. Dataset

### Synthetic Dataset (UCI Phishing Websites inspired)

The script generates a synthetic tabular binary classification dataset designed to mimic the structure of the [UCI Phishing Websites Dataset](https://archive.ics.uci.edu/ml/datasets/phishing+websites).

| Property             | Value                                                |
| -------------------- | ---------------------------------------------------- |
| Total samples        | 1000                                                 |
| Features             | 15                                                   |
| Informative features | 10                                                   |
| Redundant features   | 2                                                    |
| Class separation     | 1.2                                                  |
| Train / Test split   | 80% / 20%                                            |
| Task                 | Binary Classification (0 = legitimate, 1 = phishing) |

Generated using `sklearn.datasets.make_classification` with `random_state=42` for reproducibility.

### Erase Set (D_e) Configuration

The training set is split into a **retain set D_r** and an **erase set D_e** using one of two modes:

| Mode      | Description                                  | Size              |
| --------- | -------------------------------------------- | ----------------- |
| `random`  | 5 randomly selected training points          | ~5 points (~0.6%) |
| `cluster` | All points where feature 0 > 90th percentile | ~80 points (~10%) |

The `cluster` mode simulates a **realistic attack scenario** where a specific type of data (e.g., from one user or demographic) must be completely purged.

> **Toggle mode in `__main__`:**
>
> ```python
> active_mode = "random"  # or "cluster"
> ```

---

## 3. Mathematical Foundation

### 3.1 Bayesian Logistic Regression

The model places standard Normal priors over parameters and a Bernoulli likelihood:

$$P(w) = \prod_{j=1}^{d} \mathcal{N}(w_j; 0, 1), \quad P(b) = \mathcal{N}(b; 0, 1)$$

$$P(y_i = 1 \mid x_i, w, b) = \sigma(w^\top x_i + b)$$

where $\sigma(\cdot)$ is the sigmoid function. The joint posterior over the full dataset $D$ is:

$$P(\theta \mid D) \propto P(D \mid \theta) \cdot P(\theta)$$

### 3.2 The Unlearning Target

Let $D = D_r \cup D_e$. After unlearning $D_e$, the target distribution is the **retrained posterior**:

$$P(\theta \mid D_r) \propto P(D_r \mid \theta) \cdot P(\theta)$$

Using Bayes' rule, this can be rewritten in terms of the full posterior:

$$P(\theta \mid D_r) = \frac{P(\theta \mid D)}{P(D_e \mid \theta)} \cdot Z^{-1}$$

where $Z = \int \frac{P(\theta \mid D)}{P(D_e \mid \theta)} d\theta$ is a normalizing constant. This factorization is the key insight enabling importance sampling.

### 3.3 Importance Sampling Weights

Given MCMC samples $\{\theta_i\}_{i=1}^{N} \sim P(\theta \mid D)$, the unlearned posterior is approximated by assigning **importance weights**:

$$w_i = \frac{1}{P(D_e \mid \theta_i)} \propto \frac{P(\theta_i \mid D_r)}{P(\theta_i \mid D)}$$

In log-space for numerical stability:

$$\log w_i = -\log P(D_e \mid \theta_i) = -\sum_{(x,y) \in D_e} \log P(y \mid x, \theta_i)$$

Normalized weights (via log-sum-exp):

$$\tilde{w}_i = \frac{w_i}{\sum_{j=1}^{N} w_j} = \text{softmax}(\log w_i)$$

**Intuition:** Samples $\theta_i$ that assign **low probability** to the erased data are upweighted (they are consistent with not having seen $D_e$). Samples that fit $D_e$ well are downweighted.

### 3.4 Weighted Predictive Distribution

The predictive probability for a new point $x^*$ under the unlearned model is:

$$P(y^* = 1 \mid x^*, D_r) \approx \sum_{i=1}^{N} \tilde{w}_i \cdot \sigma(\theta_i^\top x^*)$$

### 3.5 Weight Degeneracy & The Enlarged Candidate Set

When $|D_e|$ is large (cluster mode), many weights near zero causes **weight degeneracy** — only a handful of samples dominate. The MCU paper addresses this with the **Enlarged Candidate Set**:

1. Sample from a flatter prior: $P_{\text{flat}}(\theta) = \mathcal{N}(0, T)$ where $T > 1$
2. Adjust the weight to correct for the prior mismatch:

$$w_i = \frac{P_{\text{true}}(\theta_i)}{P_{\text{flat}}(\theta_i)} \cdot \frac{1}{P(D_e \mid \theta_i)}$$

This broadens the candidate set to cover posterior regions that may shift significantly after unlearning.

---

## 4. Algorithm: MCU via Importance Sampling

```
Input:  Full posterior samples {θ_i} ~ P(θ | D),  erase set D_e
Output: Normalized weights {w̃_i} representing P(θ | D_r)

1. For each posterior sample θ_i:
      Compute log P(D_e | θ_i) = Σ_{(x,y)∈D_e} log Bernoulli(y | σ(θ_i·x))

2. Set log w_i = -log P(D_e | θ_i)

3. Normalize:  log w̃_i = log w_i - logsumexp({log w_j})
               w̃_i = exp(log w̃_i)

4. Return {w̃_i}
```

The resulting weighted empirical distribution $\sum_i \tilde{w}_i \delta_{\theta_i}$ approximates $P(\theta \mid D_r)$.

---

## 5. Code Walkthrough

### `generate_dataset()`

Creates the synthetic dataset. `make_classification` ensures a non-trivial separable structure with correlated features (`n_redundant=2`) and varying informativeness levels.

### `split_erase_data()`

Partitions training data into $D_e$ and $D_r$. The `cluster` mode is more adversarial — it creates a structured, non-random erase set that tests the robustness of the importance sampling approximation.

### `bayesian_logistic_regression()`

Defines the NumPyro probabilistic model:

- **Priors:** `w ~ Normal(0, 1)` per dimension, `b ~ Normal(0, 1)`
- **Likelihood:** `Bernoulli(logits = X·w + b)`

### `train_mcmc()`

Runs the **NUTS (No-U-Turn Sampler)** — a state-of-the-art gradient-based MCMC method that automatically tunes step size and trajectory length:

- 500 warmup steps (tuning phase, discarded)
- 1500 posterior samples retained

### `perform_unlearning()`

Core MCU function. Key shapes:

```
w:               (1500, 15)   # 1500 MCMC samples, 15 features
X_e.T:           (15, |D_e|)
logits:          (1500, |D_e|)
log_likelihoods: (1500, |D_e|)
total_log_lik:   (1500,)      # summed over D_e
log_weights:     (1500,)      # negated
weights:         (1500,)      # normalized
```

### `predict()`

For prediction, integrates over the posterior:

- **Base/Retrained:** Uniform average over all samples
- **Unlearned:** Weighted average using importance weights $\tilde{w}_i$

### `evaluate()`

Orchestrates the full pipeline:

1. Train base model on $D = D_r \cup D_e$
2. Apply MCU to get unlearning weights
3. Retrain from scratch on $D_r$ (ground truth)
4. Compare all three models on test accuracy and Wasserstein distance

---

## 6. Evaluation Metrics

### 6.1 Predictive Accuracy

Measures classification performance on the held-out test set (200 samples). Used to verify that unlearning **does not degrade utility**. Since the test set contains no erased data, all three models should have similar accuracy if the erase set is small.

> Accuracy alone is insufficient for verifying unlearning — a model can forget data while maintaining the same test accuracy.

### 6.2 Wasserstein Distance (Unlearning Fidelity)

The **1-Wasserstein distance** (Earth Mover's Distance) measures the cost of transforming one probability distribution into another. It is computed **per parameter dimension** across all 16 parameters ($w_1, \ldots, w_{15}, b$):

$$W_1(P_{\text{unlearned}}, P_{\text{retrained}}) = \inf_{\gamma \in \Gamma} \int |x - y| \, d\gamma(x, y)$$

For 1D empirical distributions with weights $\tilde{w}_i$ (unlearned) and uniform weights (retrained):

$$W_1 = \int_0^1 |F_u^{-1}(t) - F_r^{-1}(t)| \, dt$$

where $F_u^{-1}$ and $F_r^{-1}$ are the respective quantile functions.

The **average Wasserstein distance** is the mean across all 16 parameter dimensions.

| Distance     | Interpretation                                          |
| ------------ | ------------------------------------------------------- |
| `0.000`      | Perfect unlearning — distributions are identical        |
| `< 0.05`     | Excellent — very close to retrained posterior           |
| `0.05 – 0.2` | Good approximation                                      |
| `> 0.5`      | Poor unlearning — significant residual influence of D_e |

---

## 7. Installation & Usage

### Requirements

```bash
pip install jax jaxlib numpyro scikit-learn scipy numpy
```

> On Windows, install CPU-only JAX:
>
> ```bash
> pip install "jax[cpu]"
> ```

### Running

```bash
python mcmc_unlearning_mcu.py
```

### Switching Deletion Mode

In `mcmc_unlearning_mcu.py`, change line in `__main__`:

```python
# For few random deletions (fast, easy case):
active_mode = "random"

# For structured cluster deletion (~10% of data, harder case):
active_mode = "cluster"
```

---

## 8. Results Interpretation

### Random Mode Output (5 points erased)

```
Dataset Split: D_r (795 samples), D_e (5 samples)

--- Predictive Accuracy (Clean Test Set) ---
Base Model Accuracy      : 0.9100
Unlearned Model Accuracy : 0.9100
Retrained Model Accuracy : 0.9100

--- Unlearning Fidelity ---
Average Wasserstein Distance (Unlearned vs Retrained): 0.0212
```

| Observation                 | Explanation                                                                  |
| --------------------------- | ---------------------------------------------------------------------------- |
| All accuracies equal        | Erasing 5/800 points (~0.6%) has negligible impact on generalization         |
| Wasserstein = 0.0212        | Unlearned posterior is very close to retrained posterior — MCU works well    |
| Small but non-zero distance | Due to Monte Carlo noise and the approximation nature of importance sampling |

### Cluster Mode Expectations (~80 points erased)

- Accuracies may diverge slightly between base and retrained models
- Wasserstein distance will be higher due to larger posterior shift
- Weight degeneracy risk increases — the "Enlarged Candidate Set" modification (see `perform_unlearning` docstring) is recommended for this regime

### What Good Unlearning Looks Like

```
P(θ | D_r)  ≈  Unlearned weighted posterior
               ↕ small Wasserstein distance
P(θ | D_r)     Retrained posterior (ground truth)
```

A **small Wasserstein distance** combined with **maintained accuracy** is the signature of successful, utility-preserving unlearning.

---

## Key References

- Nguyen et al., _"MCMC-Based Machine Unlearning: Unlearning what needs to be forgotten"_, ACM ASIACCS 2022
- Neal, R. M., _"MCMC Using Hamiltonian Dynamics"_, Handbook of Markov Chain Monte Carlo, 2011
- Hoffman & Gelman, _"The No-U-Turn Sampler"_, JMLR 2014
- Villani, C., _"Optimal Transport: Old and New"_, Springer, 2008 (Wasserstein distance)
