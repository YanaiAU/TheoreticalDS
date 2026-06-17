"""
Learning-theory analysis for the movie-rating project.

VC dimension, empirical Rademacher complexity, PAC bounds, and
generalization-gap visualisations tied to course topics 1–3.

Run: py -3 learning_theory.py
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_movie_project import (
    DATA_DIR,
    FIG_DIR,
    RANDOM_STATE,
    build_features,
    classification_models,
    get_feature_columns,
    load_movies,
)

warnings.filterwarnings("ignore")

THEORY_DIR = FIG_DIR / "theory"
THEORY_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# VC dimension
# ---------------------------------------------------------------------------

def linear_vc_dimension(n_features: int) -> int:
    """Affine hyperplanes in R^d have VC dimension d + 1."""
    return n_features + 1


def mlp_vc_upper_bound(n_params: int) -> float:
    """Classic rough upper bound: O(W * log(W)) for W trainable weights."""
    if n_params <= 1:
        return 1.0
    return n_params * math.log2(n_params)


def count_mlp_params(n_features: int, hidden: tuple[int, ...] = (64, 32)) -> int:
    layers = [n_features, *hidden, 1]
    total = 0
    for a, b in zip(layers, layers[1:]):
        total += a * b + b
    return total


def count_rf_upper_bound(n_estimators: int, max_depth: int) -> int:
    """Very loose upper bound on threshold parameters (capacity proxy)."""
    return n_estimators * (2 ** (max_depth + 1) - 1)


def vc_summary(n_features: int) -> pd.DataFrame:
    mlp_params = count_mlp_params(n_features)
    rows = [
        {
            "hypothesis_class": "Linear (Logistic Regression)",
            "vc_or_bound": linear_vc_dimension(n_features),
            "kind": "exact VC dim",
            "note": f"Hyperplanes in R^{n_features}",
        },
        {
            "hypothesis_class": "SVM (RBF kernel)",
            "vc_or_bound": np.nan,
            "kind": "implicit high-dim",
            "note": "Kernel map -> very large effective capacity; bound via margin, not small VC",
        },
        {
            "hypothesis_class": "Random Forest (300 × depth 12)",
            "vc_or_bound": count_rf_upper_bound(300, 12),
            "kind": "capacity proxy",
            "note": "Tree ensembles shatter easily; depth/ensemble size control overfitting",
        },
        {
            "hypothesis_class": f"MLP (64->32, {mlp_params:,} weights)",
            "vc_or_bound": mlp_vc_upper_bound(mlp_params),
            "kind": "upper bound O(W log W)",
            "note": "Neural-net VC bounds are loose; regularisation matters",
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rademacher complexity (empirical Monte Carlo)
# ---------------------------------------------------------------------------

def labels_to_signs(y: np.ndarray) -> np.ndarray:
    return np.where(y.astype(int) == 1, 1, -1)


def empirical_rademacher_complexity(
    hypothesis_signs: np.ndarray,
    rng: np.random.Generator,
    n_trials: int = 4000,
) -> float:
    """
    R̂_n(F) = E_σ [ sup_{h∈F} (1/n) |Σ σ_i h(x_i)| ]

    hypothesis_signs: shape (K, n), entries in {-1, +1}
    """
    n = hypothesis_signs.shape[1]
    max_vals = np.empty(n_trials)
    for t in range(n_trials):
        sigma = rng.choice([-1, 1], size=n)
        correlations = np.abs(hypothesis_signs @ sigma) / n
        max_vals[t] = correlations.max()
    return float(max_vals.mean())


def random_linear_hypotheses(X_scaled: np.ndarray, n_hypotheses: int, rng: np.random.Generator) -> np.ndarray:
    """Random hyperplanes: sign(w^T x)."""
    d = X_scaled.shape[1]
    signs = np.empty((n_hypotheses, X_scaled.shape[0]), dtype=np.int8)
    for k in range(n_hypotheses):
        w = rng.normal(size=d)
        w /= np.linalg.norm(w) + 1e-12
        signs[k] = np.where(X_scaled @ w >= 0, 1, -1)
    return signs


def linear_rademacher_theory_bound(X_scaled: np.ndarray, weight_bound: float = 1.0) -> float:
    """
    For ||w|| ≤ B and scaled features: R̂_n ≤ B * ||X||_F / (n * sqrt(n))
    (standard norm-based bound for linear classes).
    """
    n = X_scaled.shape[0]
    fro = np.linalg.norm(X_scaled, ord="fro")
    return weight_bound * fro / (n * math.sqrt(n))


# ---------------------------------------------------------------------------
# PAC / VC generalization bound (finite VC)
# ---------------------------------------------------------------------------

def vc_generalization_bound(empirical_error: float, vc_dim: int, n: int, delta: float = 0.05) -> float:
    """
    With prob ≥ 1-δ (standard VC bound, binary classification, 0-1 loss):

        R(h) ≤ R̂(h) + sqrt( 8/n * ( VC*log(2en/VC) + log(4/δ) ) )

    Returns the RHS (worst-case upper bound on true error).
    """
    if vc_dim <= 0 or n <= 0:
        return 1.0
    vc_term = vc_dim * math.log2(2 * math.e * n / vc_dim)
    penalty = math.sqrt(8.0 / n * (vc_term + math.log(4.0 / delta)))
    return min(1.0, empirical_error + penalty)


def sample_complexity_bound(vc_dim: int, epsilon: float, delta: float) -> int:
    """PAC sample complexity ~ O( (VC + log(1/δ)) / ε² ) — illustrative constant."""
    return math.ceil(8.0 * (vc_dim * math.log2(vc_dim + 1) + math.log(1.0 / delta)) / (epsilon ** 2))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_vc_vs_sample_size(vc_dim: int, n_actual: int) -> None:
    ns = np.arange(50, max(n_actual * 2, 5000), 50)
    epsilons = [0.05, 0.10, 0.15]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for eps in epsilons:
        bounds = [vc_generalization_bound(0.0, vc_dim, int(n), delta=0.05) for n in ns]
        axes[0].plot(ns, bounds, label=f"ε proxy={eps:.0%} (δ=0.05, R̂=0)")
        axes[0].axvline(n_actual, color="crimson", ls="--", label=f"our n_train ~ {n_actual:,}")
    axes[0].set_xlabel("Training sample size n")
    axes[0].set_ylabel("PAC upper bound on true error")
    axes[0].set_title(f"VC generalization bound (VC dim = {vc_dim})")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1)

    m_needed = [sample_complexity_bound(vc_dim, eps, 0.05) for eps in epsilons]
    axes[1].bar([f"ε={e:.0%}" for e in epsilons], m_needed, color=["#4C72B0", "#55A868", "#C44E52"])
    axes[1].axhline(n_actual, color="crimson", ls="--", label=f"our n_train ~ {n_actual:,}")
    axes[1].set_ylabel("Illustrative PAC sample size m")
    axes[1].set_title("Sample complexity (order of magnitude)")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(THEORY_DIR / "01_vc_pac_bounds.png", dpi=140)
    plt.close(fig)


def plot_rademacher_comparison(
    rad_linear_mc: float,
    rad_linear_theory: float,
    rad_random_lines: float,
    rad_fitted: dict[str, float],
) -> None:
    names = ["Linear (fitted)", "Random hyperplanes", "Fitted models"] + list(rad_fitted.keys())
    values = [rad_linear_mc, rad_random_lines, np.nan] + list(rad_fitted.values())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bar_names = ["Linear (fitted)", "Random hyperplanes\n(200 lines)", "Theory bound\n(linear, ||w||≤1)"]
    bar_vals = [rad_linear_mc, rad_random_lines, rad_linear_theory]
    axes[0].bar(bar_names, bar_vals, color=["#4C72B0", "#DD8452", "#55A868"])
    axes[0].set_ylabel("Empirical Rademacher complexity R̂_n")
    axes[0].set_title("Rademacher complexity — linear classes")
    axes[0].set_ylim(0, max(bar_vals) * 1.25)

    fitted_names = list(rad_fitted.keys())
    fitted_vals = list(rad_fitted.values())
    axes[1].barh(fitted_names, fitted_vals, color="#8172B2")
    axes[1].set_xlabel("R̂_n (single hypothesis in class)")
    axes[1].set_title("Per-model correlation with random noise")
    axes[1].invert_yaxis()

    plt.tight_layout()
    fig.savefig(THEORY_DIR / "02_rademacher_complexity.png", dpi=140)
    plt.close(fig)


def plot_generalization_gap(gap_rows: list[dict]) -> None:
    df = pd.DataFrame(gap_rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["train_error"], w, label="Train error (1 - acc)", color="#4C72B0")
    ax.bar(x + w / 2, df["test_error"], w, label="Test error", color="#C44E52")
    ax2 = ax.twinx()
    ax2.plot(x, df["gap"], "ko-", label="Gap (test − train)")
    ax2.set_ylabel("Generalization gap")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=15, ha="right")
    ax.set_ylabel("Error rate")
    ax.set_title("Train vs test error — complexity vs overfitting")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(THEORY_DIR / "03_generalization_gap.png", dpi=140)
    plt.close(fig)


def plot_learning_curves(X: pd.DataFrame, y: pd.Series) -> None:
    train_sizes = np.linspace(0.1, 1.0, 8)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    specs = [
        ("Low VC (Logistic Regression)", Pipeline([
            ("prep", StandardScaler()),
            ("model", __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(
                max_iter=2000, random_state=RANDOM_STATE)),
        ])),
        ("High capacity (MLP)", Pipeline([
            ("prep", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.001, max_iter=600,
                                    early_stopping=True, random_state=RANDOM_STATE)),
        ])),
    ]

    for ax, (title, est) in zip(axes, specs):
        sizes, train_scores, val_scores = learning_curve(
            est, X, y, train_sizes=train_sizes, cv=4, scoring="accuracy", n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        ax.plot(sizes, train_scores.mean(axis=1), "o-", label="Train")
        ax.plot(sizes, val_scores.mean(axis=1), "o-", label="CV val")
        ax.fill_between(sizes,
                        val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1),
                        alpha=0.15)
        ax.set_xlabel("Training set size")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend()
        ax.set_ylim(0.5, 1.0)

    plt.tight_layout()
    fig.savefig(THEORY_DIR / "04_learning_curves.png", dpi=140)
    plt.close(fig)


def plot_shattering_intuition_2d(X_scaled: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> None:
    """
    2D PCA slice: demonstrate that 3 points in general position in R^2
    can be shattered by lines (VC dimension of affine lines in R^2 = 3).
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X2 = pca.fit_transform(X_scaled)

    idx = rng.choice(len(y), size=3, replace=False)
    pts = X2[idx]
    labels = y[idx]

    # Enumerate all 2^3 = 8 labelings on these 3 points
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.ravel()

    for ax_idx, bits in enumerate(range(8)):
        ax = axes[ax_idx]
        target = np.array([(bits >> i) & 1 for i in range(3)])
        ax.scatter(pts[:, 0], pts[:, 1], c=np.where(target, "#C44E52", "#4C72B0"), s=120, edgecolors="k")

        # Fit a line w^T x + b = 0 separating the labeling (if possible)
        sep = target * 2 - 1  # ±1
        w_found, b_found = None, None
        for _ in range(500):
            w = rng.normal(size=2)
            w /= np.linalg.norm(w) + 1e-12
            margins = sep * (pts @ w)
            b_low = margins.max()
            b_high = margins.min()
            if b_low < b_high:
                b_found = (b_low + b_high) / 2
                w_found = w
                break

        if w_found is not None:
            xs = np.linspace(pts[:, 0].min() - 0.5, pts[:, 0].max() + 0.5, 50)
            ys = -(w_found[0] * xs + b_found) / (w_found[1] + 1e-12)
            ax.plot(xs, ys, "k--", lw=1.5)
            ax.set_title(f"labeling {list(target)} OK", fontsize=9)
        else:
            ax.set_title(f"labeling {list(target)}", fontsize=9)

        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("VC intuition: 3 points in R² — all 8 labelings separable by a line (VC = 3)", y=1.02)
    plt.tight_layout()
    fig.savefig(THEORY_DIR / "05_shattering_intuition_2d.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)

    raw = load_movies()
    df = build_features(raw)
    feature_cols = get_feature_columns(df)
    model_df = df.dropna(subset=feature_cols + ["High_Rated"])
    X = model_df[feature_cols].values
    y = model_df["High_Rated"].values
    y_sign = labels_to_signs(y)

    n_samples, n_features = X.shape
    n_train = int(0.8 * n_samples)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- VC dimension ---
    vc_linear = linear_vc_dimension(n_features)
    vc_table = vc_summary(n_features)

    print("=" * 70)
    print("VC DIMENSION & CAPACITY")
    print("=" * 70)
    print(f"Feature dimension d = {n_features}")
    print(f"Sample size n = {n_samples:,}  (train ~ {n_train:,})")
    print(f"Ratio n / VC_linear = {n_samples / vc_linear:.0f}  (PAC: want n >> VC)")
    print()
    print(vc_table.to_string(index=False))

    # --- PAC bounds for logistic regression on train set ---
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train_s = scaler.fit_transform(X_train)

    log_pipe = classification_models()["1. Logistic Regression (L2)"]
    log_pipe.fit(X_train, y_train)
    train_err = 1.0 - accuracy_score(y_train, log_pipe.predict(X_train))
    test_err = 1.0 - accuracy_score(y_test, log_pipe.predict(X_test))
    pac_bound = vc_generalization_bound(train_err, vc_linear, len(y_train))

    print()
    print("=" * 70)
    print("PAC / VC GENERALIZATION BOUND — Logistic Regression")
    print("=" * 70)
    print(f"Empirical train error R_hat(h) = {train_err:.4f}")
    print(f"True test error (estimate)     = {test_err:.4f}")
    print(f"VC bound on true error        <= {pac_bound:.4f}  (delta=0.05, VC={vc_linear})")
    print(f"Gap test-train                 = {test_err - train_err:+.4f}")
    print(f"Illustrative m for eps=10%     ~ {sample_complexity_bound(vc_linear, 0.10, 0.05):,} samples")

    plot_vc_vs_sample_size(vc_linear, n_train)

    # --- Rademacher complexity ---
    print()
    print("=" * 70)
    print("EMPIRICAL RADEMACHER COMPLEXITY")
    print("=" * 70)

    # Fitted logistic as single hypothesis
    log_signs = np.where(log_pipe.predict(X_train) == 1, 1, -1).reshape(1, -1)
    rad_log = empirical_rademacher_complexity(log_signs, rng, n_trials=3000)

    # Random linear hypotheses
    rand_signs = random_linear_hypotheses(X_train_s, n_hypotheses=200, rng=rng)
    rad_random = empirical_rademacher_complexity(rand_signs, rng, n_trials=3000)

    rad_theory = linear_rademacher_theory_bound(X_train_s, weight_bound=1.0)

    # Per fitted model (single-h classifier Rademacher = E[|σ·h|]/n)
    rad_fitted = {}
    gap_rows = []
    for name, pipe in classification_models().items():
        pipe.fit(X_train, y_train)
        pred_train = pipe.predict(X_train)
        pred_test = pipe.predict(X_test)
        h_sign = np.where(pred_train == 1, 1, -1).reshape(1, -1)
        rad_fitted[name] = empirical_rademacher_complexity(h_sign, rng, n_trials=2000)
        gap_rows.append({
            "model": name.split(". ", 1)[-1],
            "train_error": 1.0 - accuracy_score(y_train, pred_train),
            "test_error": 1.0 - accuracy_score(y_test, pred_test),
            "gap": (1.0 - accuracy_score(y_test, pred_test)) - (1.0 - accuracy_score(y_train, pred_train)),
        })

    print(f"R_hat_n -- fitted logistic (single h)     : {rad_log:.4f}")
    print(f"R_hat_n -- 200 random hyperplanes (class) : {rad_random:.4f}")
    print(f"Theory bound (||w||<=1, linear)         : {rad_theory:.4f}")
    print()
    print("Per-model R_hat_n (how much each classifier aligns with random noise):")
    for name, val in sorted(rad_fitted.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {val:.4f}")

    plot_rademacher_comparison(rad_log, rad_theory, rad_random, rad_fitted)
    plot_generalization_gap(gap_rows)
    plot_learning_curves(model_df[feature_cols], model_df["High_Rated"])
    plot_shattering_intuition_2d(X_scaled, y, rng)

    # Save tables
    vc_table.to_csv(DATA_DIR / "vc_dimension_summary.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(DATA_DIR / "generalization_gaps.csv", index=False)

    print()
    print(f"Theory figures saved to: {THEORY_DIR}")
    print("  01_vc_pac_bounds.png")
    print("  02_rademacher_complexity.png")
    print("  03_generalization_gap.png")
    print("  04_learning_curves.png")
    print("  05_shattering_intuition_2d.png")


if __name__ == "__main__":
    main()
