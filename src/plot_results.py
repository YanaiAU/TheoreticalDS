"""EDA + model-comparison figures for road accident severity prediction."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Story colors: grey baselines → blue linear → orange RF trap → green trees → dark champion
COLORS = {
    "majority_baseline": "#c8c8c8",
    "naive_logreg": "#9aa0a6",
    "logistic_regression": "#5b7c99",
    "linear_svm": "#3d5a73",
    "random_forest": "#c45c26",
    "hist_gradient_boosting": "#2f6f4e",
    "catboost": "#0b3d2e",
}

DISPLAY = {
    "majority_baseline": "0 Majority (always light)",
    "naive_logreg": "1 Naive logistic regression",
    "logistic_regression": "2 Full logistic regression",
    "linear_svm": "3 Linear SVM",
    "random_forest": "4 Random forest",
    "hist_gradient_boosting": "5 HistGradientBoosting",
    "catboost": "6 CatBoost (final)",
}


def fig0_eda() -> None:
    df = pd.read_csv(DATA / "accidents_2020_2024.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    sev_map = {1: "fatal", 2: "severe", 3: "light"}
    sev = df["HUMRAT_TEUNA"].value_counts().sort_index()
    axes[0].bar([sev_map[i] for i in sev.index], sev.values, color=["#8b1e1e", "#c45c26", "#4a7c59"])
    axes[0].set_title("Severity codes (raw)")
    axes[0].set_ylabel("count")

    bin_counts = pd.Series(
        {
            "light": int((df["HUMRAT_TEUNA"] == 3).sum()),
            "serious\n(fatal+severe)": int((df["HUMRAT_TEUNA"] <= 2).sum()),
        }
    )
    axes[1].bar(bin_counts.index, bin_counts.values, color=["#4a7c59", "#8b1e1e"])
    axes[1].set_title("Model target (binary)")

    by_year = df.groupby("SHNAT_TEUNA").size()
    axes[2].plot(by_year.index, by_year.values, marker="o", color="#2f4f6f")
    axes[2].set_title("Accidents per year")
    axes[2].set_xlabel("year")
    axes[2].set_ylim(bottom=0)

    fig.suptitle("Israel CBS road accidents with casualties, PUF 2020–2024", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig0_eda.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _labels(cmp: pd.DataFrame) -> list[str]:
    return [DISPLAY.get(m, m) for m in cmp["model"]]


def fig_models() -> None:
    cmp = pd.read_csv(RESULTS / "model_comparison.csv")
    # keep story order if present
    if "act" in cmp.columns:
        cmp = cmp.copy()
    names = _labels(cmp)
    colors = [COLORS.get(m, "#444") for m in cmp["model"]]

    # Fig1: macro-F1 story arc
    fig, ax = plt.subplots(figsize=(10, 4.8))
    y = np.arange(len(cmp))
    ax.barh(y, cmp["macro_f1"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Macro-F1")
    ax.set_title("The story — Macro-F1 on held-out 20%")
    for i, v in enumerate(cmp["macro_f1"]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig1_macro_f1.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Fig2: accuracy trap vs honest metrics (dual)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(cmp))
    w = 0.35
    ax.bar(x - w / 2, cmp["accuracy"], width=w, color="#b0b0b0", label="Accuracy (can mislead)")
    ax.bar(x + w / 2, cmp["macro_f1"], width=w, color=[COLORS.get(m, "#444") for m in cmp["model"]], label="Macro-F1 (honest)")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY.get(m, m).split(" ", 1)[-1][:18] for m in cmp["model"]], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Accuracy trap: high accuracy ≠ good serious-class detection")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig2_accuracy_trap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Fig3: ROC-AUC
    fig, ax = plt.subplots(figsize=(10, 4.8))
    auc = cmp["roc_auc"].fillna(0.5)
    ax.barh(y, auc, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Ranking quality (ROC-AUC)")
    for i, v in enumerate(auc):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig3_auc.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Fig4: serious-class F1 — the public-safety metric
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh(y, cmp["f1_serious"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("F1 on serious class")
    ax.set_title("Can we catch serious accidents? (F1 serious)")
    for i, v in enumerate(cmp["f1_serious"]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig4_serious_f1.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig0_eda()
    if (RESULTS / "model_comparison.csv").exists():
        fig_models()
    print("plots written to", RESULTS)


if __name__ == "__main__":
    main()
