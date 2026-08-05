"""
plot_results.py — comparison bar chart + confusion matrix from saved preds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PICKLES = ROOT / "pickles"
RESULTS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")


def main():
    cmp = pd.read_csv(RESULTS / "model_comparison.csv")
    preds = joblib.load(PICKLES / "test_preds.joblib")

    fig, ax = plt.subplots(figsize=(9, 5))
    order = cmp.sort_values("f1_ai")
    ax.barh(order["model"], order["f1_ai"], color="#264653")
    ax.set_xlabel("F1 (AI class)")
    ax.set_title("Model comparison — AI detection F1")
    for y, v in enumerate(order["f1_ai"]):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig5_model_comparison.png", dpi=140)
    plt.close(fig)

    cm = confusion_matrix(preds["y_true"], preds["y_pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Human", "AI"],
        yticklabels=["Human", "AI"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {preds['best_model']}")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig6_confusion_matrix.png", dpi=140)
    plt.close(fig)

    # ROC-ish summary bars
    fig, ax = plt.subplots(figsize=(9, 5))
    melted = cmp.melt(
        id_vars="model",
        value_vars=["roc_auc", "pr_auc"],
        var_name="metric",
        value_name="score",
    )
    sns.barplot(data=melted, x="model", y="score", hue="metric", ax=ax)
    ax.set_ylim(0.5, 1.02)
    ax.set_title("ROC-AUC vs PR-AUC")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig7_auc_metrics.png", dpi=140)
    plt.close(fig)

    print("Wrote fig5-fig7 ->", RESULTS)


if __name__ == "__main__":
    main()
