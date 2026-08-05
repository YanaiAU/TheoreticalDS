"""EDA + model comparison figures for TrafficSpark."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

COLORS = {
    "logistic_regression": "#8a8a8a",
    "linear_svm": "#a0a0a0",
    "random_forest": "#1f6f4a",
    "hist_gradient_boosting": "#0b3d2e",
}


def fig0_eda() -> None:
    df = pd.read_csv(DATA / "accidents_2020_2024.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    sev_map = {1: "fatal", 2: "severe", 3: "light"}
    sev = df["HUMRAT_TEUNA"].value_counts().sort_index()
    axes[0].bar([sev_map[i] for i in sev.index], sev.values, color=["#8b1e1e", "#c45c26", "#4a7c59"])
    axes[0].set_title("Severity codes (raw)")
    axes[0].set_ylabel("count")

    # binary target used in models
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

    fig.suptitle("Israel CBS road accidents PUF 2020–2024", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig0_eda.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_models() -> None:
    cmp = pd.read_csv(RESULTS / "model_comparison.csv").sort_values("macro_f1")
    names = cmp["model"].tolist()
    colors = [COLORS.get(n, "#444") for n in names]

    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.barh(names, cmp["macro_f1"], color=colors)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Serious vs light — Macro-F1 (held-out 20%)")
    for i, v in enumerate(cmp["macro_f1"]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig1_macro_f1.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    metric = "roc_auc" if "roc_auc" in cmp.columns else "balanced_accuracy"
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.barh(names, cmp[metric], color=colors)
    ax.set_xlabel(metric)
    ax.set_title(f"Serious vs light — {metric}")
    for i, v in enumerate(cmp[metric]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig2_auc.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, r in cmp.iterrows():
        ax.scatter(r["train_seconds"], r["macro_f1"], s=140, color=COLORS.get(r["model"], "#444"), zorder=3)
        ax.annotate(r["model"], (r["train_seconds"], r["macro_f1"]), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Train time (s)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Quality vs train time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig3_pareto.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig0_eda()
    if (RESULTS / "model_comparison.csv").exists():
        fig_models()
    print("plots written to", RESULTS)


if __name__ == "__main__":
    main()
