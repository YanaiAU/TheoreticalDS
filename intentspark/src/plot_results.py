"""Combine classical + Qwen results into figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sns.set_theme(style="whitegrid", context="talk")


def main():
    frames = []
    for name in ("classical_comparison.csv", "qwen_probe_comparison.csv"):
        p = RESULTS / name
        if p.exists():
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RESULTS / "all_comparison.csv", index=False)

    colors = {"classical_tfidf": "#264653", "qwen_probe": "#e76f51"}

    fig, ax = plt.subplots(figsize=(9, 5))
    order = df.sort_values("macro_f1")
    ax.barh(order["model"], order["macro_f1"], color=[colors.get(f, "#999") for f in order["family"]])
    ax.set_xlabel("Macro-F1")
    ax.set_title("SST-5 fine-grained sentiment — Macro-F1")
    for y, v in enumerate(order["macro_f1"]):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig1_macro_f1.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    order = df.sort_values("accuracy")
    ax.barh(order["model"], order["accuracy"], color=[colors.get(f, "#999") for f in order["family"]])
    ax.set_xlabel("Accuracy")
    ax.set_title("SST-5 — Accuracy")
    for y, v in enumerate(order["accuracy"]):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig2_accuracy.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for fam, g in df.groupby("family"):
        ax.scatter(g["latency_ms"].clip(lower=1e-3), g["macro_f1"], s=160, label=fam, color=colors.get(fam, "#999"))
        for _, r in g.iterrows():
            ax.annotate(r["model"], (max(r["latency_ms"], 1e-3), r["macro_f1"]), fontsize=9, xytext=(5, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Latency ms (log)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Pareto: quality vs speed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig3_pareto.png", dpi=140)
    plt.close(fig)

    # class count / length EDA
    te = pd.read_csv(ROOT / "data" / "sst5_validation.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(te["text"].str.split().str.len(), bins=30, ax=ax, color="#264653")
    ax.set_title("Test utterance length (words)")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig0_length.png", dpi=140)
    plt.close(fig)

    print(df.sort_values("macro_f1", ascending=False).to_string(index=False))
    print("Wrote figures ->", RESULTS)


if __name__ == "__main__":
    main()
