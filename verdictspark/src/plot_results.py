"""
plot_results.py — classical vs Qwen generative vs Qwen probe (+ speculative).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")


def load_rows() -> pd.DataFrame:
    rows = []
    classical = RESULTS / "classical_comparison.csv"
    if classical.exists():
        c = pd.read_csv(classical)
        for _, r in c.iterrows():
            rows.append(
                {
                    "model": r["model"],
                    "family": "classical_tfidf",
                    "balanced_accuracy": r["balanced_accuracy"],
                    "macro_f1": r["macro_f1"],
                    "accuracy": r["accuracy"],
                    "latency_ms": float(r.get("latency_ms", 0.0) or 0.0),
                }
            )
    probe = RESULTS / "qwen_probe_comparison.csv"
    if probe.exists():
        p = pd.read_csv(probe)
        for _, r in p.iterrows():
            rows.append(
                {
                    "model": r["model"],
                    "family": "qwen_probe",
                    "balanced_accuracy": r["balanced_accuracy"],
                    "macro_f1": r["macro_f1"],
                    "accuracy": r["accuracy"],
                    "latency_ms": float(r["latency_ms"]),
                }
            )
    for mode in ("vanilla", "speculative", "assisted", "dspark"):
        path = RESULTS / f"llm_summary_{mode}.json"
        if path.exists():
            s = json.loads(path.read_text())
            rows.append(
                {
                    "model": f"qwen_gen_{mode}",
                    "family": "qwen_generative",
                    "balanced_accuracy": s["balanced_accuracy"],
                    "macro_f1": s["macro_f1"],
                    "accuracy": s["accuracy"],
                    "latency_ms": s["latency_ms_mean"],
                }
            )
    return pd.DataFrame(rows)


def main():
    df = load_rows()
    if df.empty:
        raise SystemExit("No results yet.")
    df.to_csv(RESULTS / "all_comparison.csv", index=False)

    color = {
        "classical_tfidf": "#264653",
        "qwen_probe": "#e76f51",
        "qwen_generative": "#2a9d8f",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    order = df.sort_values("balanced_accuracy")
    ax.barh(
        order["model"],
        order["balanced_accuracy"],
        color=[color.get(f, "#999") for f in order["family"]],
    )
    ax.axvline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel("Balanced accuracy")
    ax.set_title("SCOTUS winner prediction — who actually wins?")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig1_balanced_accuracy.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    order = df.sort_values("latency_ms")
    ax.barh(
        order["model"],
        order["latency_ms"].clip(lower=1e-3),
        color=[color.get(f, "#999") for f in order["family"]],
    )
    ax.set_xscale("log")
    ax.set_xlabel("Latency ms / case (log)")
    ax.set_title("Inference latency")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig2_latency.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for fam, g in df.groupby("family"):
        ax.scatter(
            g["latency_ms"].clip(lower=1e-3),
            g["balanced_accuracy"],
            s=140,
            label=fam,
            color=color.get(fam, "#999"),
        )
        for _, r in g.iterrows():
            ax.annotate(
                r["model"],
                (max(r["latency_ms"], 1e-3), r["balanced_accuracy"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.set_xscale("log")
    ax.set_xlabel("Latency ms (log)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Pareto: quality vs speed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig3_pareto.png", dpi=140)
    plt.close(fig)

    data = ROOT / "data" / "scotus_justice.csv"
    if data.exists():
        raw = pd.read_csv(data)
        fig, ax = plt.subplots(figsize=(7, 5))
        counts = raw["label_name"].value_counts()
        ax.bar(counts.index.astype(str), counts.values, color=["#2a9d8f", "#e9c46a"])
        ax.set_title("Class balance")
        fig.tight_layout()
        fig.savefig(RESULTS / "fig0_class_balance.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(raw["text"].str.len().clip(upper=3000), bins=40, ax=ax, color="#264653")
        ax.set_title("Fact length (characters)")
        fig.tight_layout()
        fig.savefig(RESULTS / "fig0_fact_length.png", dpi=140)
        plt.close(fig)

    print(df.sort_values("balanced_accuracy", ascending=False).to_string(index=False))
    print("Wrote figures ->", RESULTS)


if __name__ == "__main__":
    main()
