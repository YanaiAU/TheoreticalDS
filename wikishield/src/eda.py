"""
eda.py — exploratory plots + print preconceptions checklist.
Usage: python src/eda.py data/wiki_ai_detection.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")


def main(path: str):
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str)
    df["char_len"] = df["text"].str.len()
    if "word_count" not in df.columns:
        df["word_count"] = df["text"].str.split().str.len()

    print("=== shape ===", df.shape)
    print("=== label balance ===")
    print(df["label"].value_counts().rename({0: "human", 1: "AI"}))
    print("=== mean word_count by label ===")
    print(df.groupby("label")["word_count"].agg(["mean", "median", "std"]))

    # fig1 class balance
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["label"].value_counts().sort_index()
    ax.bar(["Human (0)", "AI (1)"], counts.values, color=["#2a9d8f", "#e76f51"])
    ax.set_ylabel("Count")
    ax.set_title("Class balance: Wikipedia vs GPT intros")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig1_class_balance.png", dpi=140)
    plt.close(fig)

    # fig2 length distributions
    fig, ax = plt.subplots(figsize=(8, 5))
    for lab, name, c in [(0, "Human", "#2a9d8f"), (1, "AI", "#e76f51")]:
        sns.kdeplot(
            df.loc[df["label"] == lab, "word_count"].clip(upper=400),
            ax=ax,
            label=name,
            color=c,
            fill=True,
            alpha=0.35,
        )
    ax.set_xlabel("Word count")
    ax.set_title("Intro length by authorship")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig2_length_distribution.png", dpi=140)
    plt.close(fig)

    # fig3 char length box
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_df = df.copy()
    plot_df["authorship"] = plot_df["label"].map({0: "Human", 1: "AI"})
    sns.boxplot(
        data=plot_df,
        x="authorship",
        y="char_len",
        ax=ax,
        showfliers=False,
        palette=["#2a9d8f", "#e76f51"],
    )
    ax.set_title("Character length (outliers hidden)")
    ax.set_ylabel("Characters")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig3_char_length.png", dpi=140)
    plt.close(fig)

    # fig4 type-token ratio proxy (unique / words on a sample)
    sample = df.sample(n=min(8000, len(df)), random_state=42)
    ttr = []
    for _, r in sample.iterrows():
        words = [w.lower() for w in str(r["text"]).split() if w.isalpha()]
        ttr.append(len(set(words)) / max(len(words), 1))
    sample = sample.assign(ttr=ttr, authorship=sample["label"].map({0: "Human", 1: "AI"}))
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(
        data=sample,
        x="authorship",
        y="ttr",
        ax=ax,
        palette=["#2a9d8f", "#e76f51"],
        cut=0,
    )
    ax.set_title("Type–token ratio (lexical diversity proxy)")
    ax.set_ylabel("Unique words / words")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig4_type_token_ratio.png", dpi=140)
    plt.close(fig)

    print("\nSaved figures ->", RESULTS)
    print(
        """
Preconceptions (write these into the notebook / README before peeking at metrics):
  1. GPT intros may be shorter / more uniform in length than real Wikipedia leads.
  2. Lexical diversity (TTR) and punctuation cadence may differ systematically.
  3. TF-IDF unigrams/bigrams should separate well on this paired topic design —
     but that can overestimate real-world robustness (same title, different author).
  4. Linear models (LogReg / LinearSVC) should be strong; trees help if style metas matter.
"""
    )


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "wiki_ai_detection.csv")
    main(path)
