"""
download_data.py — SCOTUS JUSTICE judgment dataset (real case facts → winner).

Source: https://huggingface.co/datasets/drgary/ft9_justice
~3300 Supreme Court cases with clerk-written facts and first_party_winner.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "scotus_justice.csv"
RANDOM_STATE = 42


def strip_html(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = BeautifulSoup(text, "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", t).strip()


def main():
    ds = load_dataset("drgary/ft9_justice", split="train")
    df = ds.to_pandas()
    df = df.dropna(subset=["facts", "first_party_winner"]).copy()
    df["facts_clean"] = df["facts"].map(strip_html)
    df = df[df["facts_clean"].str.len() >= 200].copy()
    df["label"] = df["first_party_winner"].astype(bool).astype(int)
    # 1 = first party (petitioner) wins; 0 = second party (respondent) wins
    df["label_name"] = df["label"].map({1: "petitioner_wins", 0: "respondent_wins"})
    df["word_count"] = df["facts_clean"].str.split().str.len().astype(int)
    # also keep char length — upstream "facts_len" is unreliable
    df["char_count"] = df["facts_clean"].str.len().astype(int)

    keep = [
        "ID",
        "name",
        "term",
        "first_party",
        "second_party",
        "facts_clean",
        "word_count",
        "issue_area",
        "label",
        "label_name",
        "decision_type",
        "disposition",
    ]
    out = df[keep].rename(columns={"facts_clean": "text"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out):,} cases -> {OUT}")
    print(out["label_name"].value_counts().to_string())
    print(
        f"words: mean={out['word_count'].mean():.0f}  "
        f"median={out['word_count'].median():.0f}  "
        f"p90={out['word_count'].quantile(0.9):.0f}"
    )


if __name__ == "__main__":
    main()
