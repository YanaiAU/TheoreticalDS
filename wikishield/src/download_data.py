"""
download_data.py — fetch real Wikipedia + GPT intros from Hugging Face.

Source: https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro
150k Wikipedia topics, each with a human intro and a GPT-Curie generated intro.
We expand pairs into a binary classification table: human (0) vs AI (1).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "wiki_ai_detection.csv"
RANDOM_STATE = 42


def build_frame(max_topics: int | None) -> pd.DataFrame:
    ds = load_dataset("aadityaubhat/GPT-wiki-intro", split="train")
    n = len(ds) if max_topics is None else min(max_topics, len(ds))
    if max_topics is not None and max_topics < len(ds):
        idx = (
            pd.Series(range(len(ds)))
            .sample(n=n, random_state=RANDOM_STATE)
            .sort_values()
            .tolist()
        )
        ds = ds.select(idx)

    print("Converting to pandas...")
    raw = ds.to_pandas()
    human = pd.DataFrame(
        {
            "topic_id": raw["id"],
            "title": raw["title"],
            "url": raw["url"],
            "text": raw["wiki_intro"],
            "label": 0,
            "source": "wikipedia",
            "word_count": raw["wiki_intro_len"].astype(int),
        }
    )
    ai = pd.DataFrame(
        {
            "topic_id": raw["id"],
            "title": raw["title"],
            "url": raw["url"],
            "text": raw["generated_intro"],
            "label": 1,
            "source": "gpt_curie",
            "word_count": raw["generated_intro_len"].astype(int),
        }
    )
    return pd.concat([human, ai], ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--max-topics",
        type=int,
        default=75000,
        help="Number of Wikipedia topics to keep (each yields 2 rows). "
        "Use 150000 for the full corpus (~300k texts).",
    )
    p.add_argument("--out", type=str, default=str(OUT))
    args = p.parse_args()

    max_topics = None if args.max_topics <= 0 else args.max_topics
    df = build_frame(max_topics)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    n_h = int((df["label"] == 0).sum())
    n_a = int((df["label"] == 1).sum())
    print(f"Wrote {len(df):,} rows -> {out}")
    print(f"  human (0): {n_h:,}  |  AI (1): {n_a:,}")
    print(f"  topics: {df['topic_id'].nunique():,}")


if __name__ == "__main__":
    main()
