"""Download real SST-5 fine-grained sentiment (SetFit/sst5)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"
LABELS = {
    0: "very_negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "very_positive",
}


def main():
    ds = load_dataset("SetFit/sst5")
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        df = ds[split].to_pandas()
        if "label_text" not in df.columns:
            df["label_text"] = df["label"].map(LABELS)
        path = OUT / f"sst5_{split}.csv"
        df.to_csv(path, index=False)
        print(f"{split}: {len(df):,} -> {path}")
    print(df["label_text"].value_counts().to_dict())


if __name__ == "__main__":
    main()
