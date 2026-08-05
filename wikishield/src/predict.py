"""
predict.py — score a single text blob as human vs AI.
Usage:
  python src/predict.py --text "Some Wikipedia-style paragraph..."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import DenseTransformer, WikiFeaturizer  # noqa: E402  # register for unpickle

PICKLES = ROOT / "pickles"


def main():
    p = argparse.ArgumentParser(description="Score text as human (0) or AI (1)")
    p.add_argument("--text", required=True, help="Paragraph to classify")
    p.add_argument("--featurizer", default=str(PICKLES / "featurizer.joblib"))
    p.add_argument("--model", default=str(PICKLES / "best_model.joblib"))
    args = p.parse_args()

    featurizer = joblib.load(args.featurizer)
    model = joblib.load(args.model)
    df = pd.DataFrame([{"text": args.text}])
    X = featurizer.transform(df)
    pred = int(model.predict(X)[0])
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0, 1])
    else:
        proba = float(pred)

    label = "AI-generated" if pred == 1 else "Human-written"
    print(f"Prediction : {label} (label={pred})")
    print(f"P(AI)      : {proba:.4f}")


if __name__ == "__main__":
    main()
