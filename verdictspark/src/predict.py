"""
predict.py — score one case facts blob with best classical model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
PICKLES = ROOT / "pickles"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--model", default=str(PICKLES / "best_classical.joblib"))
    args = p.parse_args()
    clf = joblib.load(args.model)
    pred = int(clf.predict([args.text])[0])
    label = "PETITIONER" if pred == 1 else "RESPONDENT"
    print(f"Prediction: {label} (label={pred})")
    if hasattr(clf, "predict_proba"):
        print(f"P(petitioner): {clf.predict_proba([args.text])[0, 1]:.4f}")


if __name__ == "__main__":
    main()
