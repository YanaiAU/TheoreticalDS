"""TF-IDF baselines on SST-5 (5-class sentiment) — expect weak scores."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PICKLES = ROOT / "pickles"
RESULTS.mkdir(exist_ok=True)
PICKLES.mkdir(exist_ok=True)


def main():
    tr = pd.read_csv(DATA / "sst5_train.csv")
    te = pd.read_csv(DATA / "sst5_validation.csv")  # official val for fair compare
    X_tr, y_tr = tr["text"].astype(str).tolist(), tr["label"].astype(int).to_numpy()
    X_te, y_te = te["text"].astype(str).tolist(), te["label"].astype(int).to_numpy()

    models = {
        "complement_nb": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("clf", ComplementNB()),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1, random_state=42)),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("clf", LinearSVC(class_weight="balanced", random_state=42, dual="auto")),
            ]
        ),
    }

    rows = []
    best_name, best_f1, best_pipe = None, -1.0, None
    for name, pipe in models.items():
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        train_s = time.perf_counter() - t0
        n = min(128, len(X_te))
        t1 = time.perf_counter()
        _ = pipe.predict(X_te[:n])
        lat = (time.perf_counter() - t1) / n * 1000
        pred = pipe.predict(X_te)
        row = {
            "model": name,
            "family": "classical_tfidf",
            "accuracy": float(accuracy_score(y_te, pred)),
            "macro_f1": float(f1_score(y_te, pred, average="macro")),
            "weighted_f1": float(f1_score(y_te, pred, average="weighted")),
            "latency_ms": float(lat),
            "train_seconds": float(train_s),
        }
        rows.append(row)
        print(f"\n=== {name} ===")
        print(classification_report(y_te, pred, digits=3, zero_division=0))
        print(row)
        if row["macro_f1"] > best_f1:
            best_name, best_f1, best_pipe = name, row["macro_f1"], pipe

    cmp = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    cmp.to_csv(RESULTS / "classical_comparison.csv", index=False)
    joblib.dump(best_pipe, PICKLES / "best_classical.joblib")
    (RESULTS / "classical_meta.json").write_text(
        json.dumps(
            {
                "best": best_name,
                "n_train": len(tr),
                "n_test": len(te),
                "n_classes": 5,
                "task": "sst5_fine_grained_sentiment",
                "split": "validation",
            },
            indent=2,
        )
    )
    print(cmp.to_string(index=False))
    print("Best classical:", best_name, best_f1)


if __name__ == "__main__":
    main()
