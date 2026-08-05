"""
classical.py — TF-IDF baselines on long SCOTUS facts (expect mediocre scores).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "scotus_justice.csv"
RESULTS = ROOT / "results"
PICKLES = ROOT / "pickles"
RESULTS.mkdir(exist_ok=True)
PICKLES.mkdir(exist_ok=True)
RANDOM_STATE = 42


def metrics(y_true, y_pred, y_proba=None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "f1_petitioner": float(f1_score(y_true, y_pred, pos_label=1)),
        "f1_respondent": float(f1_score(y_true, y_pred, pos_label=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_proba is not None:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return out


def majority_baseline(y_train, y_test):
    maj = int(np.bincount(y_train).argmax())
    pred = np.full_like(y_test, maj)
    return metrics(y_test, pred)


def main():
    df = pd.read_csv(DATA)
    y = df["label"].astype(int).to_numpy()
    X = df["text"].astype(str).to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    # persist split indices via hash of texts for LLM alignment
    te_df = pd.DataFrame({"text": X_te, "label": y_te})
    te_df.to_csv(RESULTS / "test_split.csv", index=False)
    pd.DataFrame({"text": X_tr, "label": y_tr}).to_csv(RESULTS / "train_split.csv", index=False)

    rows = []
    maj = majority_baseline(y_tr, y_te)
    maj["model"] = "majority_baseline"
    maj["latency_ms"] = 0.0
    rows.append(maj)
    print("majority:", maj)

    models = {
        "complement_nb": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=40000,
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                ("clf", ComplementNB()),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=40000,
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=40000,
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            dual="auto",
                        ),
                        cv=3,
                    ),
                ),
            ]
        ),
    }

    best_name, best_bal, best_pipe = None, -1.0, None
    for name, pipe in models.items():
        print(f"\n=== {name} ===")
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        train_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        pred = pipe.predict(X_te)
        # per-sample latency on a small batch
        n_lat = min(64, len(X_te))
        t2 = time.perf_counter()
        _ = pipe.predict(X_te[:n_lat])
        lat_ms = (time.perf_counter() - t2) / n_lat * 1000

        proba = None
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_te)[:, 1]
        m = metrics(y_te, pred, proba)
        m["model"] = name
        m["train_seconds"] = float(train_s)
        m["latency_ms"] = float(lat_ms)
        m["predict_seconds"] = float(time.perf_counter() - t1)
        rows.append(m)
        print(classification_report(y_te, pred, target_names=["respondent", "petitioner"], digits=3))
        print(m)
        if m["balanced_accuracy"] > best_bal:
            best_name, best_bal, best_pipe = name, m["balanced_accuracy"], pipe

    cmp = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    cmp.to_csv(RESULTS / "classical_comparison.csv", index=False)
    (RESULTS / "classical_comparison.json").write_text(cmp.to_json(orient="records", indent=2))
    joblib.dump(best_pipe, PICKLES / "best_classical.joblib")
    meta = {
        "best_classical": best_name,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "majority_rate_train": float(y_tr.mean()),
        "selection": "balanced_accuracy",
    }
    (RESULTS / "classical_meta.json").write_text(json.dumps(meta, indent=2))
    print("\n", cmp.to_string(index=False))
    print(f"Best classical: {best_name}")


if __name__ == "__main__":
    main()
