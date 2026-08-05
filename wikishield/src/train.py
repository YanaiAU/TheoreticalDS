"""
train.py — train & compare models for AI vs human Wikipedia text.
Usage: python src/train.py data/wiki_ai_detection.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import DenseTransformer, WikiFeaturizer  # noqa: E402

RESULTS = ROOT / "results"
PICKLES = ROOT / "pickles"
RESULTS.mkdir(exist_ok=True)
PICKLES.mkdir(exist_ok=True)
RANDOM_STATE = 42


def get_models():
    # Trees see a dense SVD projection so we can scale to 100k+ rows without
    # materializing the full TF-IDF matrix.
    tree_pipe = lambda clf: Pipeline(
        [
            ("scale", MaxAbsScaler()),
            ("svd", TruncatedSVD(n_components=120, random_state=RANDOM_STATE)),
            ("dense", DenseTransformer()),
            ("clf", clf),
        ]
    )
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "complement_nb": ComplementNB(),
        "linear_svm": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, dual="auto"),
            cv=3,
        ),
        "random_forest": tree_pipe(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=18,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "hist_gradient_boosting": tree_pipe(
            HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.08,
                max_iter=200,
                random_state=RANDOM_STATE,
            )
        ),
    }


def _proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1 / (1 + np.exp(-s))
    return model.predict(X).astype(float)


def main(path: str):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)
    y = df["label"].astype(int).values

    # Split by topic so human/AI twins of the same page never leak across folds
    topics = df["topic_id"].unique()
    tr_topics, te_topics = train_test_split(
        topics, test_size=0.2, random_state=RANDOM_STATE
    )
    tr_mask = df["topic_id"].isin(tr_topics)
    te_mask = df["topic_id"].isin(te_topics)
    df_tr, df_te = df.loc[tr_mask], df.loc[te_mask]
    y_tr, y_te = y[tr_mask.values], y[te_mask.values]

    print(f"Train: {len(df_tr):,}  Test: {len(df_te):,}  (topic-held-out split)")

    featurizer = WikiFeaturizer(max_features=8000)
    print("Fitting TF-IDF + style features...")
    X_tr = featurizer.fit_transform(df_tr)
    X_te = featurizer.transform(df_te)
    print(f"Feature matrix: {X_tr.shape}")

    rows = []
    best_name, best_f1, best_model = None, -1.0, None
    best_pred, best_proba = None, None

    for name, model in get_models().items():
        print(f"\n=== {name} ===")
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        proba = _proba(model, X_te)
        f1 = f1_score(y_te, pred, pos_label=1)
        row = {
            "model": name,
            "accuracy": float((pred == y_te).mean()),
            "precision_ai": float(precision_score(y_te, pred, pos_label=1, zero_division=0)),
            "recall_ai": float(recall_score(y_te, pred, pos_label=1, zero_division=0)),
            "f1_ai": float(f1),
            "roc_auc": float(roc_auc_score(y_te, proba)),
            "pr_auc": float(average_precision_score(y_te, proba)),
        }
        rows.append(row)
        print(classification_report(y_te, pred, target_names=["human", "AI"], digits=3))
        print(
            f"F1_AI={row['f1_ai']:.4f}  ROC-AUC={row['roc_auc']:.4f}  PR-AUC={row['pr_auc']:.4f}"
        )
        if f1 > best_f1:
            best_name, best_f1, best_model = name, f1, model
            best_pred, best_proba = pred, proba

    cmp = pd.DataFrame(rows).sort_values("f1_ai", ascending=False)
    cmp.to_csv(RESULTS / "model_comparison.csv", index=False)
    (RESULTS / "model_comparison.json").write_text(cmp.to_json(orient="records", indent=2))
    print("\nComparison:\n", cmp.to_string(index=False))
    print(f"\nBest: {best_name} (F1_AI={best_f1:.4f})")

    joblib.dump(featurizer, PICKLES / "featurizer.joblib")
    joblib.dump(best_model, PICKLES / "best_model.joblib")
    joblib.dump(
        {
            "y_true": y_te,
            "y_pred": best_pred,
            "y_proba": best_proba,
            "best_model": best_name,
        },
        PICKLES / "test_preds.joblib",
    )
    meta = {
        "best_model": best_name,
        "n_train": int(len(df_tr)),
        "n_test": int(len(df_te)),
        "n_features": int(X_tr.shape[1]),
        "split": "topic_held_out_20pct",
        "positive_class": "AI (label=1)",
    }
    (RESULTS / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved pickles -> {PICKLES}")
    print(f"Saved results -> {RESULTS}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "wiki_ai_detection.csv")
    main(path)
