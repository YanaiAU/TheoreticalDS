"""Predict serious vs light accident severity from scene features.

Israel CBS PUF 2020–2024 (~50k accidents). Classical linear models vs tree ensembles.
Target: serious (fatal + severe) vs light.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PICKLES = ROOT / "pickles"
RESULTS.mkdir(exist_ok=True)
PICKLES.mkdir(exist_ok=True)

CAT_FEATURES = [
    "SUG_DEREH", "THUM_GEOGRAFI", "HODESH_TEUNA", "SHAA", "SUG_YOM", "YOM_LAYLA",
    "YOM_BASHAVUA", "SUG_TEUNA", "HAD_MASLUL", "RAV_MASLUL", "MEHIRUT_MUTERET",
    "TKINUT", "ROHAV", "SIMUN_TIMRUR", "TEURA", "MEZEG_AVIR", "PNE_KVISH",
    "MAHOZ", "NAFA", "EZOR_TIVI", "MAAMAD_MINIZIPALI", "ZURAT_ISHUV", "YEHIDA",
    "STATUS_IGUN", "SEMEL_YISHUV",
]
NUM_FEATURES = ["SHNAT_TEUNA", "X", "Y"]


def load_xy():
    path = DATA / "accidents_2020_2024.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python src/download_data.py")
    df = pd.read_csv(path).dropna(subset=["HUMRAT_TEUNA"])
    # 1=fatal, 2=severe, 3=light → serious if <=2
    y = (df["HUMRAT_TEUNA"].astype(int) <= 2).astype(int).to_numpy()
    cols = [c for c in CAT_FEATURES + NUM_FEATURES if c in df.columns]
    X = df[cols].copy()
    for c in CAT_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype("object").where(X[c].notna(), other="__MISSING__").astype(str)
    return X, y


def ohe_pre(X):
    cat = [c for c in CAT_FEATURES if c in X.columns]
    num = [c for c in NUM_FEATURES if c in X.columns]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=40), cat),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ]
    )


def ord_pre(X):
    cat = [c for c in CAT_FEATURES if c in X.columns]
    num = [c for c in NUM_FEATURES if c in X.columns]
    return ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
            ("num", SimpleImputer(strategy="median"), num),
        ]
    )


def score_row(name, family, clf, X_tr, y_tr, X_te, y_te):
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    train_s = time.perf_counter() - t0
    pred = clf.predict(X_te)
    proba = None
    if hasattr(clf, "predict_proba"):
        try:
            proba = clf.predict_proba(X_te)[:, 1]
        except Exception:
            proba = None
    if proba is None and hasattr(clf, "decision_function"):
        proba = clf.decision_function(X_te)
    row = {
        "model": name,
        "family": family,
        "accuracy": float(accuracy_score(y_te, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
        "macro_f1": float(f1_score(y_te, pred, average="macro")),
        "f1_serious": float(f1_score(y_te, pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_te, proba)) if proba is not None else None,
        "train_seconds": float(train_s),
    }
    print(f"\n=== {name} ===")
    print(classification_report(y_te, pred, target_names=["light", "serious"], digits=3, zero_division=0))
    print(row)
    return row, clf


def main():
    X, y = load_xy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "logistic_regression": (
            "classical",
            Pipeline([("pre", ohe_pre(X)), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1, random_state=42))]),
        ),
        "linear_svm": (
            "classical",
            Pipeline([("pre", ohe_pre(X)), ("clf", LinearSVC(class_weight="balanced", random_state=42, dual="auto", max_iter=4000))]),
        ),
        "random_forest": (
            "tree",
            Pipeline([("pre", ord_pre(X)), ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=1, class_weight="balanced_subsample", n_jobs=-1, random_state=42))]),
        ),
        "hist_gradient_boosting": (
            "tree",
            Pipeline([("pre", ord_pre(X)), ("clf", HistGradientBoostingClassifier(max_depth=12, learning_rate=0.06, max_iter=350, l2_regularization=0.05, class_weight="balanced", random_state=42))]),
        ),
    }

    rows, best_name, best_f1, best_clf = [], None, -1.0, None
    for name, (family, clf) in models.items():
        row, fitted = score_row(name, family, clf, X_tr, y_tr, X_te, y_te)
        rows.append(row)
        if row["macro_f1"] > best_f1:
            best_name, best_f1, best_clf = name, row["macro_f1"], fitted

    cmp = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    cmp.to_csv(RESULTS / "model_comparison.csv", index=False)
    print("\n=== ranking ===")
    print(cmp.to_string(index=False))

    meta = {
        "task": "binary_serious_vs_light",
        "target": "HUMRAT_TEUNA<=2 → serious; else light",
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "pos_rate": float(y.mean()),
        "best_model": best_name,
        "best_macro_f1": float(best_f1),
        "features_cat": [c for c in CAT_FEATURES if c in X.columns],
        "features_num": [c for c in NUM_FEATURES if c in X.columns],
        "source": "https://data.gov.il/dataset/02789da8-7a3e-4bfc-b771-1732b1cf403c",
        "cleaning": [
            "Drop rows missing HUMRAT_TEUNA",
            "Map severity 1/2 → serious, 3 → light",
            "Categorical NA → __MISSING__",
            "Numeric NA → median impute",
            "Stratified 80/20 holdout",
        ],
    }
    (RESULTS / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    joblib.dump(best_clf, PICKLES / "best_model.joblib")
    print(f"Best: {best_name} macro_f1={best_f1:.4f}")


if __name__ == "__main__":
    main()
