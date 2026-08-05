"""Train staged models for serious vs light road-accident severity.

Act 0  majority_baseline
Act 1  naive_logreg
Act 2  logistic_regression
Act 3  linear_svm
Act 4  random_forest
Act 5  hist_gradient_boosting
Act 6  catboost  (final)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
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
MODEL_FILES = ROOT / "model_files"
RESULTS.mkdir(exist_ok=True)
MODEL_FILES.mkdir(exist_ok=True)

CAT_FEATURES = [
    "SUG_DEREH", "THUM_GEOGRAFI", "HODESH_TEUNA", "SHAA", "SUG_YOM", "YOM_LAYLA",
    "YOM_BASHAVUA", "SUG_TEUNA", "HAD_MASLUL", "RAV_MASLUL", "MEHIRUT_MUTERET",
    "TKINUT", "ROHAV", "SIMUN_TIMRUR", "TEURA", "MEZEG_AVIR", "PNE_KVISH",
    "MAHOZ", "NAFA", "EZOR_TIVI", "MAAMAD_MINIZIPALI", "ZURAT_ISHUV", "YEHIDA",
    "STATUS_IGUN", "SEMEL_YISHUV",
]
NUM_FEATURES = ["SHNAT_TEUNA", "X", "Y"]

NAIVE_CAT = ["SUG_DEREH", "SHAA", "YOM_LAYLA", "YOM_BASHAVUA", "MAHOZ"]
NAIVE_NUM = ["SHNAT_TEUNA"]

STORY_ORDER = [
    "majority_baseline",
    "naive_logreg",
    "logistic_regression",
    "linear_svm",
    "random_forest",
    "hist_gradient_boosting",
    "catboost",
]


def load_xy() -> tuple[pd.DataFrame, np.ndarray]:
    path = DATA / "accidents_2020_2024.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python src/download_data.py")
    df = pd.read_csv(path).dropna(subset=["HUMRAT_TEUNA"])
    y = (df["HUMRAT_TEUNA"].astype(int) <= 2).astype(int).to_numpy()
    cols = [c for c in CAT_FEATURES + NUM_FEATURES if c in df.columns]
    X = df[cols].copy()
    for c in CAT_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype("object").where(X[c].notna(), other="__MISSING__").astype(str)
    return X, y


def ohe_pre(cat: list[str], num: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=40), cat),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ]
    )


def ord_pre(cat: list[str], num: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
            ("num", SimpleImputer(strategy="median"), num),
        ]
    )


def save_model(name: str, clf) -> Path:
    """Write one artifact per model under model_files/."""
    if isinstance(clf, CatBoostClassifier):
        path = MODEL_FILES / f"{name}.cbm"
        clf.save_model(str(path))
    else:
        path = MODEL_FILES / f"{name}.joblib"
        joblib.dump(clf, path, compress=3)
    print(f"  saved {path.name} ({path.stat().st_size} bytes)")
    return path


def evaluate(name: str, family: str, act: str, clf, X_tr, y_tr, X_te, y_te) -> tuple[dict, object]:
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    train_s = time.perf_counter() - t0
    pred = clf.predict(X_te)
    if hasattr(pred, "ravel"):
        pred = np.asarray(pred).ravel().astype(int)

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
        "act": act,
        "accuracy": float(accuracy_score(y_te, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
        "macro_f1": float(f1_score(y_te, pred, average="macro")),
        "f1_serious": float(f1_score(y_te, pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_te, proba)) if proba is not None else float("nan"),
        "train_seconds": float(train_s),
    }
    print(f"\n=== {act}: {name} ===")
    print(classification_report(y_te, pred, target_names=["light", "serious"], digits=3, zero_division=0))
    print(row)
    return row, clf


def main() -> None:
    X, y = load_xy()
    cat = [c for c in CAT_FEATURES if c in X.columns]
    num = [c for c in NUM_FEATURES if c in X.columns]
    naive_cat = [c for c in NAIVE_CAT if c in X.columns]
    naive_num = [c for c in NAIVE_NUM if c in X.columns]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    specs: list[tuple[str, str, str, object, pd.DataFrame, pd.DataFrame]] = [
        (
            "majority_baseline",
            "baseline",
            "0_majority_trap",
            DummyClassifier(strategy="most_frequent"),
            X_tr,
            X_te,
        ),
        (
            "naive_logreg",
            "classical",
            "1_first_guess",
            Pipeline(
                [
                    ("pre", ohe_pre(naive_cat, naive_num)),
                    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1, random_state=42)),
                ]
            ),
            X_tr[naive_cat + naive_num],
            X_te[naive_cat + naive_num],
        ),
        (
            "logistic_regression",
            "classical",
            "2_full_linear",
            Pipeline(
                [
                    ("pre", ohe_pre(cat, num)),
                    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1, random_state=42)),
                ]
            ),
            X_tr,
            X_te,
        ),
        (
            "linear_svm",
            "classical",
            "3_full_linear",
            Pipeline(
                [
                    ("pre", ohe_pre(cat, num)),
                    ("clf", LinearSVC(class_weight="balanced", random_state=42, dual="auto", max_iter=4000)),
                ]
            ),
            X_tr,
            X_te,
        ),
        (
            "random_forest",
            "tree",
            "4_accuracy_trap",
            Pipeline(
                [
                    ("pre", ord_pre(cat, num)),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=200,
                            max_depth=20,
                            min_samples_leaf=2,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            X_tr,
            X_te,
        ),
        (
            "hist_gradient_boosting",
            "tree",
            "5_boosted_trees",
            Pipeline(
                [
                    ("pre", ord_pre(cat, num)),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_depth=12,
                            learning_rate=0.06,
                            max_iter=350,
                            l2_regularization=0.05,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            X_tr,
            X_te,
        ),
        (
            "catboost",
            "champion",
            "6_final_model",
            CatBoostClassifier(
                iterations=600,
                depth=8,
                learning_rate=0.07,
                l2_leaf_reg=3.0,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=42,
                verbose=False,
                auto_class_weights="Balanced",
                cat_features=cat,
                thread_count=8,
            ),
            X_tr[cat + num],
            X_te[cat + num],
        ),
    ]

    rows = []
    best_name, best_f1, best_clf = None, -1.0, None
    artifact_map = {}

    for name, family, act, clf, xtr, xte in specs:
        row, fitted = evaluate(name, family, act, clf, xtr, y_tr, xte, y_te)
        rows.append(row)
        path = save_model(name, fitted)
        artifact_map[name] = path.name
        if name != "majority_baseline" and row["macro_f1"] > best_f1:
            best_name, best_f1, best_clf = name, row["macro_f1"], fitted

    order = {m: i for i, m in enumerate(STORY_ORDER)}
    cmp = pd.DataFrame(rows)
    cmp["_ord"] = cmp["model"].map(order)
    cmp = cmp.sort_values("_ord").drop(columns="_ord")
    cmp.to_csv(RESULTS / "model_comparison.csv", index=False)

    print("\n=== ranking (macro-F1, excl. majority) ===")
    print(
        cmp[cmp["model"] != "majority_baseline"]
        .sort_values("macro_f1", ascending=False)
        .to_string(index=False)
    )

    meta = {
        "task": "binary_serious_vs_light",
        "story": STORY_ORDER,
        "target": "HUMRAT_TEUNA<=2 → serious; else light",
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "pos_rate": float(y.mean()),
        "best_model": best_name,
        "best_macro_f1": float(best_f1),
        "model_artifacts": artifact_map,
        "naive_features": naive_cat + naive_num,
        "features_cat": cat,
        "features_num": num,
        "source": "https://data.gov.il/he/datasets/lamas/2023-puf",
        "cleaning": [
            "Drop rows missing HUMRAT_TEUNA",
            "Map severity 1/2 → serious, 3 → light",
            "Categorical NA → __MISSING__",
            "Numeric NA → median impute",
            "Stratified 80/20 holdout (seed 42)",
        ],
    }
    (RESULTS / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # Convenience copy of the best model under a stable name
    if best_name:
        best_src = MODEL_FILES / artifact_map[best_name]
        if best_src.suffix == ".cbm":
            best_clf.save_model(str(MODEL_FILES / "best_model.cbm"))
        else:
            joblib.dump(best_clf, MODEL_FILES / "best_model.joblib", compress=3)
    print(f"\nBest: {best_name} macro_f1={best_f1:.4f}")
    print("Artifacts:", ", ".join(artifact_map.values()))


if __name__ == "__main__":
    main()
