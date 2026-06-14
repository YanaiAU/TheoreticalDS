"""
Movie rating prediction — model comparison project.
Run with: py -3 run_movie_project.py
Generates figures/ and prints comparison tables.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

DATA_DIR = Path(__file__).parent
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
HIGH_RATING_THRESHOLD = 7.0
TOP_GENRES = 12
TOP_LANGUAGES = 8


def load_movies() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "mymoviedb.csv", engine="python", on_bad_lines="warn")
    df["Release_Date"] = pd.to_datetime(df["Release_Date"], errors="coerce")
    df["Release_Year"] = df["Release_Date"].dt.year
    for col in ("Popularity", "Vote_Count", "Vote_Average"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["log_popularity"] = np.log1p(df["Popularity"].clip(lower=0))
    df["log_vote_count"] = np.log1p(df["Vote_Count"].clip(lower=0))
    return df


def top_values(series: pd.Series, n: int) -> list[str]:
    return series.value_counts().head(n).index.tolist()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    genre_lists = (
        out["Genre"].fillna("Unknown").str.split(", ").apply(lambda xs: [g.strip() for g in xs])
    )
    top_g = top_values(
        genre_lists.explode().dropna(),
        TOP_GENRES,
    )
    for genre in top_g:
        out[f"genre_{genre}"] = genre_lists.apply(lambda xs, g=genre: int(g in xs))

    top_lang = top_values(out["Original_Language"].fillna("Unknown"), TOP_LANGUAGES)
    for lang in top_lang:
        out[f"lang_{lang}"] = (out["Original_Language"].fillna("Unknown") == lang).astype(int)

    out["High_Rated"] = (out["Vote_Average"] >= HIGH_RATING_THRESHOLD).astype(int)
    return out


FEATURE_COLS = (
    ["log_popularity", "log_vote_count", "Release_Year"]
    + [f"genre_{g}" for g in []]  # filled after build_features
)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("log_", "genre_", "lang_")) or c == "Release_Year"]


def classification_models() -> dict:
    models = {
        "1. Logistic Regression (L2)": Pipeline(
            [
                ("prep", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "2. SVM (RBF kernel)": Pipeline(
            [
                ("prep", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "3. Random Forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=12,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    if HAS_CATBOOST:
        models["4. CatBoost"] = Pipeline(
            [
                (
                    "model",
                    CatBoostClassifier(
                        iterations=400,
                        depth=6,
                        learning_rate=0.05,
                        verbose=0,
                        random_state=RANDOM_STATE,
                        auto_class_weights="Balanced",
                    ),
                ),
            ]
        )
    else:
        models["4. HistGradient Boosting"] = Pipeline(
            [
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=6,
                        learning_rate=0.05,
                        max_iter=400,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    models["5. MLP (SGD/Adam)"] = Pipeline(
        [
            ("prep", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    alpha=0.001,
                    max_iter=500,
                    early_stopping=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return models


def plot_eda(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    df["Vote_Average"].dropna().hist(bins=30, ax=axes[0, 0], color="#6A5ACD", edgecolor="white")
    axes[0, 0].axvline(HIGH_RATING_THRESHOLD, color="crimson", ls="--", label=f"threshold = {HIGH_RATING_THRESHOLD}")
    axes[0, 0].set_title("Vote average distribution")
    axes[0, 0].legend()

    sample = df.dropna(subset=["Popularity", "Vote_Average"]).sample(min(2500, len(df)), random_state=RANDOM_STATE)
    axes[0, 1].scatter(sample["log_popularity"], sample["Vote_Average"], alpha=0.25, s=12, c="#FF8C00")
    axes[0, 1].set_xlabel("log(1 + popularity)")
    axes[0, 1].set_ylabel("Vote average")
    axes[0, 1].set_title("Popularity vs rating")

    year_means = df.groupby("Release_Year")["Vote_Average"].mean().dropna()
    axes[1, 0].plot(year_means.index, year_means.values, color="#2E8B57")
    axes[1, 0].set_title("Mean rating by release year")
    axes[1, 0].set_xlabel("Year")

    genres = df["Genre"].dropna().str.split(", ").explode().str.strip()
    genres.value_counts().head(10).plot(kind="barh", ax=axes[1, 1], color="#CD5C5C")
    axes[1, 1].set_title("Top 10 genres")
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_eda_overview.png", dpi=140)
    plt.close(fig)


def plot_pca(df: pd.DataFrame, feature_cols: list[str]) -> None:
    matrix = df[feature_cols].dropna()
    labels = df.loc[matrix.index, "Vote_Average"]
    scaled = StandardScaler().fit_transform(matrix)
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="plasma", alpha=0.45, s=14)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA on engineered features (colored by vote average)")
    plt.colorbar(sc, ax=ax, label="Vote average")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_pca_features.png", dpi=140)
    plt.close(fig)


def compare_classifiers(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
    rows = []
    fitted = {}

    for name, pipe in classification_models().items():
        cv = cross_validate(pipe, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        rows.append(
            {
                "model": name,
                "cv_accuracy": cv["test_accuracy"].mean(),
                "cv_f1": cv["test_f1"].mean(),
                "cv_roc_auc": cv["test_roc_auc"].mean(),
                "test_accuracy": accuracy_score(y_test, pred),
                "test_f1": f1_score(y_test, pred),
                "test_roc_auc": roc_auc_score(y_test, proba),
            }
        )
        fitted[name] = pipe

    results = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)
    results.to_csv(DATA_DIR / "classification_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results))
    width = 0.25
    ax.bar(x - width, results["cv_accuracy"], width, label="CV accuracy")
    ax.bar(x, results["cv_f1"], width, label="CV F1")
    ax.bar(x + width, results["cv_roc_auc"], width, label="CV ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(results["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Model comparison (5-fold CV on training set)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_model_comparison.png", dpi=140)
    plt.close(fig)

    best_name = results.iloc[0]["model"]
    RocCurveDisplay.from_predictions(y_test, fitted[best_name].predict_proba(X_test)[:, 1], ax=plt.subplots()[1])
    plt.title(f"ROC curve — best model: {best_name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_roc_best_model.png", dpi=140)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(y_test, fitted[best_name].predict(X_test))
    plt.title(f"Confusion matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_confusion_best_model.png", dpi=140)
    plt.close()

    return results


def regularization_sweep(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    c_values = [0.01, 0.1, 1, 10, 100]
    rows = []
    for c in c_values:
        pipe = Pipeline(
            [
                ("prep", StandardScaler()),
                ("model", LogisticRegression(C=c, max_iter=2000, random_state=RANDOM_STATE)),
            ]
        )
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        rows.append({"C": c, "test_roc_auc": roc_auc_score(y_test, proba)})
    sweep = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(sweep["C"], sweep["test_roc_auc"], marker="o")
    ax.set_xlabel("C (inverse regularization strength)")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Tikhonov/L2 regularization effect — Logistic Regression")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_regularization_sweep.png", dpi=140)
    plt.close(fig)
    return sweep


def regression_baseline(X_train, X_test, y_train, y_test) -> dict:
    pipe = Pipeline(
        [
            ("prep", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, pred),
        "r2": r2_score(y_test, pred),
    }


def main() -> None:
    raw = load_movies()
    df = build_features(raw)
    feature_cols = get_feature_columns(df)
    model_df = df.dropna(subset=feature_cols + ["Vote_Average", "High_Rated"])

    print(f"Rows used for modeling: {len(model_df):,}")
    print(f"High-rated prevalence (>= {HIGH_RATING_THRESHOLD}): {model_df['High_Rated'].mean():.1%}")
    print(f"CatBoost available: {HAS_CATBOOST}")

    plot_eda(model_df)
    plot_pca(model_df, feature_cols)

    X = model_df[feature_cols]
    y_cls = model_df["High_Rated"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cls
    )

    print("\n=== Classification: predict High_Rated ===")
    cls_results = compare_classifiers(X_train, X_test, y_train, y_test)
    print(cls_results.to_string(index=False))

    print("\n=== Regularization sweep (course: stability / Tikhonov) ===")
    print(regularization_sweep(X_train, y_train, X_test, y_test).to_string(index=False))

    y_reg = model_df["Vote_Average"]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=RANDOM_STATE
    )
    reg_metrics = regression_baseline(X_train_r, X_test_r, y_train_r, y_test_r)
    print("\n=== Regression baseline (Ridge on Vote_Average) ===")
    print(reg_metrics)

    print(f"\nFigures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
