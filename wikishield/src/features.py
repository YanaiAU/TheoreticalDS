"""
features.py — TF-IDF + stylistic cues for AI vs human Wikipedia text.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


_WORD = re.compile(r"[A-Za-z]+")


def stylistic_features(texts: list[str]) -> np.ndarray:
    """Cheap surface stats often used in AI-text detection literature."""
    rows = []
    for t in texts:
        t = t if isinstance(t, str) else ""
        n = max(len(t), 1)
        words = _WORD.findall(t)
        n_words = max(len(words), 1)
        uniq = len({w.lower() for w in words})
        chars = len(t)
        upper = sum(1 for c in t if c.isupper())
        digits = sum(1 for c in t if c.isdigit())
        punct = sum(1 for c in t if c in ".,;:!?\"'()-")
        commas = t.count(",")
        periods = t.count(".")
        avg_wlen = float(np.mean([len(w) for w in words])) if words else 0.0
        rows.append(
            [
                n_words,
                chars,
                uniq / n_words,  # type-token ratio
                avg_wlen,
                upper / n,
                digits / n,
                punct / n,
                commas / n_words,
                periods / n_words,
            ]
        )
    return np.asarray(rows, dtype=np.float64)


class DenseTransformer(BaseEstimator, TransformerMixin):
    """Sparse → dense for tree / boosting models."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if sparse.issparse(X) else np.asarray(X)


class WikiFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features: int = 8000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=3,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
        )

    def fit(self, df: pd.DataFrame, y=None):
        texts = df["text"].astype(str).tolist()
        self.tfidf.fit(texts)
        return self

    def transform(self, df: pd.DataFrame):
        texts = df["text"].astype(str).tolist()
        X_txt = self.tfidf.transform(texts)
        X_meta = sparse.csr_matrix(stylistic_features(texts))
        return sparse.hstack([X_txt, X_meta], format="csr")

    def fit_transform(self, df: pd.DataFrame, y=None):
        return self.fit(df, y).transform(df)
