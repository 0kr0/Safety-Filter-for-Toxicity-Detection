"""Baseline 2: TF-IDF + Logistic Regression - fast, interpretable, CPU-friendly."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TFIDFLogisticRegressionClassifier:
    """
    TF-IDF vectorizer + Logistic Regression for toxicity classification.
    """

    def __init__(self, max_features: int = 50000, C: float = 1.0, class_weight: str = "balanced"):
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
                ("lr", LogisticRegression(C=C, class_weight=class_weight, max_iter=500, random_state=42)),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.pipeline.fit(X.ravel(), y)
        return self

    @staticmethod
    def _normalize_input(X, normalize: bool):
        X = np.atleast_2d(X) if X.ndim == 1 else X
        if normalize:
            from ..text_normalize import normalize_text
            X = np.array([normalize_text(str(t)) for t in X.ravel()])
        else:
            X = X.ravel()
        return X

    def predict(self, X, normalize: bool = True) -> np.ndarray:
        X = self._normalize_input(np.atleast_1d(X), normalize)
        return self.pipeline.predict(X).astype(int)

    def predict_proba(self, X, normalize: bool = True) -> np.ndarray:
        X = self._normalize_input(np.atleast_1d(X), normalize)
        return self.pipeline.predict_proba(X)
