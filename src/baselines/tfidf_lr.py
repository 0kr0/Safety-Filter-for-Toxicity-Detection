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

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(X) if X.ndim == 1 else X
        return self.pipeline.predict(X.ravel()).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        X = np.atleast_2d(X) if X.ndim == 1 else X
        return self.pipeline.predict_proba(X.ravel())
