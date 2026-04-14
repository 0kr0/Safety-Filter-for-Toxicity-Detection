"""
Ensemble classifier combining DistilBERT and TF-IDF+LR for speed-accuracy trade-off.
Uses weighted averaging of predicted probabilities.
"""

import numpy as np
from .tfidf_lr import TFIDFLogisticRegressionClassifier
from .distilbert_clf import DistilBERTClassifier


class EnsembleClassifier:
    """
    Weighted ensemble of DistilBERT and TF-IDF+LR.
    At inference, combines predicted probabilities: p = w * p_bert + (1-w) * p_tfidf.
    Supports a cascade mode where TF-IDF+LR handles "easy" cases and DistilBERT
    is only called for uncertain ones, reducing average inference cost.
    """

    def __init__(
        self,
        bert_weight: float = 0.7,
        cascade: bool = False,
        cascade_threshold: float = 0.3,
        distilbert_kwargs: dict = None,
        tfidf_kwargs: dict = None,
    ):
        self.bert_weight = bert_weight
        self.cascade = cascade
        self.cascade_threshold = cascade_threshold
        self.tfidf = TFIDFLogisticRegressionClassifier(**(tfidf_kwargs or {}))
        self.bert = DistilBERTClassifier(**(distilbert_kwargs or {}))

    def fit(self, X: np.ndarray, y: np.ndarray, bert_epochs: int = 3):
        self.tfidf.fit(X, y)
        self.bert.fit(X, y, epochs=bert_epochs)
        return self

    def predict_proba(self, X: np.ndarray, normalize: bool = True) -> np.ndarray:
        X = np.atleast_1d(X)
        tfidf_proba = self.tfidf.predict_proba(X, normalize=normalize)

        if self.cascade:
            return self._cascade_predict_proba(X, tfidf_proba, normalize=normalize)

        bert_proba = self.bert.predict_proba(X, normalize=normalize)
        combined = self.bert_weight * bert_proba + (1 - self.bert_weight) * tfidf_proba
        return combined

    def _cascade_predict_proba(
        self, X: np.ndarray, tfidf_proba: np.ndarray, normalize: bool = True
    ) -> np.ndarray:
        """
        Cascade: only call DistilBERT for samples where TF-IDF+LR is uncertain
        (probability of toxic class between cascade_threshold and 1-cascade_threshold).
        """
        toxic_prob = tfidf_proba[:, 1]
        uncertain = (toxic_prob > self.cascade_threshold) & (toxic_prob < (1 - self.cascade_threshold))
        result = tfidf_proba.copy()

        if uncertain.any():
            uncertain_X = X[uncertain]
            bert_proba = self.bert.predict_proba(uncertain_X, normalize=normalize)
            result[uncertain] = (
                self.bert_weight * bert_proba + (1 - self.bert_weight) * tfidf_proba[uncertain]
            )
        return result

    def predict(self, X: np.ndarray, threshold: float = 0.5, normalize: bool = True) -> np.ndarray:
        proba = self.predict_proba(X, normalize=normalize)
        return (proba[:, 1] >= threshold).astype(int)

    def tune_weights(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = "auc",
        weights: np.ndarray = None,
    ) -> float:
        """
        Sweep bert_weight on the validation set to maximise the chosen metric.
        Returns the best weight and sets self.bert_weight to it.
        """
        from sklearn.metrics import roc_auc_score, f1_score

        X_val = np.atleast_1d(X_val)
        tfidf_proba = self.tfidf.predict_proba(X_val, normalize=True)
        bert_proba = self.bert.predict_proba(X_val, normalize=True)

        if weights is None:
            weights = np.arange(0.50, 0.96, 0.05)

        best_score = -1.0
        best_w = self.bert_weight

        for w in weights:
            combined = w * bert_proba + (1 - w) * tfidf_proba
            if metric == "auc":
                score = roc_auc_score(y_val, combined[:, 1])
            else:
                preds = (combined[:, 1] >= 0.5).astype(int)
                score = f1_score(y_val, preds)
            if score > best_score:
                best_score = score
                best_w = w

        self.bert_weight = best_w
        return best_w
