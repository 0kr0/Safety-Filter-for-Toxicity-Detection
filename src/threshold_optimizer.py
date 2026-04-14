"""
Threshold optimization for binary classifiers.
Finds the F1-optimal decision threshold on a validation set instead of using the default 0.5.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    n_thresholds: int = 200,
) -> dict:
    """
    Search over thresholds to maximize the given metric on the validation set.

    Returns dict with 'threshold', 'score', and per-threshold curves.
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    scores = []

    metric_fn = {
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
        "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
    }[metric]

    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        scores.append(metric_fn(y_true, y_pred))

    best_idx = int(np.argmax(scores))
    return {
        "threshold": float(thresholds[best_idx]),
        "score": float(scores[best_idx]),
        "metric": metric,
        "thresholds": thresholds.tolist(),
        "precisions": precisions,
        "recalls": recalls,
        "f1s": f1s,
    }


def apply_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """Apply a custom threshold to probability predictions."""
    return (y_proba >= threshold).astype(int)
