"""
Evaluation metrics for toxicity classifiers.
Per proposal: Precision, Recall, F1, FPR, FNR, ROC-AUC, inference time.
"""

import time
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
    """
    Compute evaluation metrics.
    y_proba: predicted probabilities for positive class (for ROC-AUC). If None, ROC-AUC skipped.
    """
    metrics = {}
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = None

    return metrics


def measure_inference_time(model, X: np.ndarray, n_runs: int = 3) -> float:
    """Measure average inference time in seconds."""
    X = np.atleast_1d(X)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(X)
        times.append(time.perf_counter() - start)
    return np.mean(times)


def run_baseline(name: str, model, data: dict, has_proba: bool = False) -> dict:
    """Train, evaluate, and time a baseline model."""
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    print(f"\n--- {name} ---")
    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - train_start
    print(f"  Train time: {train_time:.2f}s")

    y_pred = model.predict(X_val)

    y_proba = None
    if has_proba and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)[:, 1]

    metrics = evaluate(y_val, y_pred, y_proba)
    infer_time = measure_inference_time(model, X_val[:1000])

    return {
        "name": name,
        "metrics": metrics,
        "train_time_s": train_time,
        "inference_time_s": infer_time,
    }


def print_results(results: list):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    for r in results:
        print(f"\n{r['name']}")
        print(f"  Train time:     {r['train_time_s']:.2f}s")
        print(f"  Inference time: {r['inference_time_s']:.4f}s (per 1000 samples)")
        m = r["metrics"]
        print(f"  Precision:      {m['precision']:.4f}")
        print(f"  Recall:         {m['recall']:.4f}")
        print(f"  F1:             {m['f1']:.4f}")
        print(f"  FPR:            {m['false_positive_rate']:.4f}")
        print(f"  FNR:            {m['false_negative_rate']:.4f}")
        if m.get("roc_auc") is not None:
            print(f"  ROC-AUC:        {m['roc_auc']:.4f}")
