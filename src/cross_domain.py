"""
Cross-domain evaluation: train on one dataset, evaluate on another.
Tests generalization from Toxic Conversations to Jigsaw/Civil Comments and vice versa.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .config import RANDOM_SEED, VAL_RATIO
from .data import clean_text


def load_civil_comments(n_samples: int = 10000) -> dict:
    """Load Civil Comments dataset from HuggingFace for cross-domain testing."""
    from datasets import load_dataset as hf_load

    ds = hf_load("google/civil_comments", split="train")
    df = ds.to_pandas()

    df = df.sample(n=min(n_samples, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)
    df["text"] = df["text"].apply(clean_text)
    df["label"] = (df["toxicity"] >= 0.5).astype(int)

    X = np.array(df["text"].tolist())
    y = np.array(df["label"].tolist(), dtype=int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_RATIO, random_state=RANDOM_SEED, stratify=y
    )

    return {
        "train_df": df,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": None,
        "y_test": None,
        "test_df": None,
        "name": "Civil Comments",
    }


def cross_domain_evaluate(
    model,
    train_data: dict,
    eval_data: dict,
    has_proba: bool = False,
) -> dict:
    """
    Train model on train_data, evaluate on eval_data's validation set.
    Returns metrics dict showing cross-domain generalization.
    """
    model.fit(train_data["X_train"], train_data["y_train"])

    X_eval, y_eval = eval_data["X_val"], eval_data["y_val"]
    y_pred = model.predict(X_eval)

    y_proba = None
    if has_proba and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_eval)[:, 1]

    metrics = {
        "precision": precision_score(y_eval, y_pred, zero_division=0),
        "recall": recall_score(y_eval, y_pred, zero_division=0),
        "f1": f1_score(y_eval, y_pred, zero_division=0),
    }

    if y_proba is not None and len(np.unique(y_eval)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_eval, y_proba)

    return metrics
