"""
Data loading and preprocessing for toxicity detection.
Expects Jigsaw Toxic Comment dataset: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, TRAIN_CSV, TEST_CSV, VAL_RATIO, RANDOM_SEED, TOXICITY_LABELS


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercasing, handle special chars."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = text.strip()
    return text


def create_binary_label(df: pd.DataFrame) -> np.ndarray:
    """
    Create binary toxicity label: toxic if ANY of the toxicity categories is 1.
    """
    if all(label in df.columns for label in TOXICITY_LABELS):
        return (df[TOXICITY_LABELS].max(axis=1) >= 1).astype(int).values
    if "toxic" in df.columns:
        return (df["toxic"] >= 1).astype(int).values
    raise ValueError("Dataset must have 'toxic' or toxicity label columns")


def load_demo_data(n_samples: int = 2000) -> dict:
    """
    Generate small synthetic dataset for testing when Jigsaw is not available.
    """
    np.random.seed(RANDOM_SEED)
    toxic_samples = [
        "you are such an idiot",
        "shut up you stupid moron",
        "go die somewhere",
        "this is garbage",
        "you ugly loser",
        "what trash",
        "dumb comment",
        "I hate you",
        "you suck",
        "such a jerk",
    ]
    neutral_samples = [
        "thanks for sharing",
        "I agree with your point",
        "that is interesting",
        "could you explain more?",
        "let me think about it",
        "good idea",
        "I see what you mean",
        "have a nice day",
        "looking forward to it",
        "appreciate the update",
    ]

    texts, labels = [], []
    for _ in range(n_samples // 2):
        t = np.random.choice(toxic_samples)
        texts.append(t + " " * np.random.randint(0, 3) + np.random.choice(["!", "?", ""]))
        labels.append(1)
    for _ in range(n_samples // 2):
        t = np.random.choice(neutral_samples)
        texts.append(t)
        labels.append(0)

    shuffle = np.random.permutation(n_samples)
    X = np.array([clean_text(texts[i]) for i in shuffle])
    y = np.array([labels[i] for i in shuffle])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_RATIO, random_state=RANDOM_SEED, stratify=y
    )

    return {
        "train_df": None,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": None,
        "y_test": None,
        "test_df": None,
    }


def load_jigsaw(data_dir: Path = None) -> dict:
    """
    Load Jigsaw Toxic Comment dataset.
    Returns: (train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test)
    """
    data_dir = data_dir or DATA_DIR
    train_path = data_dir / TRAIN_CSV
    test_path = data_dir / TEST_CSV

    if not train_path.exists():
        raise FileNotFoundError(
            f"Train file not found at {train_path}. "
            "Download from Kaggle: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge"
        )

    train_df = pd.read_csv(train_path)
    train_df["comment_text"] = train_df["comment_text"].fillna("").apply(clean_text)

    # Binary target: toxic if any category is 1
    y = create_binary_label(train_df)
    X = train_df["comment_text"].values

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_RATIO, random_state=RANDOM_SEED, stratify=y
    )

    test_df = None
    X_test, y_test = None, None

    if test_path.exists():
        test_df = pd.read_csv(test_path)
        test_df["comment_text"] = test_df["comment_text"].fillna("").apply(clean_text)
        X_test = test_df["comment_text"].values
        # Jigsaw test has no labels - use validation as test proxy for evaluation
        # In real use, you'd evaluate on labeled test or hold-out set
        y_test = None

    return {
        "train_df": train_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "test_df": test_df,
    }


def load_hf_toxic(n_samples: int = 10000) -> dict:
    from datasets import load_dataset as hf_load

    ds = hf_load("SetFit/toxic_conversations", split="train")
    df = ds.to_pandas()
    df = df.sample(n=min(n_samples, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)

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
    }


def get_eval_split(data: dict) -> tuple:
    """Use validation set for evaluation (test has no labels in Jigsaw)."""
    return data["X_val"], data["y_val"]
