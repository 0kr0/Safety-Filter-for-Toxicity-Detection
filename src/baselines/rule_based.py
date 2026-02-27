"""Baseline 1: Rule-based keyword matching with bad words list."""

import re
from pathlib import Path
import numpy as np

from ..config import PROJECT_ROOT


class RuleBasedClassifier:
    """
    Simple rule-based toxicity detector.
    Marks text as toxic if it contains words from a bad-words list.
    """

    def __init__(self, bad_words_path: Path = None):
        self.bad_words_path = bad_words_path or (PROJECT_ROOT / "data" / "bad_words.txt")
        self.bad_words = set()
        self._load_bad_words()

    def _load_bad_words(self):
        if self.bad_words_path.exists():
            with open(self.bad_words_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip().lower()
                    if line and not line.startswith("#"):
                        self.bad_words.add(line)
        else:
            # Fallback minimal set
            self.bad_words = {"stupid", "idiot", "hate", "kill", "dumb", "ugly", "trash"}

    def predict(self, X) -> np.ndarray:
        """Predict 1 if toxic, 0 otherwise."""
        if isinstance(X, str):
            X = [X]
        X = np.atleast_1d(X)
        preds = np.zeros(len(X), dtype=int)
        for i, text in enumerate(X):
            text = str(text).lower()
            words = set(re.findall(r"\b\w+\b", text))
            if words & self.bad_words:
                preds[i] = 1
        return preds

    def fit(self, X, y=None):
        """No training needed for rule-based model."""
        return self
