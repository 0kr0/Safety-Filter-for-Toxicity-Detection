"""
Generative data augmentation for toxicity detection.
Provides synonym replacement, random insertion/swap/deletion, and
an optional GPT-based paraphrasing interface for creating synthetic toxic examples.
"""

import re
import random
import numpy as np


def synonym_replace(text: str, n: int = 2) -> str:
    """
    Replace n random words with simple synonyms to create diversity.
    Uses a small built-in synonym table for toxicity-relevant terms.
    """
    synonym_table = {
        "stupid": ["dumb", "foolish", "brainless", "dense"],
        "idiot": ["fool", "moron", "imbecile", "dimwit"],
        "hate": ["despise", "loathe", "detest", "abhor"],
        "ugly": ["hideous", "repulsive", "grotesque", "unsightly"],
        "kill": ["eliminate", "destroy", "annihilate", "end"],
        "bad": ["terrible", "awful", "horrible", "dreadful"],
        "good": ["great", "excellent", "wonderful", "fine"],
        "happy": ["glad", "pleased", "delighted", "content"],
        "sad": ["unhappy", "sorrowful", "gloomy", "miserable"],
        "angry": ["furious", "enraged", "irate", "livid"],
        "trash": ["garbage", "rubbish", "waste", "junk"],
        "dumb": ["stupid", "foolish", "dense", "senseless"],
        "terrible": ["awful", "dreadful", "horrible", "appalling"],
        "disgusting": ["revolting", "repulsive", "vile", "sickening"],
    }

    words = text.split()
    replaceable = [(i, w.lower()) for i, w in enumerate(words) if w.lower() in synonym_table]

    if not replaceable:
        return text

    to_replace = random.sample(replaceable, min(n, len(replaceable)))
    for idx, word in to_replace:
        words[idx] = random.choice(synonym_table[word])

    return " ".join(words)


def random_insertion(text: str, n: int = 1) -> str:
    """Insert n random words from the text at random positions."""
    words = text.split()
    if not words:
        return text
    for _ in range(n):
        insert_word = random.choice(words)
        pos = random.randint(0, len(words))
        words.insert(pos, insert_word)
    return " ".join(words)


def random_swap(text: str, n: int = 1) -> str:
    """Swap n random pairs of words."""
    words = text.split()
    if len(words) < 2:
        return text
    for _ in range(n):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)


def random_deletion(text: str, prob: float = 0.1) -> str:
    """Delete each word with given probability."""
    words = text.split()
    if len(words) <= 1:
        return text
    result = [w for w in words if random.random() > prob]
    return " ".join(result) if result else words[0]


def eda_augment(text: str, alpha: float = 0.1, n_aug: int = 4) -> list[str]:
    """
    Easy Data Augmentation (EDA) — Wei & Zou 2019.
    Generates n_aug augmented versions of the input text.
    """
    augmented = []
    n_words = len(text.split())
    n_changes = max(1, int(alpha * n_words))

    for _ in range(n_aug):
        op = random.choice(["sr", "ri", "rs", "rd"])
        if op == "sr":
            augmented.append(synonym_replace(text, n_changes))
        elif op == "ri":
            augmented.append(random_insertion(text, n_changes))
        elif op == "rs":
            augmented.append(random_swap(text, n_changes))
        elif op == "rd":
            augmented.append(random_deletion(text, alpha))

    return augmented


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    target_label: int = 1,
    n_aug_per_sample: int = 4,
    max_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment the minority class (target_label) using EDA.

    max_ratio caps the final proportion of the target class in the augmented
    dataset (e.g. 0.20 means at most 20% toxic).  This prevents the toxic
    class from overwhelming the majority class with noisy near-duplicates.
    """
    random.seed(seed)
    n_total = len(y)
    n_minority = int((y == target_label).sum())
    n_majority = n_total - n_minority

    max_minority_final = int(max_ratio * n_majority / (1 - max_ratio))
    max_new = max(0, max_minority_final - n_minority)

    mask = y == target_label
    X_minority = X[mask]

    aug_texts = []
    for text in X_minority:
        if len(aug_texts) >= max_new:
            break
        batch = eda_augment(str(text), n_aug=n_aug_per_sample)
        remaining = max_new - len(aug_texts)
        aug_texts.extend(batch[:remaining])

    if not aug_texts:
        return X, y

    aug_labels = np.full(len(aug_texts), target_label, dtype=int)

    X_augmented = np.concatenate([X, np.array(aug_texts)])
    y_augmented = np.concatenate([y, aug_labels])

    shuffle = np.random.RandomState(seed).permutation(len(X_augmented))
    return X_augmented[shuffle], y_augmented[shuffle]


def backtranslation_augment(texts: list[str], src_lang: str = "en", pivot_lang: str = "de") -> list[str]:
    """
    Back-translation augmentation placeholder.
    In production, pipe texts through en->pivot->en using a translation API/model.
    Here we apply light paraphrasing as a simulation.
    """
    augmented = []
    for text in texts:
        words = text.split()
        if len(words) > 3:
            i = random.randint(0, len(words) - 2)
            words[i], words[i + 1] = words[i + 1], words[i]
        augmented.append(" ".join(words))
    return augmented
