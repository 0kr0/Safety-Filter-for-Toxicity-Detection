"""
Adversarial evaluation: test model robustness to text perturbations.
Implements character substitution, leetspeak, typo injection, and homoglyph attacks.
"""

import re
import random
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

LEET_MAP = {
    "a": ["@", "4"],
    "e": ["3"],
    "i": ["1", "!"],
    "o": ["0"],
    "s": ["$", "5"],
    "t": ["7"],
    "l": ["1"],
    "b": ["8"],
    "g": ["9"],
}

HOMOGLYPH_MAP = {
    "a": "\u0430",  # Cyrillic а
    "c": "\u0441",  # Cyrillic с
    "e": "\u0435",  # Cyrillic е
    "o": "\u043e",  # Cyrillic о
    "p": "\u0440",  # Cyrillic р
    "x": "\u0445",  # Cyrillic х
}


def leetspeak_transform(text: str, prob: float = 0.5) -> str:
    """Replace characters with leetspeak equivalents probabilistically."""
    result = []
    for ch in text:
        if ch.lower() in LEET_MAP and random.random() < prob:
            result.append(random.choice(LEET_MAP[ch.lower()]))
        else:
            result.append(ch)
    return "".join(result)


def char_substitution(text: str, prob: float = 0.1) -> str:
    """Randomly substitute characters with nearby keyboard keys."""
    keyboard_neighbors = {
        "a": "sqw", "b": "vgn", "c": "xdf", "d": "sfce", "e": "rdw",
        "f": "dgcr", "g": "fhtb", "h": "gjyn", "i": "uko", "j": "hkum",
        "k": "jli", "l": "ko", "m": "nj", "n": "bmh", "o": "ilp",
        "p": "ol", "q": "wa", "r": "eft", "s": "adwx", "t": "rgy",
        "u": "yji", "v": "cfb", "w": "qase", "x": "zsc", "y": "tuh",
        "z": "xa",
    }
    result = []
    for ch in text:
        if ch.lower() in keyboard_neighbors and random.random() < prob:
            result.append(random.choice(keyboard_neighbors[ch.lower()]))
        else:
            result.append(ch)
    return "".join(result)


def insert_spaces(text: str) -> str:
    """Insert spaces within words to evade word-level detection.  e.g. 'idiot' -> 'i d i o t'."""
    words = text.split()
    result = []
    for word in words:
        if len(word) > 3 and random.random() < 0.4:
            result.append(" ".join(word))
        else:
            result.append(word)
    return " ".join(result)


def homoglyph_attack(text: str, prob: float = 0.3) -> str:
    """Replace characters with visually similar Unicode homoglyphs."""
    result = []
    for ch in text:
        if ch.lower() in HOMOGLYPH_MAP and random.random() < prob:
            result.append(HOMOGLYPH_MAP[ch.lower()])
        else:
            result.append(ch)
    return "".join(result)


def repeat_chars(text: str, prob: float = 0.15) -> str:
    """Repeat characters to obfuscate words.  e.g. 'stupid' -> 'stuuupid'."""
    result = []
    for ch in text:
        result.append(ch)
        if ch.isalpha() and random.random() < prob:
            result.append(ch * random.randint(1, 3))
    return "".join(result)


ATTACK_REGISTRY = {
    "leetspeak": leetspeak_transform,
    "char_substitution": char_substitution,
    "insert_spaces": insert_spaces,
    "homoglyph": homoglyph_attack,
    "repeat_chars": repeat_chars,
}


def adversarial_evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    attacks: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """
    Evaluate model robustness against each attack type.
    Returns per-attack metrics and the drop in F1 from clean predictions.
    """
    random.seed(seed)
    attacks = attacks or list(ATTACK_REGISTRY.keys())

    clean_preds = model.predict(X)
    clean_f1 = f1_score(y, clean_preds, zero_division=0)

    results = {
        "clean": {
            "f1": clean_f1,
            "precision": precision_score(y, clean_preds, zero_division=0),
            "recall": recall_score(y, clean_preds, zero_division=0),
        }
    }

    for attack_name in attacks:
        attack_fn = ATTACK_REGISTRY[attack_name]
        X_adv = np.array([attack_fn(str(x)) for x in X])
        adv_preds = model.predict(X_adv)

        adv_f1 = f1_score(y, adv_preds, zero_division=0)
        results[attack_name] = {
            "f1": adv_f1,
            "precision": precision_score(y, adv_preds, zero_division=0),
            "recall": recall_score(y, adv_preds, zero_division=0),
            "f1_drop": clean_f1 - adv_f1,
            "flip_rate": float(np.mean(clean_preds != adv_preds)),
            "example_original": str(X[0])[:100] if len(X) > 0 else "",
            "example_perturbed": str(X_adv[0])[:100] if len(X_adv) > 0 else "",
        }

    return results
