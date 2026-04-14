"""
Bias analysis: check if the classifier disproportionately flags certain identity subgroups.
Inspired by Borkan et al. (2019) — measuring unintended bias with real data.
"""

import re
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

IDENTITY_TERMS = {
    "gender": {
        "male": ["he", "him", "his", "man", "men", "boy", "boys", "male", "father", "husband", "son"],
        "female": ["she", "her", "hers", "woman", "women", "girl", "girls", "female", "mother", "wife", "daughter"],
    },
    "religion": {
        "christian": ["christian", "christianity", "church", "bible", "jesus", "catholic", "protestant"],
        "muslim": ["muslim", "islam", "islamic", "mosque", "quran", "allah"],
        "jewish": ["jewish", "jew", "jews", "judaism", "synagogue", "torah"],
        "atheist": ["atheist", "atheism", "agnostic"],
    },
    "race_ethnicity": {
        "black": ["black", "african", "african american"],
        "white": ["white", "caucasian", "european"],
        "asian": ["asian", "chinese", "japanese", "korean", "indian"],
        "hispanic": ["hispanic", "latino", "latina", "mexican"],
    },
    "sexuality": {
        "lgbtq": ["gay", "lesbian", "bisexual", "transgender", "queer", "lgbtq", "homosexual"],
        "straight": ["straight", "heterosexual"],
    },
}


def _text_mentions_group(text: str, terms: list[str]) -> bool:
    """Check if text mentions any of the identity terms."""
    text_lower = text.lower()
    words = set(re.findall(r"\b\w+\b", text_lower))
    for term in terms:
        if " " in term:
            if term in text_lower:
                return True
        else:
            if term in words:
                return True
    return False


def compute_subgroup_metrics(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_samples: int = 10,
) -> dict:
    """
    Compute per-subgroup metrics (FPR, FNR, F1) to identify bias.
    Only includes subgroups with >= min_samples examples.
    """
    results = {}

    for category, groups in IDENTITY_TERMS.items():
        results[category] = {}
        for group_name, terms in groups.items():
            mask = np.array([_text_mentions_group(str(x), terms) for x in X])
            n_samples = mask.sum()

            if n_samples < min_samples:
                continue

            y_t = y_true[mask]
            y_p = y_pred[mask]

            n_pos = y_t.sum()
            n_neg = len(y_t) - n_pos
            fp = ((y_p == 1) & (y_t == 0)).sum()
            fn = ((y_p == 0) & (y_t == 1)).sum()

            results[category][group_name] = {
                "n_samples": int(n_samples),
                "n_toxic": int(n_pos),
                "n_nontoxic": int(n_neg),
                "toxic_rate": float(n_pos / n_samples) if n_samples > 0 else 0.0,
                "fpr": float(fp / n_neg) if n_neg > 0 else 0.0,
                "fnr": float(fn / n_pos) if n_pos > 0 else 0.0,
                "f1": f1_score(y_t, y_p, zero_division=0),
                "precision": precision_score(y_t, y_p, zero_division=0),
                "recall": recall_score(y_t, y_p, zero_division=0),
            }

    return results


def compute_bias_metrics(subgroup_results: dict) -> dict:
    """
    Compute aggregate bias metrics across subgroups.
    Returns max FPR gap, max FNR gap, and overall bias score.
    """
    all_fprs = []
    all_fnrs = []
    all_f1s = []

    for category, groups in subgroup_results.items():
        for group_name, metrics in groups.items():
            all_fprs.append(metrics["fpr"])
            all_fnrs.append(metrics["fnr"])
            all_f1s.append(metrics["f1"])

    if not all_fprs:
        return {"fpr_gap": 0.0, "fnr_gap": 0.0, "f1_gap": 0.0, "bias_score": 0.0}

    fpr_gap = max(all_fprs) - min(all_fprs)
    fnr_gap = max(all_fnrs) - min(all_fnrs)
    f1_gap = max(all_f1s) - min(all_f1s)

    bias_score = (fpr_gap + fnr_gap + f1_gap) / 3

    return {
        "fpr_gap": fpr_gap,
        "fnr_gap": fnr_gap,
        "f1_gap": f1_gap,
        "bias_score": bias_score,
        "n_subgroups_evaluated": len(all_fprs),
    }


def format_bias_report(subgroup_results: dict, bias_metrics: dict) -> str:
    """Format bias analysis results as a readable report."""
    lines = ["=" * 60, "BIAS ANALYSIS REPORT", "=" * 60, ""]

    for category, groups in subgroup_results.items():
        if not groups:
            continue
        lines.append(f"\n--- {category.upper()} ---")
        for group_name, m in sorted(groups.items()):
            lines.append(
                f"  {group_name:15s}: n={m['n_samples']:4d}  "
                f"FPR={m['fpr']:.3f}  FNR={m['fnr']:.3f}  F1={m['f1']:.3f}  "
                f"toxic_rate={m['toxic_rate']:.3f}"
            )

    lines.extend([
        "\n--- AGGREGATE BIAS METRICS ---",
        f"  FPR gap (max-min): {bias_metrics['fpr_gap']:.4f}",
        f"  FNR gap (max-min): {bias_metrics['fnr_gap']:.4f}",
        f"  F1 gap  (max-min): {bias_metrics['f1_gap']:.4f}",
        f"  Bias score (avg):  {bias_metrics['bias_score']:.4f}",
        f"  Subgroups evaluated: {bias_metrics['n_subgroups_evaluated']}",
    ])

    return "\n".join(lines)
