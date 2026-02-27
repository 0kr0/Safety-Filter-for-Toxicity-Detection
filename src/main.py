"""
Main pipeline: train and evaluate all baselines on the same test set.
Usage:
  1. Download Jigsaw dataset from Kaggle: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
  2. Place train.csv in data/
  3. Run: python -m src.main [--baselines rule,tfidf,distilbert] [--skip-neural]
"""

import argparse
import json
from pathlib import Path

from .config import DATA_DIR, RESULTS_DIR
from .data import load_jigsaw, load_demo_data
from .baselines import RuleBasedClassifier, TFIDFLogisticRegressionClassifier, DistilBERTClassifier
from .evaluation import run_baseline, print_results


def main():
    parser = argparse.ArgumentParser(description="Safety Filter for Toxicity Detection - Baselines")
    parser.add_argument(
        "--baselines",
        type=str,
        default="rule,tfidf,distilbert",
        help="Comma-separated: rule, tfidf, distilbert",
    )
    parser.add_argument(
        "--skip-neural",
        action="store_true",
        help="Skip DistilBERT (slower, requires more memory)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic demo data (when train.csv not available)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Path to data directory with train.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save results JSON to this path",
    )
    args = parser.parse_args()

    baseline_names = [b.strip() for b in args.baselines.split(",")]
    if args.skip_neural:
        baseline_names = [b for b in baseline_names if b != "distilbert"]

    print("Loading data...")
    if args.demo:
        data = load_demo_data()
        print("  Using synthetic demo data (--demo)")
    else:
        data = load_jigsaw(args.data_dir)
    print(f"  Train: {len(data['X_train'])} samples")
    print(f"  Val:   {len(data['X_val'])} samples")

    results = []

    if "rule" in baseline_names:
        model = RuleBasedClassifier()
        r = run_baseline("Baseline 1: Rule-based", model, data, has_proba=False)
        results.append(r)

    if "tfidf" in baseline_names:
        model = TFIDFLogisticRegressionClassifier()
        r = run_baseline("Baseline 2: TF-IDF + Logistic Regression", model, data, has_proba=True)
        results.append(r)

    if "distilbert" in baseline_names:
        model = DistilBERTClassifier()
        r = run_baseline("Baseline 3: DistilBERT", model, data, has_proba=False)
        results.append(r)

    print_results(results)

    # Save results
    out_path = args.output or RESULTS_DIR / "baseline_results.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Convert for JSON (numpy types, etc.)
    results_serializable = []
    for r in results:
        rr = {
            "name": r["name"],
            "train_time_s": r["train_time_s"],
            "inference_time_s": r["inference_time_s"],
            "metrics": {k: (float(v) if v is not None else None) for k, v in r["metrics"].items()},
        }
        results_serializable.append(rr)
    with open(out_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
