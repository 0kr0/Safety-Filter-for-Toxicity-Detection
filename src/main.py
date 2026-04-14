"""
Main pipeline: train and evaluate all baselines on the same test set.
Usage:
  1. Download Jigsaw dataset from Kaggle: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
  2. Place train.csv in data/
  3. Run: python -m src.main [--baselines rule,tfidf,distilbert,ensemble] [--skip-neural]

New features (Week 9):
  --optimize-threshold   Find F1-optimal threshold instead of 0.5
  --augment              Augment minority class before training
  --adversarial          Run adversarial robustness evaluation
  --bias                 Run bias analysis across identity subgroups
  --cross-domain         Run cross-domain evaluation on Civil Comments
  --full-dataset         Use the full HuggingFace dataset (no sampling)
  --n-samples N          Number of samples for HuggingFace dataset
"""

import argparse
import json
from pathlib import Path

from .config import DATA_DIR, RESULTS_DIR
from .data import load_jigsaw, load_demo_data, load_hf_toxic
from .baselines import (
    RuleBasedClassifier,
    TFIDFLogisticRegressionClassifier,
    DistilBERTClassifier,
    EnsembleClassifier,
)
from .evaluation import run_baseline, print_results


def main():
    parser = argparse.ArgumentParser(description="Safety Filter for Toxicity Detection")
    parser.add_argument(
        "--baselines",
        type=str,
        default="rule,tfidf,distilbert",
        help="Comma-separated: rule, tfidf, distilbert, ensemble",
    )
    parser.add_argument("--skip-neural", action="store_true", help="Skip DistilBERT and ensemble")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data")
    parser.add_argument("--hf", action="store_true", help="Use HuggingFace Toxic Conversations dataset")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Path to data directory with train.csv")
    parser.add_argument("--output", type=Path, default=None, help="Save results JSON to this path")
    parser.add_argument("--n-samples", type=int, default=10000, help="Number of samples for HF dataset")
    parser.add_argument("--full-dataset", action="store_true", help="Use the full HF dataset (no sampling)")
    parser.add_argument("--optimize-threshold", action="store_true", help="Find F1-optimal threshold")
    parser.add_argument("--augment", action="store_true", help="Augment minority class before training")
    parser.add_argument("--adversarial", action="store_true", help="Run adversarial robustness evaluation")
    parser.add_argument("--bias", action="store_true", help="Run bias analysis")
    parser.add_argument("--cross-domain", action="store_true", help="Run cross-domain evaluation")
    args = parser.parse_args()

    baseline_names = [b.strip() for b in args.baselines.split(",")]
    if args.skip_neural:
        baseline_names = [b for b in baseline_names if b not in ("distilbert", "ensemble")]

    print("Loading data...")
    if args.demo:
        data = load_demo_data()
        print("  Using synthetic demo data (--demo)")
    elif args.hf:
        data = load_hf_toxic(n_samples=args.n_samples, full=args.full_dataset)
        print(f"  Using HuggingFace Toxic Conversations (full={args.full_dataset})")
    else:
        data = load_jigsaw(args.data_dir)
    print(f"  Train: {len(data['X_train'])} samples")
    print(f"  Val:   {len(data['X_val'])} samples")

    if args.augment:
        from .augmentation import augment_dataset
        original_size = len(data["X_train"])
        data["X_train"], data["y_train"] = augment_dataset(
            data["X_train"], data["y_train"], target_label=1, n_aug_per_sample=4
        )
        print(f"  Augmented train: {len(data['X_train'])} samples (was {original_size})")

    results = []

    if "rule" in baseline_names:
        model = RuleBasedClassifier()
        r = run_baseline("Baseline 1: Rule-based", model, data, has_proba=False)
        results.append(r)

    if "tfidf" in baseline_names:
        model = TFIDFLogisticRegressionClassifier()
        r = run_baseline(
            "Baseline 2: TF-IDF + Logistic Regression", model, data,
            has_proba=True, optimize_threshold=args.optimize_threshold,
        )
        results.append(r)

    if "distilbert" in baseline_names:
        model = DistilBERTClassifier()
        r = run_baseline(
            "Baseline 3: DistilBERT", model, data,
            has_proba=True, optimize_threshold=args.optimize_threshold,
        )
        results.append(r)

    if "ensemble" in baseline_names:
        model = EnsembleClassifier()
        r = run_baseline(
            "Baseline 4: Ensemble (DistilBERT + TF-IDF)", model, data,
            has_proba=True, optimize_threshold=args.optimize_threshold,
        )
        results.append(r)

    print_results(results)

    if args.adversarial:
        print("\n" + "=" * 60)
        print("ADVERSARIAL ROBUSTNESS EVALUATION")
        print("=" * 60)
        from .adversarial import adversarial_evaluate
        X_val, y_val = data["X_val"], data["y_val"]
        for r in results:
            name = r["name"]
            model_map = {
                "Baseline 1: Rule-based": RuleBasedClassifier,
                "Baseline 2: TF-IDF + Logistic Regression": TFIDFLogisticRegressionClassifier,
                "Baseline 3: DistilBERT": DistilBERTClassifier,
            }
            for key, cls in model_map.items():
                if key in name:
                    m = cls()
                    m.fit(data["X_train"], data["y_train"])
                    adv = adversarial_evaluate(m, X_val, y_val)
                    r["adversarial"] = adv
                    print(f"\n{name}:")
                    for attack, metrics in adv.items():
                        if attack == "clean":
                            print(f"  Clean F1: {metrics['f1']:.4f}")
                        else:
                            print(f"  {attack}: F1={metrics['f1']:.4f} (drop={metrics['f1_drop']:.4f}, flip_rate={metrics['flip_rate']:.3f})")
                    break

    if args.bias:
        print("\n")
        from .bias_analysis import compute_subgroup_metrics, compute_bias_metrics, format_bias_report
        X_val, y_val = data["X_val"], data["y_val"]
        for r in results:
            name = r["name"]
            model_map = {
                "Baseline 1: Rule-based": RuleBasedClassifier,
                "Baseline 2: TF-IDF + Logistic Regression": TFIDFLogisticRegressionClassifier,
                "Baseline 3: DistilBERT": DistilBERTClassifier,
            }
            for key, cls in model_map.items():
                if key in name:
                    m = cls()
                    m.fit(data["X_train"], data["y_train"])
                    y_pred = m.predict(X_val)
                    subgroup = compute_subgroup_metrics(X_val, y_val, y_pred)
                    bias = compute_bias_metrics(subgroup)
                    r["bias"] = {"subgroups": subgroup, "aggregate": bias}
                    print(format_bias_report(subgroup, bias))
                    break

    if args.cross_domain:
        print("\n" + "=" * 60)
        print("CROSS-DOMAIN EVALUATION")
        print("=" * 60)
        from .cross_domain import load_civil_comments, cross_domain_evaluate
        civil_data = load_civil_comments(n_samples=args.n_samples)
        print(f"  Civil Comments — Train: {len(civil_data['X_train'])}, Val: {len(civil_data['X_val'])}")

        for model_name, cls, proba in [
            ("Rule-based", RuleBasedClassifier, False),
            ("TF-IDF+LR", TFIDFLogisticRegressionClassifier, True),
        ]:
            m = cls()
            cd_metrics = cross_domain_evaluate(m, data, civil_data, has_proba=proba)
            print(f"\n  {model_name} (trained on Toxic Conv., eval on Civil Comments):")
            print(f"    F1={cd_metrics['f1']:.4f}  Prec={cd_metrics['precision']:.4f}  Rec={cd_metrics['recall']:.4f}")
            if "roc_auc" in cd_metrics:
                print(f"    AUC={cd_metrics['roc_auc']:.4f}")

    # Save results
    out_path = args.output or RESULTS_DIR / "baseline_results.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_serializable = []
    for r in results:
        rr = {
            "name": r["name"],
            "train_time_s": r["train_time_s"],
            "inference_time_s": r["inference_time_s"],
            "metrics": {k: (float(v) if v is not None else None) for k, v in r["metrics"].items()},
        }
        if "optimal_threshold" in r:
            rr["optimal_threshold"] = r["optimal_threshold"]
            rr["metrics_optimized"] = {
                k: (float(v) if v is not None else None) for k, v in r["metrics_optimized"].items()
            }
        results_serializable.append(rr)

    with open(out_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
