"""
Full experiment suite for toxicity detection — Week 9 improvements.
Includes: baseline comparison, threshold optimization, ensemble, adversarial evaluation,
cross-domain testing, data augmentation ablation, and bias analysis.

Usage: python experiments/run_all.py [--skip-neural] [--skip-cross-domain] [--full-dataset]
"""

import sys, os, json, time, warnings, argparse
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)

from src.baselines import (
    RuleBasedClassifier, TFIDFLogisticRegressionClassifier,
    DistilBERTClassifier, EnsembleClassifier,
)
from src.evaluation import evaluate, measure_inference_time
from src.data import load_hf_toxic
from src.threshold_optimizer import find_optimal_threshold, apply_threshold
from src.adversarial import adversarial_evaluate
from src.bias_analysis import compute_subgroup_metrics, compute_bias_metrics, format_bias_report
from src.augmentation import augment_dataset

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

FIGURES_DIR = os.path.join(PROJECT_ROOT, "report", "figures")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
SEED = 42


def savefig(fig, name):
    fig.savefig(os.path.join(FIGURES_DIR, name)); plt.close(fig)
    print(f"  -> {name}")


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_class_dist(y_tr, y_val):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, d, t in [(axes[0], y_tr, "Train"), (axes[1], y_val, "Validation")]:
        c = np.bincount(d, minlength=2)
        bars = ax.bar(["Non-toxic", "Toxic"], c, color=["steelblue", "coral"])
        for b, v in zip(bars, c):
            ax.text(b.get_x()+b.get_width()/2, v+5, str(v), ha="center", fontweight="bold")
        ax.set_title(t); ax.set_ylabel("Count")
        ax.text(0.95, 0.95, f"Toxic: {c[1]/c.sum()*100:.1f}%", transform=ax.transAxes,
                ha="right", va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.suptitle("Class Distribution", fontweight="bold"); fig.tight_layout()
    savefig(fig, "class_distribution.pdf")

def plot_metrics_comparison(results):
    metrics = ["precision", "recall", "f1", "fpr", "fnr"]
    x = np.arange(len(metrics)); w = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, r in enumerate(results):
        vals = [r["m"].get(m, 0) for m in metrics]
        bars = ax.bar(x + i*w, vals, w, label=r["name"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x + w*(len(results)-1)/2)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score"); ax.set_title("Baseline Comparison"); ax.legend(fontsize=8)
    fig.tight_layout(); savefig(fig, "metrics_comparison.pdf")

def plot_confusion_matrices(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1: axes = [axes]
    for ax, r in zip(axes, results):
        cm = confusion_matrix(r["y_true"], r["preds"], labels=[0,1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Non-toxic","Toxic"], yticklabels=["Non-toxic","Toxic"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(r["name"])
    fig.tight_layout(); savefig(fig, "confusion_matrices.pdf")

def plot_roc(roc_data):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, fpr, tpr, auc in roc_data:
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
    ax.plot([0,1],[0,1],"k--",alpha=0.5,label="Random")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curves"); ax.legend()
    fig.tight_layout(); savefig(fig, "roc_curves.pdf")

def plot_pr(pr_data):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, prec, rec in pr_data:
        ax.plot(rec, prec, label=name, linewidth=2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall Curves"); ax.legend()
    fig.tight_layout(); savefig(fig, "pr_curves.pdf")

def plot_threshold(y_true, y_proba, name, fname):
    thresholds = np.linspace(0.01, 0.99, 100)
    p, r, f, fp = [], [], [], []
    for t in thresholds:
        yp = (y_proba >= t).astype(int)
        p.append(precision_score(y_true, yp, zero_division=0))
        r.append(recall_score(y_true, yp, zero_division=0))
        f.append(f1_score(y_true, yp, zero_division=0))
        cm = confusion_matrix(y_true, yp, labels=[0,1]); tn_,fp_,fn_,tp_ = cm.ravel()
        fp.append(fp_/(fp_+tn_) if (fp_+tn_)>0 else 0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, p, label="Precision", lw=2)
    ax.plot(thresholds, r, label="Recall", lw=2)
    ax.plot(thresholds, f, label="F1", lw=2)
    ax.plot(thresholds, fp, label="FPR", lw=2, ls="--")
    best = thresholds[np.argmax(f)]
    ax.axvline(best, color="gray", ls=":", alpha=0.7, label=f"Best F1 @{best:.2f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title(f"Threshold Analysis: {name}"); ax.legend(fontsize=9)
    fig.tight_layout(); savefig(fig, fname)

def plot_loss(losses, name):
    if not losses: return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(losses)+1), losses, "o-", lw=2, ms=3)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_title(f"Training Loss: {name}")
    fig.tight_layout(); savefig(fig, "distilbert_training_loss.pdf")

def plot_timing(results):
    names = [r["name"] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bars = axes[0].barh(names, [r["infer_t"] for r in results], color=sns.color_palette("viridis", len(names)))
    axes[0].set_xlabel("Inference Time (s)"); axes[0].set_title("Inference Speed")
    for b, r in zip(bars, results):
        axes[0].text(b.get_width()*1.02, b.get_y()+b.get_height()/2, f"{r['infer_t']:.4f}s", va="center", fontsize=9)
    bars2 = axes[1].barh(names, [r["train_t"] for r in results], color=sns.color_palette("magma", len(names)))
    axes[1].set_xlabel("Training Time (s)"); axes[1].set_title("Training Speed")
    for b, r in zip(bars2, results):
        axes[1].text(b.get_width()*1.02, b.get_y()+b.get_height()/2, f"{r['train_t']:.2f}s", va="center", fontsize=9)
    fig.tight_layout(); savefig(fig, "timing_comparison.pdf")

def plot_error_analysis(r):
    preds, y_true, X_val = r["preds"], r["y_true"], r["X_val"]
    fp_mask = (preds==1)&(y_true==0); fn_mask = (preds==0)&(y_true==1); ok_mask = (preds==y_true)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    data_h, labels_h = [], []
    for mask, lbl in [(ok_mask,"Correct"),(fp_mask,"False Positive"),(fn_mask,"False Negative")]:
        lens = [len(X_val[i]) for i in range(len(X_val)) if mask[i]]
        if lens: data_h.append(lens); labels_h.append(lbl)
    if data_h: axes[0].hist(data_h, bins=20, label=labels_h, alpha=0.7, density=True)
    axes[0].set_xlabel("Text Length (chars)"); axes[0].set_ylabel("Density")
    axes[0].set_title("Error by Text Length"); axes[0].legend()
    cm = confusion_matrix(y_true, preds, labels=[0,1]); tn,fp_,fn,tp = cm.ravel()
    types = ["TP","TN","FP","FN"]; counts = [tp,tn,fp_,fn]
    axes[1].bar(types, counts, color=["#2ecc71","#3498db","#e74c3c","#e67e22"])
    for i, c in enumerate(counts): axes[1].text(i, c+0.5, str(c), ha="center", fontweight="bold")
    axes[1].set_ylabel("Count"); axes[1].set_title("Prediction Breakdown")
    fig.suptitle(f"Error Analysis: {r['name']}", fontweight="bold"); fig.tight_layout()
    safe = r["name"].lower().replace(" ","_").replace("+","")
    savefig(fig, f"error_analysis_{safe}.pdf")

def plot_data_ablation(abl):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, d in abl.items():
        ax.plot(d["sizes"], d["f1"], "o-", label=name, lw=2)
    ax.set_xlabel("Training Set Size"); ax.set_ylabel("F1 Score")
    ax.set_title("Effect of Training Data Size on F1"); ax.legend()
    fig.tight_layout(); savefig(fig, "data_size_ablation.pdf")

def plot_hyperparam(sweep):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(sweep["C_vals"], sweep["C_f1"], "o-", lw=2, color="teal")
    axes[0].set_xscale("log"); axes[0].set_xlabel("C"); axes[0].set_ylabel("F1")
    axes[0].set_title("TF-IDF+LR: Regularization C")
    best = sweep["C_vals"][np.argmax(sweep["C_f1"])]
    axes[0].axvline(best, color="red", ls="--", alpha=0.7, label=f"Best C={best}")
    axes[0].legend()
    axes[1].plot(sweep["feat_vals"], sweep["feat_f1"], "s-", lw=2, color="purple")
    axes[1].set_xlabel("Max Features"); axes[1].set_ylabel("F1")
    axes[1].set_title("TF-IDF+LR: Vocabulary Size")
    best2 = sweep["feat_vals"][np.argmax(sweep["feat_f1"])]
    axes[1].axvline(best2, color="red", ls="--", alpha=0.7, label=f"Best={best2}")
    axes[1].legend()
    fig.tight_layout(); savefig(fig, "hyperparam_sweep.pdf")


# ── NEW: Threshold optimization plot ─────────────────────────────────────────

def plot_threshold_optimization(threshold_results):
    """Plot before/after threshold optimization comparison."""
    names = list(threshold_results.keys())
    default_f1 = [threshold_results[n]["default_f1"] for n in names]
    optimized_f1 = [threshold_results[n]["optimized_f1"] for n in names]
    thresholds = [threshold_results[n]["threshold"] for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - 0.2, default_f1, 0.35, label="Default (t=0.5)", color="steelblue")
    bars2 = ax.bar(x + 0.2, optimized_f1, 0.35, label="Optimized", color="coral")
    for b, v in zip(bars1, default_f1):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.3f}", ha="center", fontsize=9)
    for b, v, t in zip(bars2, optimized_f1, thresholds):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.3f}\n(t={t:.2f})", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("F1 Score"); ax.set_title("Threshold Optimization: F1 Improvement")
    ax.legend(); ax.set_ylim(0, max(max(default_f1), max(optimized_f1)) * 1.2)
    fig.tight_layout(); savefig(fig, "threshold_optimization.pdf")


# ── NEW: Adversarial robustness plot ──────────────────────────────────────────

def plot_adversarial(adv_results):
    """Plot adversarial robustness across attack types and models."""
    models = list(adv_results.keys())
    attacks = [k for k in list(adv_results.values())[0].keys() if k != "clean"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(attacks))
    w = 0.8 / len(models)
    for i, model in enumerate(models):
        f1s = [adv_results[model][a]["f1"] for a in attacks]
        axes[0].bar(x + i*w, f1s, w, label=model)
    axes[0].set_xticks(x + w*(len(models)-1)/2)
    axes[0].set_xticklabels(attacks, rotation=30, ha="right")
    axes[0].set_ylabel("F1"); axes[0].set_title("F1 Under Adversarial Attacks")
    axes[0].legend(fontsize=8)

    for i, model in enumerate(models):
        flips = [adv_results[model][a]["flip_rate"] for a in attacks]
        axes[1].bar(x + i*w, flips, w, label=model)
    axes[1].set_xticks(x + w*(len(models)-1)/2)
    axes[1].set_xticklabels(attacks, rotation=30, ha="right")
    axes[1].set_ylabel("Flip Rate"); axes[1].set_title("Prediction Flip Rate Under Attacks")
    axes[1].legend(fontsize=8)

    fig.tight_layout(); savefig(fig, "adversarial_robustness.pdf")


# ── NEW: Bias analysis plot ───────────────────────────────────────────────────

def plot_bias(bias_results):
    """Plot per-subgroup FPR and FNR for bias analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    all_groups = []
    all_fprs = []
    all_fnrs = []
    for category, groups in bias_results.items():
        for group, m in groups.items():
            all_groups.append(f"{group}\n({category})")
            all_fprs.append(m["fpr"])
            all_fnrs.append(m["fnr"])

    if not all_groups:
        plt.close(fig)
        return

    y_pos = np.arange(len(all_groups))
    axes[0].barh(y_pos, all_fprs, color="coral", alpha=0.8)
    axes[0].set_yticks(y_pos); axes[0].set_yticklabels(all_groups, fontsize=8)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_title("FPR by Identity Subgroup")
    for i, v in enumerate(all_fprs):
        axes[0].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)

    axes[1].barh(y_pos, all_fnrs, color="steelblue", alpha=0.8)
    axes[1].set_yticks(y_pos); axes[1].set_yticklabels(all_groups, fontsize=8)
    axes[1].set_xlabel("False Negative Rate"); axes[1].set_title("FNR by Identity Subgroup")
    for i, v in enumerate(all_fnrs):
        axes[1].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)

    fig.suptitle("Bias Analysis: Per-Subgroup Error Rates", fontweight="bold")
    fig.tight_layout(); savefig(fig, "bias_analysis.pdf")


# ── NEW: Augmentation comparison plot ─────────────────────────────────────────

def plot_augmentation_comparison(aug_results):
    """Plot F1 with and without data augmentation."""
    names = list(aug_results.keys())
    baseline_f1 = [aug_results[n]["baseline"] for n in names]
    augmented_f1 = [aug_results[n]["augmented"] for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - 0.2, baseline_f1, 0.35, label="No Augmentation", color="steelblue")
    bars2 = ax.bar(x + 0.2, augmented_f1, 0.35, label="With EDA Augmentation", color="coral")
    for b, v in zip(bars1, baseline_f1):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.3f}", ha="center", fontsize=9)
    for b, v in zip(bars2, augmented_f1):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("F1 Score"); ax.set_title("Effect of Data Augmentation on F1")
    ax.legend()
    fig.tight_layout(); savefig(fig, "augmentation_comparison.pdf")


# ── NEW: Cross-domain plot ────────────────────────────────────────────────────

def plot_cross_domain(cd_results):
    """Plot in-domain vs cross-domain F1 comparison."""
    names = list(cd_results.keys())
    in_domain = [cd_results[n]["in_domain_f1"] for n in names]
    cross_domain = [cd_results[n]["cross_domain_f1"] for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - 0.2, in_domain, 0.35, label="In-Domain", color="steelblue")
    bars2 = ax.bar(x + 0.2, cross_domain, 0.35, label="Cross-Domain", color="coral")
    for b, v in zip(bars1, in_domain):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.3f}", ha="center", fontsize=9)
    for b, v in zip(bars2, cross_domain):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("F1 Score"); ax.set_title("In-Domain vs Cross-Domain Generalization")
    ax.legend()
    fig.tight_layout(); savefig(fig, "cross_domain_comparison.pdf")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-neural", action="store_true", help="Skip DistilBERT + Ensemble")
    parser.add_argument("--skip-cross-domain", action="store_true", help="Skip cross-domain (needs download)")
    parser.add_argument("--full-dataset", action="store_true", help="Use full HF dataset")
    parser.add_argument("--n-samples", type=int, default=10000, help="Number of HF samples")
    args = parser.parse_args()

    print("=" * 60)
    print("TOXICITY DETECTION — FULL EXPERIMENT SUITE (WEEK 9)")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    step = 1
    total_steps = 12 if not args.skip_neural else 10
    if args.skip_cross_domain:
        total_steps -= 1

    print(f"\n[{step}/{total_steps}] Loading data via src.data.load_hf_toxic")
    data = load_hf_toxic(n_samples=args.n_samples, full=args.full_dataset)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"  Toxic ratio — train: {y_train.mean():.3f}, val: {y_val.mean():.3f}")
    plot_class_dist(y_train, y_val)

    # ── 2. Train baselines ────────────────────────────────────────────────────
    step += 1
    print(f"\n[{step}/{total_steps}] Training baselines")
    all_results = []; roc_data = []; pr_data = []

    print("\n  >> RuleBasedClassifier")
    rule = RuleBasedClassifier()
    t0 = time.perf_counter(); rule.fit(X_train, y_train); rule_tt = time.perf_counter()-t0
    t0 = time.perf_counter(); rule_p = rule.predict(X_val); rule_it = time.perf_counter()-t0
    rule_m = evaluate(y_val, rule_p)
    rule_m["fpr"] = rule_m.pop("false_positive_rate")
    rule_m["fnr"] = rule_m.pop("false_negative_rate")
    print(f"     F1={rule_m['f1']:.4f} Prec={rule_m['precision']:.4f} Rec={rule_m['recall']:.4f}")
    all_results.append({"name":"Rule-based","m":rule_m,"train_t":rule_tt,"infer_t":rule_it,
                         "preds":rule_p,"y_true":y_val,"X_val":X_val})

    print("\n  >> TFIDFLogisticRegressionClassifier")
    tfidf = TFIDFLogisticRegressionClassifier()
    t0 = time.perf_counter(); tfidf.fit(X_train, y_train); tfidf_tt = time.perf_counter()-t0
    t0 = time.perf_counter(); tfidf_p = tfidf.predict(X_val); tfidf_it = time.perf_counter()-t0
    tfidf_proba = tfidf.predict_proba(X_val)[:,1]
    tfidf_m = evaluate(y_val, tfidf_p, tfidf_proba)
    tfidf_m["fpr"] = tfidf_m.pop("false_positive_rate")
    tfidf_m["fnr"] = tfidf_m.pop("false_negative_rate")
    print(f"     F1={tfidf_m['f1']:.4f} Prec={tfidf_m['precision']:.4f} Rec={tfidf_m['recall']:.4f} AUC={tfidf_m.get('roc_auc',0):.4f}")
    all_results.append({"name":"TF-IDF + LR","m":tfidf_m,"train_t":tfidf_tt,"infer_t":tfidf_it,
                         "preds":tfidf_p,"proba":tfidf_proba,"y_true":y_val,"X_val":X_val})
    fpr_c,tpr_c,_ = roc_curve(y_val, tfidf_proba)
    roc_data.append(("TF-IDF+LR", fpr_c, tpr_c, tfidf_m["roc_auc"]))
    pr_c,rc_c,_ = precision_recall_curve(y_val, tfidf_proba)
    pr_data.append(("TF-IDF+LR", pr_c, rc_c))

    bert = None
    bert_proba = None
    if not args.skip_neural:
        print("\n  >> DistilBERTClassifier")
        bert = DistilBERTClassifier()
        t0 = time.perf_counter(); bert.fit(X_train, y_train, epochs=3); bert_tt = time.perf_counter()-t0
        t0 = time.perf_counter(); bert_p = bert.predict(X_val); bert_it = time.perf_counter()-t0
        bert_proba = bert.predict_proba(X_val)[:,1]
        bert_m = evaluate(y_val, bert_p, bert_proba)
        bert_m["fpr"] = bert_m.pop("false_positive_rate")
        bert_m["fnr"] = bert_m.pop("false_negative_rate")
        print(f"     F1={bert_m['f1']:.4f} Prec={bert_m['precision']:.4f} Rec={bert_m['recall']:.4f} AUC={bert_m.get('roc_auc',0):.4f}")
        all_results.append({"name":"DistilBERT","m":bert_m,"train_t":bert_tt,"infer_t":bert_it,
                             "preds":bert_p,"proba":bert_proba,"y_true":y_val,"X_val":X_val})
        fpr_c,tpr_c,_ = roc_curve(y_val, bert_proba)
        roc_data.append(("DistilBERT", fpr_c, tpr_c, bert_m["roc_auc"]))
        pr_c,rc_c,_ = precision_recall_curve(y_val, bert_proba)
        pr_data.append(("DistilBERT", pr_c, rc_c))

        print("\n  >> EnsembleClassifier (DistilBERT + TF-IDF)")
        ensemble = EnsembleClassifier()
        t0 = time.perf_counter(); ensemble.fit(X_train, y_train); ens_tt = time.perf_counter()-t0
        best_w = ensemble.tune_weights(X_val, y_val, metric="auc")
        print(f"     Tuned bert_weight={best_w:.2f} (optimizing AUC)")
        t0 = time.perf_counter(); ens_p = ensemble.predict(X_val); ens_it = time.perf_counter()-t0
        ens_proba = ensemble.predict_proba(X_val)[:,1]
        ens_m = evaluate(y_val, ens_p, ens_proba)
        ens_m["fpr"] = ens_m.pop("false_positive_rate")
        ens_m["fnr"] = ens_m.pop("false_negative_rate")
        print(f"     F1={ens_m['f1']:.4f} Prec={ens_m['precision']:.4f} Rec={ens_m['recall']:.4f} AUC={ens_m.get('roc_auc',0):.4f}")
        all_results.append({"name":"Ensemble","m":ens_m,"train_t":ens_tt,"infer_t":ens_it,
                             "preds":ens_p,"proba":ens_proba,"y_true":y_val,"X_val":X_val})
        fpr_c,tpr_c,_ = roc_curve(y_val, ens_proba)
        roc_data.append(("Ensemble", fpr_c, tpr_c, ens_m["roc_auc"]))
        pr_c,rc_c,_ = precision_recall_curve(y_val, ens_proba)
        pr_data.append(("Ensemble", pr_c, rc_c))

    # ── 3. Comparison plots ───────────────────────────────────────────────────
    step += 1
    print(f"\n[{step}/{total_steps}] Comparison plots")
    plot_metrics_comparison(all_results)
    plot_confusion_matrices(all_results)
    plot_roc(roc_data)
    plot_pr(pr_data)
    plot_timing(all_results)

    # ── 4. Threshold optimization ─────────────────────────────────────────────
    step += 1
    print(f"\n[{step}/{total_steps}] Threshold optimization")
    threshold_results = {}
    for r in all_results:
        if "proba" not in r:
            continue
        opt = find_optimal_threshold(y_val, r["proba"], metric="f1")
        y_pred_opt = apply_threshold(r["proba"], opt["threshold"])
        f1_opt = f1_score(y_val, y_pred_opt, zero_division=0)
        threshold_results[r["name"]] = {
            "threshold": opt["threshold"],
            "default_f1": r["m"]["f1"],
            "optimized_f1": f1_opt,
        }
        print(f"  {r['name']}: default F1={r['m']['f1']:.4f} -> optimized F1={f1_opt:.4f} (threshold={opt['threshold']:.3f})")
    plot_threshold(y_val, tfidf_proba, "TF-IDF+LR", "threshold_tfidf.pdf")
    if bert_proba is not None:
        plot_threshold(y_val, bert_proba, "DistilBERT", "threshold_distilbert.pdf")
    if threshold_results:
        plot_threshold_optimization(threshold_results)

    # ── 5. Training loss ──────────────────────────────────────────────────────
    step += 1
    print(f"\n[{step}/{total_steps}] Training loss")
    if bert is not None:
        plot_loss(bert.train_losses, "DistilBERT")

    # ── 6. Error analysis ─────────────────────────────────────────────────────
    step += 1
    print(f"\n[{step}/{total_steps}] Error analysis")
    for r in all_results:
        plot_error_analysis(r)

    fp_fn = {}
    for r in all_results:
        fp_idx = np.where((r["preds"]==1)&(y_val==0))[0][:5]
        fn_idx = np.where((r["preds"]==0)&(y_val==1))[0][:5]
        fp_fn[r["name"]] = {
            "false_positives": [X_val[i][:120] for i in fp_idx],
            "false_negatives": [X_val[i][:120] for i in fn_idx],
        }

    # ── 7. Hyperparameter sweep ───────────────────────────────────────────────
    step += 1
    print(f"\n[{step}/{total_steps}] Hyperparameter sweep")
    C_vals = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    C_f1 = []
    for c in C_vals:
        m = TFIDFLogisticRegressionClassifier(C=c); m.fit(X_train, y_train)
        C_f1.append(f1_score(y_val, m.predict(X_val), zero_division=0))
    feat_vals = [1000, 5000, 10000, 20000, 50000]
    feat_f1 = []
    for f in feat_vals:
        m = TFIDFLogisticRegressionClassifier(max_features=f); m.fit(X_train, y_train)
        feat_f1.append(f1_score(y_val, m.predict(X_val), zero_division=0))
    plot_hyperparam({"C_vals":C_vals,"C_f1":C_f1,"feat_vals":feat_vals,"feat_f1":feat_f1})
    print(f"  Best C={C_vals[np.argmax(C_f1)]}, F1={max(C_f1):.4f}")

    # ── 8. Data size ablation ─────────────────────────────────────────────────
    step += 1
    print(f"\n[{step}/{total_steps}] Data size ablation")
    sizes = [200, 500, 1000, 2000, 4000, 6000, len(X_train)]
    abl = {"Rule-based":{"sizes":[],"f1":[]}, "TF-IDF+LR":{"sizes":[],"f1":[]}}
    for s in sizes:
        if s > len(X_train): continue
        Xs, ys = X_train[:s], y_train[:s]
        if len(np.unique(ys)) < 2: continue
        rm = RuleBasedClassifier(); rm.fit(Xs, ys)
        abl["Rule-based"]["sizes"].append(s); abl["Rule-based"]["f1"].append(f1_score(y_val, rm.predict(X_val), zero_division=0))
        tm = TFIDFLogisticRegressionClassifier(); tm.fit(Xs, ys)
        abl["TF-IDF+LR"]["sizes"].append(s); abl["TF-IDF+LR"]["f1"].append(f1_score(y_val, tm.predict(X_val), zero_division=0))
    plot_data_ablation(abl)

    # ── 9. Data augmentation ablation ─────────────────────────────────────────
    step += 1
    print(f"\n[{step}/{total_steps}] Data augmentation ablation")
    aug_results = {}

    X_aug, y_aug = augment_dataset(X_train, y_train, target_label=1, n_aug_per_sample=4, seed=SEED)
    print(f"  Original train: {len(X_train)} -> Augmented: {len(X_aug)}")
    print(f"  Toxic ratio: {y_train.mean():.3f} -> {y_aug.mean():.3f}")

    tfidf_base = TFIDFLogisticRegressionClassifier()
    tfidf_base.fit(X_train, y_train)
    f1_base = f1_score(y_val, tfidf_base.predict(X_val), zero_division=0)

    tfidf_aug = TFIDFLogisticRegressionClassifier()
    tfidf_aug.fit(X_aug, y_aug)
    f1_aug = f1_score(y_val, tfidf_aug.predict(X_val), zero_division=0)

    aug_results["TF-IDF+LR"] = {"baseline": f1_base, "augmented": f1_aug}
    print(f"  TF-IDF+LR: F1 baseline={f1_base:.4f} -> augmented={f1_aug:.4f}")

    rule_base = RuleBasedClassifier()
    rule_base.fit(X_train, y_train)
    f1_rb = f1_score(y_val, rule_base.predict(X_val), zero_division=0)
    aug_results["Rule-based"] = {"baseline": f1_rb, "augmented": f1_rb}

    plot_augmentation_comparison(aug_results)

    # ── 10. Adversarial robustness ────────────────────────────────────────────
    step += 1
    print(f"\n[{step}/{total_steps}] Adversarial robustness evaluation")
    adv_results = {}

    print("  >> Rule-based adversarial eval")
    adv_results["Rule-based"] = adversarial_evaluate(rule, X_val, y_val, seed=SEED)

    print("  >> TF-IDF+LR adversarial eval")
    adv_results["TF-IDF+LR"] = adversarial_evaluate(tfidf, X_val, y_val, seed=SEED)

    if bert is not None:
        print("  >> DistilBERT adversarial eval")
        adv_results["DistilBERT"] = adversarial_evaluate(bert, X_val, y_val, seed=SEED)

    for model_name, res in adv_results.items():
        print(f"\n  {model_name}:")
        print(f"    Clean F1: {res['clean']['f1']:.4f}")
        for attack, m in res.items():
            if attack == "clean": continue
            print(f"    {attack}: F1={m['f1']:.4f} (drop={m['f1_drop']:.4f}, flip={m['flip_rate']:.3f})")

    plot_adversarial(adv_results)

    # ── 11. Bias analysis ─────────────────────────────────────────────────────
    step += 1
    print(f"\n[{step}/{total_steps}] Bias analysis")

    best_model = tfidf if bert is None else bert
    best_name = "TF-IDF+LR" if bert is None else "DistilBERT"
    best_preds = tfidf_p if bert is None else bert_p

    subgroup_results = compute_subgroup_metrics(X_val, y_val, best_preds, min_samples=5)
    bias_metrics = compute_bias_metrics(subgroup_results)
    print(format_bias_report(subgroup_results, bias_metrics))
    plot_bias(subgroup_results)

    # ── 12. Cross-domain evaluation ───────────────────────────────────────────
    cd_results = {}
    if not args.skip_cross_domain:
        step += 1
        print(f"\n[{step}/{total_steps}] Cross-domain evaluation")
        try:
            from src.cross_domain import load_civil_comments, cross_domain_evaluate
            civil_data = load_civil_comments(n_samples=min(5000, args.n_samples))
            print(f"  Civil Comments — Train: {len(civil_data['X_train'])}, Val: {len(civil_data['X_val'])}")

            rule_cd = RuleBasedClassifier()
            cd_m = cross_domain_evaluate(rule_cd, data, civil_data, has_proba=False)
            cd_results["Rule-based"] = {
                "in_domain_f1": rule_m["f1"],
                "cross_domain_f1": cd_m["f1"],
            }
            print(f"  Rule-based: in-domain F1={rule_m['f1']:.4f}, cross-domain F1={cd_m['f1']:.4f}")

            tfidf_cd = TFIDFLogisticRegressionClassifier()
            cd_m = cross_domain_evaluate(tfidf_cd, data, civil_data, has_proba=True)
            cd_results["TF-IDF+LR"] = {
                "in_domain_f1": tfidf_m["f1"],
                "cross_domain_f1": cd_m["f1"],
            }
            print(f"  TF-IDF+LR: in-domain F1={tfidf_m['f1']:.4f}, cross-domain F1={cd_m['f1']:.4f}")

            if cd_results:
                plot_cross_domain(cd_results)

        except Exception as e:
            print(f"  Cross-domain eval skipped: {e}")

    # ── Save all results ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    rj = []
    for r in all_results:
        entry = {"name":r["name"], "train_time":round(r["train_t"],4), "infer_time":round(r["infer_t"],6),
                 "metrics":{k:round(float(v),4) if isinstance(v,(int,float,np.floating)) else v
                            for k,v in r["m"].items()}}
        rj.append(entry)
        print(f"\n{r['name']}: F1={r['m']['f1']:.4f} Prec={r['m']['precision']:.4f} "
              f"Rec={r['m']['recall']:.4f} FPR={r['m']['fpr']:.4f} FNR={r['m']['fnr']:.4f} "
              f"AUC={r['m'].get('roc_auc','N/A')}")
        print(f"  Train: {r['train_t']:.2f}s  Inference: {r['infer_t']:.4f}s")

    adv_serializable = {}
    for model_name, res in adv_results.items():
        adv_serializable[model_name] = {}
        for attack, m in res.items():
            adv_serializable[model_name][attack] = {
                k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
                for k, v in m.items()
            }

    bias_serializable = {}
    for cat, groups in subgroup_results.items():
        bias_serializable[cat] = {}
        for grp, m in groups.items():
            bias_serializable[cat][grp] = {
                k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
                for k, v in m.items()
            }

    out = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(out, "w") as f:
        json.dump({
            "results": rj,
            "threshold_optimization": {
                k: {kk: round(float(vv), 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                for k, v in threshold_results.items()
            },
            "hyperparam": {
                "C_vals": C_vals,
                "C_f1": [round(x, 4) for x in C_f1],
                "feat_vals": feat_vals,
                "feat_f1": [round(x, 4) for x in feat_f1],
            },
            "ablation": {
                k: {"sizes": v["sizes"], "f1": [round(x, 4) for x in v["f1"]]}
                for k, v in abl.items()
            },
            "augmentation": {
                k: {kk: round(vv, 4) for kk, vv in v.items()}
                for k, v in aug_results.items()
            },
            "adversarial": adv_serializable,
            "bias": {
                "subgroups": bias_serializable,
                "aggregate": {k: round(float(v), 4) if isinstance(v, (int, float)) else v for k, v in bias_metrics.items()},
            },
            "cross_domain": {
                k: {kk: round(vv, 4) for kk, vv in v.items()}
                for k, v in cd_results.items()
            } if cd_results else {},
            "fp_fn": fp_fn,
        }, f, indent=2)
    print(f"\nAll results saved to {out}")


if __name__ == "__main__":
    main()
