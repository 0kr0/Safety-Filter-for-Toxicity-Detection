import sys, os, json, time, warnings
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

from src.baselines import RuleBasedClassifier, TFIDFLogisticRegressionClassifier, DistilBERTClassifier
from src.evaluation import evaluate, measure_inference_time
from src.data import load_hf_toxic

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
    fig, ax = plt.subplots(figsize=(10, 5))
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
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
    if len(results) == 1: axes = [axes]
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


def main():
    print("toxicity detection — experiments (real data)")

    print("\n[1/8] Loading data via src.data.load_hf_toxic")
    data = load_hf_toxic(n_samples=10000)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"  Toxic ratio — train: {y_train.mean():.3f}, val: {y_val.mean():.3f}")
    plot_class_dist(y_train, y_val)

    print("\n[2/8] Training baselines (from src.baselines)")
    all_results = []; roc_data = []; pr_data = []

    print("\n  >> RuleBasedClassifier (src/baselines/rule_based.py)")
    rule = RuleBasedClassifier()
    t0 = time.perf_counter(); rule.fit(X_train, y_train); rule_tt = time.perf_counter()-t0
    t0 = time.perf_counter(); rule_p = rule.predict(X_val); rule_it = time.perf_counter()-t0
    rule_m = evaluate(y_val, rule_p)
    rule_m["fpr"] = rule_m.pop("false_positive_rate")
    rule_m["fnr"] = rule_m.pop("false_negative_rate")
    print(f"     F1={rule_m['f1']:.4f} Prec={rule_m['precision']:.4f} Rec={rule_m['recall']:.4f}")
    all_results.append({"name":"Rule-based","m":rule_m,"train_t":rule_tt,"infer_t":rule_it,
                         "preds":rule_p,"y_true":y_val,"X_val":X_val})

    print("\n  >> TFIDFLogisticRegressionClassifier (src/baselines/tfidf_lr.py)")
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

    print("\n  >> DistilBERTClassifier (src/baselines/distilbert_clf.py)")
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

    print("\n[3/8] Comparison plots")
    plot_metrics_comparison(all_results)
    plot_confusion_matrices(all_results)
    plot_roc(roc_data)
    plot_pr(pr_data)
    plot_timing(all_results)

    print("\n[4/8] Threshold analysis")
    plot_threshold(y_val, tfidf_proba, "TF-IDF+LR", "threshold_tfidf.pdf")
    plot_threshold(y_val, bert_proba, "DistilBERT", "threshold_distilbert.pdf")

    print("\n[5/8] Training loss")
    plot_loss(bert.train_losses, "DistilBERT")

    print("\n[6/8] Error analysis")
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

    print("\n[7/8] Hyperparameter sweep")
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

    print("\n[8/8] Data size ablation")
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

    print("Results")
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

    out = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(out, "w") as f:
        json.dump({"results":rj,
                    "hyperparam":{"C_vals":C_vals,"C_f1":[round(x,4) for x in C_f1],
                                  "feat_vals":feat_vals,"feat_f1":[round(x,4) for x in feat_f1]},
                    "ablation":{k:{"sizes":v["sizes"],"f1":[round(x,4) for x in v["f1"]]} for k,v in abl.items()},
                    "fp_fn":fp_fn}, f, indent=2)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
