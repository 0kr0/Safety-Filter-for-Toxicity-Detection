# Safety Filter for Toxicity Detection

Lightweight toxicity classifier for detecting harmful user content. Project for Innopolis GenAI course.

## Baselines

1. **Rule-based**: Keyword matching with bad words list
2. **TF-IDF + Logistic Regression**: Fast, interpretable, CPU-friendly
3. **DistilBERT**: Small transformer for better accuracy
4. **Ensemble** *(new)*: Weighted combination of DistilBERT + TF-IDF+LR

## Setup

```bash
pip install -r requirements.txt
```

Required packages: `numpy`, `pandas`, `scikit-learn`, `transformers`, `torch`, `datasets`, `matplotlib`, `seaborn`.

---

## Quick Start

### Option A — Full experiment suite (recommended for the report)

Generates all results, metrics, and visualizations in one run:

```bash
python experiments/run_all.py
```

This runs a 12-step pipeline and saves everything to `results/experiment_results.json` and `report/figures/`.

### Option B — Modular CLI

```bash
python -m src.main --hf --optimize-threshold --augment --adversarial --bias
```

---

## How to Reproduce All Results for the Report

Below is a step-by-step guide for generating every figure and table you need.

### Step 1 — Baseline comparison (Section 4.1 of the report)

The full experiment script trains all four models and produces:

```bash
# With DistilBERT + Ensemble (GPU recommended, ~30 min on CPU)
python experiments/run_all.py

# CPU-only / fast run (skip DistilBERT and Ensemble)
python experiments/run_all.py --skip-neural
```

**Figures generated:**
| File | Report section | What it shows |
|------|---------------|---------------|
| `metrics_comparison.pdf` | Table 1 / Figure 2 | Precision, Recall, F1, FPR, FNR bars for all models |
| `confusion_matrices.pdf` | Section 4.2 | Confusion matrix heatmaps per model |
| `roc_curves.pdf` | Section 4.3a | ROC curves with AUC values |
| `pr_curves.pdf` | Section 4.3b | Precision-Recall curves |
| `timing_comparison.pdf` | Table 2 | Training and inference time comparison |
| `class_distribution.pdf` | Figure 1 | Train/val class imbalance visualization |

### Step 2 — Threshold optimization (Section 6, improvement #2)

Automatically included in `run_all.py`. Shows that the default 0.5 threshold is suboptimal for 8% toxic prevalence.

**Figures generated:**
| File | What it shows |
|------|---------------|
| `threshold_tfidf.pdf` | Precision/Recall/F1/FPR vs threshold for TF-IDF+LR |
| `threshold_distilbert.pdf` | Same for DistilBERT |
| `threshold_optimization.pdf` | Before/after F1 comparison with optimal thresholds |

**Key result to report:** The optimized threshold and the F1 improvement over default 0.5.

### Step 3 — Data augmentation ablation (Section 6, improvement #3)

Automatically included. Augments the minority (toxic) class using EDA (synonym replacement, random insertion/swap/deletion) and measures the effect.

**Figure generated:**
| File | What it shows |
|------|---------------|
| `augmentation_comparison.pdf` | F1 with vs without EDA augmentation |

**Key result to report:** Whether augmenting the toxic class improves recall and F1.

### Step 4 — Adversarial robustness (Section 6, improvement #4)

Automatically included. Tests each model against 5 attack types:
- **Leetspeak**: `a→@`, `e→3`, `s→$`, etc.
- **Character substitution**: keyboard-neighbor typos
- **Space insertion**: `idiot → i d i o t`
- **Homoglyph attack**: Cyrillic look-alike characters
- **Character repetition**: `stupid → stuuupid`

**Figure generated:**
| File | What it shows |
|------|---------------|
| `adversarial_robustness.pdf` | F1 under each attack + prediction flip rates |

**Key result to report:** Which model is most robust, which attack causes the biggest F1 drop.

### Step 5 — Cross-domain testing (Section 6, improvement #5)

Train on Toxic Conversations, evaluate on Civil Comments (different annotation style, different distribution).

```bash
# Included by default; skip if download fails:
python experiments/run_all.py --skip-cross-domain
```

**Figure generated:**
| File | What it shows |
|------|---------------|
| `cross_domain_comparison.pdf` | In-domain vs cross-domain F1 per model |

**Key result to report:** How much F1 drops when evaluating on a different dataset.

### Step 6 — Ensemble results (Section 6, improvement #6)

Automatically included when running with neural models. The ensemble combines DistilBERT (70% weight) and TF-IDF+LR (30% weight) probabilities.

The ensemble appears in all comparison figures alongside the three baselines.

**Key result to report:** Whether the ensemble improves over DistilBERT alone, and inference speed trade-off.

### Step 7 — Bias analysis (Section 6, improvement #7)

Automatically included. Checks if the classifier disproportionately flags comments mentioning certain identity groups (gender, religion, race/ethnicity, sexuality).

**Figure generated:**
| File | What it shows |
|------|---------------|
| `bias_analysis.pdf` | Per-subgroup FPR and FNR bar charts |

**Key result to report:** FPR/FNR gaps across subgroups, aggregate bias score.

### Step 8 — Hyperparameters and ablations (Sections 5.2–5.3)

Automatically included:

**Figures generated:**
| File | What it shows |
|------|---------------|
| `hyperparam_sweep.pdf` | TF-IDF+LR F1 vs regularization C and vocabulary size |
| `data_size_ablation.pdf` | F1 vs training set size (learning curves) |
| `distilbert_training_loss.pdf` | DistilBERT loss curve over training steps |

### Step 9 — Error analysis (Section 5.1)

**Figures generated (one per model):**
| File | What it shows |
|------|---------------|
| `error_analysis_rule-based.pdf` | Text length distribution by error type + prediction breakdown |
| `error_analysis_tf-idf__lr.pdf` | Same for TF-IDF+LR |
| `error_analysis_distilbert.pdf` | Same for DistilBERT |

---

## Scaling to the Full Dataset (Section 6, improvement #1)

To train on the full HuggingFace dataset instead of 10K samples:

```bash
# Full Toxic Conversations dataset (~50K samples)
python experiments/run_all.py --full-dataset

# Or a custom size
python experiments/run_all.py --n-samples 30000
```

To use the full Jigsaw dataset (1.7M samples), download from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), place `train.csv` in `data/`, and run:

```bash
python -m src.main
```

---

## All CLI Options

```bash
python -m src.main \
  --hf                    # Use HuggingFace dataset (no Kaggle download needed)
  --full-dataset          # Use entire HF dataset instead of sampling
  --n-samples 20000       # Custom sample count
  --baselines rule,tfidf,distilbert,ensemble
  --skip-neural           # Skip DistilBERT + Ensemble
  --optimize-threshold    # Find F1-optimal threshold per model
  --augment               # Augment minority class with EDA before training
  --adversarial           # Run adversarial robustness evaluation
  --bias                  # Run bias analysis across identity subgroups
  --cross-domain          # Train on Toxic Conv., evaluate on Civil Comments
  --output results.json   # Custom output path
  --demo                  # Use tiny synthetic data (for testing only)
```

---

## Output Locations

| Path | Contents |
|------|----------|
| `results/experiment_results.json` | All metrics, thresholds, adversarial results, bias scores, cross-domain |
| `results/baseline_results.json` | Basic metrics from `src.main` |
| `report/figures/*.pdf` | All generated plots (from `experiments/run_all.py`) |

---

## Project Structure

```
src/
  config.py              # Paths, constants, random seed
  data.py                # Data loading (Jigsaw, HuggingFace, demo)
  evaluation.py          # Metrics + run_baseline with threshold optimization
  main.py                # CLI pipeline
  threshold_optimizer.py  # F1-optimal threshold search
  adversarial.py         # Adversarial attack transforms + evaluation
  augmentation.py        # EDA data augmentation (synonym replace, etc.)
  bias_analysis.py       # Per-subgroup bias metrics (FPR/FNR gaps)
  cross_domain.py        # Cross-domain evaluation (Civil Comments)
  baselines/
    rule_based.py        # Baseline 1: keyword matching
    tfidf_lr.py          # Baseline 2: TF-IDF + Logistic Regression
    distilbert_clf.py    # Baseline 3: DistilBERT fine-tuning
    ensemble.py          # Baseline 4: weighted ensemble (BERT + TF-IDF)
experiments/
  run_all.py             # Full experiment suite (generates all figures)
data/                    # Place train.csv here (Jigsaw dataset)
models/                  # Saved models (auto-created)
results/                 # JSON results
report/figures/          # Generated plots
```

---

## Summary of Figures for the Report

Running `python experiments/run_all.py` generates **17+ PDF figures** in `report/figures/`:

| # | Figure | Report section |
|---|--------|----------------|
| 1 | `class_distribution.pdf` | Dataset overview |
| 2 | `metrics_comparison.pdf` | Baseline comparison (now with 4 models) |
| 3 | `confusion_matrices.pdf` | Per-model confusion matrices |
| 4 | `roc_curves.pdf` | ROC curves with AUC |
| 5 | `pr_curves.pdf` | Precision-Recall curves |
| 6 | `timing_comparison.pdf` | Training + inference speed |
| 7 | `threshold_tfidf.pdf` | Threshold analysis: TF-IDF+LR |
| 8 | `threshold_distilbert.pdf` | Threshold analysis: DistilBERT |
| 9 | `threshold_optimization.pdf` | F1 before/after threshold optimization |
| 10 | `distilbert_training_loss.pdf` | Training dynamics |
| 11 | `error_analysis_*.pdf` (×3–4) | Error analysis per model |
| 12 | `hyperparam_sweep.pdf` | Regularization + vocabulary sensitivity |
| 13 | `data_size_ablation.pdf` | Learning curves |
| 14 | `augmentation_comparison.pdf` | Effect of data augmentation |
| 15 | `adversarial_robustness.pdf` | Robustness to adversarial attacks |
| 16 | `bias_analysis.pdf` | Per-subgroup FPR/FNR |
| 17 | `cross_domain_comparison.pdf` | In-domain vs cross-domain generalization |

## Authors

Nikita Shiyanov, Anton Korotkov — Innopolis University
