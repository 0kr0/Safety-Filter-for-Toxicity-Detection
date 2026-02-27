# Safety Filter for Toxicity Detection

Lightweight toxicity classifier for detecting harmful user content. Project for Innopolis GenAI course (Week 3 proposal).

## Baselines

1. **Rule-based**: Keyword matching with bad words list
2. **TF-IDF + Logistic Regression**: Fast, interpretable, CPU-friendly
3. **DistilBERT**: Small transformer for better accuracy

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset from Kaggle.

Place `train.csv` (and optionally `test.csv`) in the `data/` folder:

```
data/
  train.csv
  test.csv
  bad_words.txt  # for rule-based baseline
```

## Run

```bash
# All baselines (from project root)
python -m src.main

# Only rule + TF-IDF (skip DistilBERT)
python -m src.main --skip-neural

# Specific baselines
python -m src.main --baselines rule,tfidf

# Custom data path
python -m src.main --data-dir /path/to/data
```

Results are saved to `results/baseline_results.json`.

## Evaluation Metrics

- Precision, Recall, F1
- False positive rate (minimize to avoid blocking normal users)
- False negative rate (minimize to catch toxic content)
- ROC-AUC
- Inference time

## Project Structure

```
src/
  config.py       # Configuration
  data.py         # Data loading and preprocessing
  evaluation.py   # Metrics
  main.py         # Main pipeline
  baselines/
    rule_based.py
    tfidf_lr.py
    distilbert_clf.py
data/             # Place train.csv here
models/           # Saved models (optional)
results/          # Evaluation results
```

## Authors

Nikita Shiyanov, Anton Korotkov — Innopolis University
