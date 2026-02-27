"""Configuration for toxicity detection baselines."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Jigsaw dataset filenames (place train.csv and test.csv in data/)
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
SAMPLE_SUBMISSION = "sample_submission.csv"

# Train/val split (use original test for final eval, split train into train/val)
VAL_RATIO = 0.1
RANDOM_SEED = 42

# Toxicity labels in Jigsaw dataset
TOXICITY_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# For binary classification: if ANY toxicity label is 1, text is toxic
TARGET_COL = "toxic"  # Can use single label or aggregate
