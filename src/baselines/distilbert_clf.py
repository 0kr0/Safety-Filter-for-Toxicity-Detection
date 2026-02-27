"""Baseline 3: DistilBERT - small transformer for better accuracy, still CPU-viable."""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


class DistilBERTClassifier:
    """
    DistilBERT-based toxicity classifier.
    Uses distilbert-base-uncased for speed; runs on CPU.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.trainer = None

    def _prepare_dataset(self, X, y=None):
        encodings = self.tokenizer(
            list(X),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        data = {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"]}
        if y is not None:
            data["labels"] = list(y)
        return Dataset.from_dict(data)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 2, batch_size: int = 16):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )

        train_dataset = self._prepare_dataset(X, y)

        training_args = TrainingArguments(
            output_dir="./models/distilbert_toxicity",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=50,
            evaluation_strategy="no",
            save_strategy="no",
            use_cpu=not torch.cuda.is_available(),
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )

        self.trainer.train()
        return self

    def predict(self, X) -> np.ndarray:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.atleast_1d(X)
        self.model.eval()
        preds_list = []
        batch_size = 32

        for i in range(0, len(X), batch_size):
            batch = list(X[i : i + batch_size])
            encodings = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self.model(**encodings)
                preds = torch.argmax(outputs.logits, dim=1)
                preds_list.extend(preds.cpu().numpy())

        return np.array(preds_list, dtype=int)
