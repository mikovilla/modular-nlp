import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer

from src.config import SharedConfig

def build_label_maps(series: pd.Series):
    if series.dtype.kind in {"i", "u"}:
        classes = sorted(series.unique().tolist())
        label2id = {int(c): int(c) for c in classes}
        id2label = {int(c): str(c) for c in classes}
        return label2id, id2label, series.astype(int)
    else:
        classes = sorted(series.astype(str).unique().tolist())
        label2id = {c: i for i, c in enumerate(classes)}
        id2label = {i: c for c, i in label2id.items()}
        return label2id, id2label, series.astype(str).map(label2id)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }

# Expects a CSV with columns: text,label
def load_split_dataset(csv_path: str, test_size: float = 0.2, val_size: float = 0.1):
    df = pd.read_csv(csv_path)
    assert SharedConfig.TEXT_COL in df.columns and SharedConfig.LABEL_COL in df.columns, f"CSV must have '{SharedConfig.TEXT_COL}' and '{SharedConfig.LABEL_COL}' columns."

    label2id, id2label, y = build_label_maps(df[SharedConfig.LABEL_COL])
    df = df.assign(label_id=y)

    X_train, X_temp, y_train, y_temp = train_test_split(
        df[SharedConfig.TEXT_COL], df["label_id"], test_size=test_size + val_size, random_state=SharedConfig.SEED, stratify=df["label_id"]
    )
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - rel_val, random_state=SharedConfig.SEED, stratify=y_temp
    )

    def to_ds(X, y):
        return Dataset.from_pandas(pd.DataFrame({SharedConfig.TEXT_COL: X.values, "label": y.values}))

    return (
        to_ds(X_train, y_train),
        to_ds(X_val, y_val),
        to_ds(X_test, y_test),
        label2id,
        {k: v for k, v in enumerate({v: k for k, v in build_label_maps(df[SharedConfig.LABEL_COL])[0].items()})}  # id2label (clean)
    )

def make_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    def tok(batch):
        return tokenizer(
            batch[SharedConfig.TEXT_COL],
            truncation=True,
            max_length=SharedConfig.MAX_LENGTH,
            padding=False,
        )
    return tokenizer, tok