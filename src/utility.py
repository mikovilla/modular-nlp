import io
import os
import pandas as pd

from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from src.config import DefaultTrainingArguments, Mamba, Data

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

def load_split_dataset(jsonl: str, test_size: float = 0.2, val_size: float = 0.1):
    df = pd.read_json(io.StringIO(jsonl), lines=True)

    label2id, id2label, y = build_label_maps(df[Data.LABEL_COL])
    df = df.assign(label_id=y)

    X_train, X_temp, y_train, y_temp = train_test_split(
        df[Data.TEXT_COL],
        df["label_id"],
        test_size=test_size + val_size,
        random_state=DefaultTrainingArguments.SEED,
        stratify=df["label_id"]
    )

    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - rel_val,
        random_state=DefaultTrainingArguments.SEED,
        stratify=y_temp
    )

    def to_ds(X, y):
        return Dataset.from_pandas(
            pd.DataFrame({Data.TEXT_COL: X.values, "label": y.values})
        )

    return (
        to_ds(X_train, y_train),
        to_ds(X_val, y_val),
        to_ds(X_test, y_test),
        label2id,
        id2label
    )

def make_tokenizer(instance_cls):
    instance = instance_cls()

    if os.path.exists(instance.OUTPUT_DIR):
        tokenizer = AutoTokenizer.from_pretrained(instance.OUTPUT_DIR)
    else: 
        if isinstance(instance, Mamba):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    instance.MODEL_NAME,
                    trust_remote_code=True,
                    use_fast=False,
                    local_files_only=False,
                )
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(
                    instance.MODEL_NAME,
                    trust_remote_code=True,
                    use_fast=True,
                    local_files_only=False,
                )
        
            if tokenizer.pad_token_id is None:
                if getattr(tokenizer, "eos_token", None) is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        else:
            tokenizer = AutoTokenizer.from_pretrained(instance.MODEL_NAME, use_fast=True)

    return tokenizer

def make_tokenizer_fn(tokenizer):
    def tok(batch):
        return tokenizer(
            batch[Data.TEXT_COL],
            truncation=True,
            max_length=Data.MAX_LENGTH,
            padding=False,
        )
    return tok
    