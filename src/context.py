import numpy as np
import os
import torch

from dataclasses import dataclass
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification, 
    AutoModel, 
    AutoTokenizer, 
    DataCollatorWithPadding
)
from typing import Type


from src import helper, utility, translator, mamba
from src.config import App, Data, Mamba

@dataclass
class Context:
    instance: Type
    model: object
    tokenizer: object
    train_ds: object
    val_ds: object
    test_ds: object
    data_collator: object
    class_weights: torch.Tensor

def setup_pipeline(instance_cls, require_translation: bool = False) -> Context:
    instance = instance_cls()
    jsonl = ""

    if App.DEBUG and Data.SHOW_ON_DEBUG:
        helper.print_header("original data")
        print(helper.read_jsonl_as_string(Path(Data.DATASET)))
    
    if not require_translation:
        jsonl = helper.read_jsonl_as_string(Path(Data.DATASET))
    else:
        jsonl = translator.from_jsonl(Data.DATASET)
        if App.DEBUG and Data.SHOW_ON_DEBUG:
            helper.print_header("translated data")
            print(translator.from_jsonl(Data.DATASET))

    train_ds, val_ds, test_ds, label2id, id2label = utility.load_split_dataset(jsonl)
    num_labels = len(id2label)

    model = None
    tokenizer = None
    if isinstance(instance, Mamba):
        load_res = mamba.load_model_for_classification(
            model_name=instance.MODEL_NAME,
            num_labels=num_labels,
            id2label={i: str(i) for i in range(num_labels)} if not isinstance(id2label, dict) else id2label,
            label2id={str(v): k for k, v in ({v: k for k, v in id2label.items()}).items()} if isinstance(id2label, dict) else None
        )
        model = load_res.model
        tokenizer = load_res.tokenizer
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
                instance.MODEL_NAME,
                num_labels=num_labels,
                id2label={i: str(i) for i in range(num_labels)} if not isinstance(id2label, dict) else id2label,
                label2id={str(v): k for k, v in ({v: k for k, v in id2label.items()}).items()} if isinstance(id2label, dict) else None,
                problem_type="single_label_classification",
            )
        tokenizer = utility.make_tokenizer(instance_cls)
    
    tok_fn = utility.make_tokenizer_fn(tokenizer)
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=[Data.TEXT_COL])
    val_ds   = val_ds.map(tok_fn, batched=True, remove_columns=[Data.TEXT_COL])
    test_ds  = test_ds.map(tok_fn, batched=True, remove_columns=[Data.TEXT_COL])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    y_train = np.array(train_ds["label"])
    counts = np.bincount(y_train, minlength=num_labels).astype(np.float32)
    weights = counts.sum() / (counts + 1e-9)
    class_weights = torch.tensor(weights / weights.mean())
    
    return Context(
        instance=instance,
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        data_collator=data_collator,
        class_weights=class_weights
    )