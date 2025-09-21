import os
import torch
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer, DataCollatorWithPadding
from pathlib import Path
from typing import Type
from dataclasses import dataclass

from src import helper, utility, translator, mamba
from src.config import AppConfig, SharedConfig, MambaConfig

@dataclass
class Context:
    modelConfig: Type
    model: object
    tokenizer: object
    train_ds: object
    val_ds: object
    test_ds: object
    data_collator: object
    class_weights: torch.Tensor

def setup_pipeline(configClass, require_translation: bool = False) -> Context:
    modelConfig = configClass()
    jsonl = ""

    if AppConfig.DEBUG and AppConfig.SHOW_DATA:
        helper.print_header("original data")
        print(helper.read_jsonl_as_string(Path(AppConfig.DATASET)))
    
    if not require_translation:
        jsonl = helper.read_jsonl_as_string(Path(AppConfig.DATASET))
    else:
        jsonl = translator.from_jsonl(AppConfig.DATASET)
        if AppConfig.DEBUG and AppConfig.SHOW_DATA:
            helper.print_header("translated data")
            print(translator.from_jsonl(AppConfig.DATASET))
        
    train_ds, val_ds, test_ds, label2id, id2label = utility.load_split_dataset(jsonl)
    num_labels = len(id2label)

    model = None
    tokenizer = None
    if isinstance(modelConfig, MambaConfig):
        load_res = mamba.load_model_for_classification(
            model_name=modelConfig.OUTPUT_DIR if modelConfig.USE_SAVED_MODEL else modelConfig.MODEL_NAME,
            backbone_name = modelConfig.MODEL_NAME,
            num_labels=num_labels,
            id2label={i: str(i) for i in range(num_labels)} if not isinstance(id2label, dict) else id2label,
            label2id={str(v): k for k, v in ({v: k for k, v in id2label.items()}).items()} if isinstance(id2label, dict) else None
        )
        model = load_res.model
        tokenizer = load_res.tokenizer
    else:
        if modelConfig.USE_SAVED_MODEL:
            model = AutoModelForSequenceClassification.from_pretrained(modelConfig.OUTPUT_DIR)
            tokenizer = AutoTokenizer.from_pretrained(modelConfig.OUTPUT_DIR)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                modelConfig.MODEL_NAME,
                num_labels=num_labels,
                id2label={i: str(i) for i in range(num_labels)} if not isinstance(id2label, dict) else id2label,
                label2id={str(v): k for k, v in ({v: k for k, v in id2label.items()}).items()} if isinstance(id2label, dict) else None,
                problem_type="single_label_classification",
            )
            tokenizer = utility.make_tokenizer(configClass)
    
    tok_fn = utility.make_tokenizer_fn(tokenizer)
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=[SharedConfig.TEXT_COL])
    val_ds   = val_ds.map(tok_fn, batched=True, remove_columns=[SharedConfig.TEXT_COL])
    test_ds  = test_ds.map(tok_fn, batched=True, remove_columns=[SharedConfig.TEXT_COL])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    y_train = np.array(train_ds["label"])
    counts = np.bincount(y_train, minlength=num_labels).astype(np.float32)
    weights = counts.sum() / (counts + 1e-9)
    class_weights = torch.tensor(weights / weights.mean())

    return Context(
        modelConfig=modelConfig,
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        data_collator=data_collator,
        class_weights=class_weights
    )