import inspect
import json
import os
import pandas as pd
import torch
import torch.optim as optim

from datasets import load_from_disk
from pathlib import Path

from src.config import *

def read_jsonl_as_string(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def to_jsonl(rows):
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)

def to_list_str(col):
    out = []
    for x in col:
        if x is None:
            out.append("")
        elif hasattr(x, "as_py"):
            out.append(str(x.as_py()))
        else:
            out.append(str(x))
    return out

def print_header(text):
    print(f"\r\n### {text.upper()} ###")

def list_config():
    if App.DEBUG:
        print("\r\n### CONFIGURATION: START ###")
        print("### DEFAULT TRAINING ARGUMENTS CONFIG ###")
        for name, value in vars(DefaultTrainingArguments).items():
            if not name.startswith("__"):
                print(f"\t{name}: {value}")
        print("### APPLICATION CONFIG ###")
        for name, value in vars(App).items():
            if not name.startswith("__"):
                print(f"\t{name}: {value}")
        print("### TRANSLATION CONFIG ###")
        for name, value in vars(Translation).items():
            if not name.startswith("__"):
                print(f"\t{name}: {value}")
        print("### DATA CONFIG ###")
        for name, value in vars(Data).items():
            if not name.startswith("__"):
                print(f"\t{name}: {value}")
        print("### MBERT CONFIG ###")
        for name, value in vars(MBert).items():
            if not name.startswith("__"):
                print(f"\t{name}: {value}")
        print("### XLM-R CONFIG ###")
        for name, value in vars(Xlmr).items():
            if not name.startswith("__"):
                print(f"\t{name}: {value}")
        print("### MAMBA CONFIG ###")
        for name, value in vars(Mamba).items():
            if not name.startswith("__"):
                print(f"\t{name}: {value}") 
        print("### CONFIGURATION: END ###")

def load_dataset_if_exists(folder, fallback, ignore: bool = False):
    if ignore:
            return fallback, False
    try:
        return load_from_disk(f"{Data.DATASET_SPLITS_DIR}/{folder}"), True
    except Exception:
        return fallback, False

from transformers import TrainingArguments

def to_args(cls, valid_keys, exclude):
    args = {}
    for name in dir(cls):
        if name.startswith("__"):
            continue
        if name in exclude:
            continue
            
        value = getattr(cls, name)
        if callable(value):
            continue

        key = name.lower()
        if key in valid_keys:
            args[key] = value
    return args

def to_training_args(obj_or_cls) -> TrainingArguments:
    cls = obj_or_cls if isinstance(obj_or_cls, type) else obj_or_cls.__class__
    exclude = set(getattr(cls, "EXCLUDE_KEYS", [])) | set(getattr(cls, "exclude_keys", []))
    valid_keys = set(TrainingArguments.__dataclass_fields__.keys())

    args = {}
    for name in dir(obj_or_cls):
        if name.startswith("__"):
            continue
        if name in exclude:
            continue
            
        value = getattr(obj_or_cls, name)
        if callable(value):
            continue

        key = name.lower()
        if key in valid_keys:
            args[key] = value

    if App.DEBUG:
        print(args)

    return TrainingArguments(**args)

def to_optimizer_args(obj_or_cls):
    cfg = obj_or_cls
    cls = obj_or_cls if isinstance(obj_or_cls, type) else obj_or_cls.__class__
    name = getattr(cfg, "OPTIMIZER_NAME", None) or getattr(cls, "OPTIMIZER_NAME", None)
    opt_cls = getattr(optim, name)
    sig = inspect.signature(opt_cls)        
    valid_keys = {name.lower() for name in sig.parameters if name != "params"}
    exclude = set(getattr(cls, "EXCLUDE_KEYS", [])) | set(getattr(cls, "exclude_keys", []))
    
    def harvest(obj):
        out = {}
        for name in dir(obj):
            if name.startswith("__"):
                continue
            if name in exclude:
                continue
            try:
                val = getattr(obj, name)
            except Exception:
                continue
            if callable(val):
                continue
            out[name.lower()] = val
        return out

    raw = harvest(cls)
    if not isinstance(obj_or_cls, type):
        raw.update(harvest(cfg))

    kwargs = {k: raw[k] for k in valid_keys if k in raw}
    return opt_cls, kwargs
    