import json
import os
import pandas as pd

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
    if AppConfig.DEBUG:
        print("\r\n### CONFIGURATION: START ###")
        print("### SHARED CONFIG ###")
        for name, value in vars(SharedConfig).items():
            if not name.startswith("__"):
                print(f"\t{name}: {value}")
        print("### APPLICATION CONFIG ###")
        for name, value in vars(AppConfig).items():
            if not name.startswith("__"):
                print(f"\t{name}: {value}")
        print("### TRANSLATION CONFIG ###")
        for name, value in vars(TranslateConfig).items():
            if not name.startswith("__"):
                print(f"\t{name}: {value}")
        print("### MBERT CONFIG ###")
        for name, value in vars(MBertConfig).items():
            if not name.startswith("__"):
                print(f"\t{name}: {value}")
        print("### XLM-R CONFIG ###")
        for name, value in vars(XlmrConfig).items():
            if not name.startswith("__"):
                print(f"\t{name}: {value}")
        print("### MAMBA CONFIG ###")
        for name, value in vars(MambaConfig).items():
            if not name.startswith("__"):
                print(f"\t{name}: {value}") 
        print("### CONFIGURATION: END ###")



def load_dataset_if_exists(folder, fallback, ignore: bool = False):
    if ignore:
            return fallback, False
    try:
        return load_from_disk(f"{AppConfig.DATASET_SPLITS_DIR}/{folder}"), True
    except Exception:
        return fallback, False