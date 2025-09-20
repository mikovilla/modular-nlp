import pandas as pd
import json

from src.config import *
from pathlib import Path

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