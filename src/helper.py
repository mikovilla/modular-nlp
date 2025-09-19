import pandas as pd
import json

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
