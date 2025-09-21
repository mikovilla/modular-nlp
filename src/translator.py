import io
import json
import os
import sys

from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
from typing import (
    List, 
    Dict, 
    Any, 
    Union
)

from src import helper
from src.config import *

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def load_translation_cache(path: Path) -> Dict[str, str]:
    cache = {}
    if path is None or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                txt = obj.get("text")
                trn = obj.get("translation_en")
                if isinstance(txt, str) and isinstance(trn, str):
                    cache[txt] = trn
            except Exception:
                continue
    return cache

def append_translation_cache(path: Path, new_items: Dict[str, str]) -> None:
    if path is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    with path.open("a", encoding="utf-8") as f:
        for txt, trn in new_items.items():
            f.write(json.dumps({"text": txt, "translation_en": trn}, ensure_ascii=False) + "\n")

def raw_translate(texts: List[str], translator, max_new_tokens: int = 256) -> List[str]:
    outs = translator(texts, truncation=True, max_new_tokens=max_new_tokens)
    if isinstance(outs, dict):
        outs = [outs]
    return [o["translation_text"] for o in outs]

def batched(iterable, n: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def from_jsonl(dataset: Union[str, Path] = AppConfig.DATASET) -> str:
    dataset = Path(dataset)
    cache_path = Path(getattr(TranslateConfig, "CACHE_FILE", "cache/translations_cache.jsonl"))
    cache = load_translation_cache(cache_path)

    translator = pipeline(
        "translation",
        model=TranslateConfig.MODEL,
        device=AppConfig.DEVICE,
    )

    rows = read_jsonl(dataset)
    print(f"[INFO] Loaded {len(rows)} rows")

    outputs: List[Dict[str, Any]] = []
    new_cache_accumulator: Dict[str, str] = {}

    for batch in tqdm(list(batched(rows, TranslateConfig.BATCH_SIZE)), desc="Processing"):
        texts = [r.get("text", "") for r in batch]

        translations: List[Any] = []
        to_translate_idx: List[int] = []
        to_translate_texts: List[str] = []
        for i, t in enumerate(texts):
            if t in cache:
                translations.append(cache[t])
            else:
                translations.append(None)
                to_translate_idx.append(i)
                to_translate_texts.append(t)

        if to_translate_texts:
            try:
                new_tr = raw_translate(
                    to_translate_texts, translator, max_new_tokens=TranslateConfig.MAX_NEW_TOKENS
                )
            except Exception as e:
                print(f"[WARN] Translation failed for a batch ({len(to_translate_texts)} items): {e}", file=sys.stderr)
                new_tr = to_translate_texts  # fallback: passthrough

            for i, tr in zip(to_translate_idx, new_tr):
                translations[i] = tr
                src = texts[i]
                cache[src] = tr
                new_cache_accumulator[src] = tr

            if new_cache_accumulator and len(new_cache_accumulator) >= 100:
                append_translation_cache(cache_path, new_cache_accumulator)
                new_cache_accumulator.clear()

        for r, tr in zip(batch, translations):
            outputs.append({
                "id": r.get("id"),
                "text": tr,
                "label": r.get("label"),
                "original_text": r.get("text"),
            })

    if new_cache_accumulator:
        append_translation_cache(cache_path, new_cache_accumulator)

    return helper.to_jsonl(outputs)