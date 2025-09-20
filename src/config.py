import torch

from dataclasses import dataclass

class SharedConfig:
    TEXT_COL = "text"
    LABEL_COL = "label"
    MAX_LENGTH = 128 # tokens after tokenization (increase for longer sentiments)
    SEED = 42 # randomization for reproducibility
    USE_FP16 = torch.cuda.is_available() # Use GPU
    BATCH_SIZE = 8 # dataset/64 to maximize 16G VRAM of effective batching
    EPOCHS = 5
    LR = 2e-5 # Range: 1e-5 to 5e-5, higher is faster but can cause instability
    WEIGHT_DECAY = 0.01 # BERT default to reduce overfitting, increasing shrinks the model's weight and may cause underfitting
    WARMUP_RATIO = 0 # 0.06 Linearly scale with LR
    GRAD_ACCUM_STEPS = 1 # BATCH_SIZE * GRAD_ACCUM_STEPS
    EVAL_STEPS = 0  # set > 0 for periodic eval, 0 for epoch
    SAVE_TOTAL_LIMIT = 2 # increase for more rollback points

class MBertConfig:
    MODEL_NAME = "bert-base-multilingual-cased"
    OUTPUT_DIR = "./mbert_sentiment"
    USE_SAVED_MODEL = 0

class XlmrConfig:
    MODEL_NAME = "xlm-roberta-base"
    OUTPUT_DIR = "./xlmr_sentiment"
    USE_SAVED_MODEL = 1

class MambaConfig:
    MODEL_NAME = "state-spaces/mamba-130m-hf"
    OUTPUT_DIR = "./mamba_sentiment"
    USE_SAVED_MODEL = 0

class AppConfig:
    SAVE_MODEL = True
    DATASET = "./miko.jsonl"
    CACHE_DIR = "./cache/translations_cache.jsonl"
    DEVICE = 0 if torch.cuda.is_available() else -1
    ENVIRONMENT = "sandbox"
    DEBUG = 0
    INFER_SAMPLE = 0

class TranslateConfig:
    MODEL = "Helsinki-NLP/opus-mt-tl-en" # requires sentencepiece
    CACHE_DIR = "./cache/translations_cache.jsonl"
    BATCH_SIZE = 64
    MAX_NEW_TOKENS = 128

__all__ = ["SharedConfig", "AppConfig", "TranslateConfig", "MBertConfig", "XlmrConfig", "MambaConfig"]