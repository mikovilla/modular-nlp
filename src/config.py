import torch
import os

from dataclasses import dataclass

@dataclass
class DefaultTrainingArguments:
    EVAL_STRATEGY = "epoch"
    SAVE_STRATEGY = "epoch"
    LOGGING_STRATEGY = "steps"
    LOGGING_STEPS = 50
    PER_DEVICE_TRAIN_BATCH_SIZE = 32
    PER_DEVICE_EVAL_BATCH_SIZE = 64
    NUM_TRAIN_EPOCHS = 5
    FP16 = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    BF16 = torch.cuda.is_bf16_supported()
    LEARNING_RATE = 2e-5
    GRADIENT_ACCUMULATION_STEPS = 2
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    WARMUP_RATIO = 0.06
    LR_SCHEDULER_TYPE = "linear"
    DATALOADER_NUM_WORKERS = 1
    SAVE_TOTAL_LIMIT = 3
    LOAD_BEST_MODEL_AT_END = True
    METRIC_FOR_BEST_MODEL = "eval_f1_macro"
    GREATER_IS_BETTER = True
    SEED = 42
    EXCLUDE_KEYS = [ "EXCLUDE_KEYS", "MODEL_NAME", "TEMP", "WEIGHT", "NAME" ]

@dataclass
class MBert(DefaultTrainingArguments):
    NAME = "mBERT"
    MODEL_NAME = "bert-base-multilingual-cased"
    OUTPUT_DIR = "./mbert_sentiment"
    PER_DEVICE_TRAIN_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 2
    NUM_TRAIN_EPOCHS = 4

@dataclass
class Xlmr(DefaultTrainingArguments):
    NAME = "XLM-R"
    MODEL_NAME = "xlm-roberta-base"
    OUTPUT_DIR = "./xlmr_sentiment"
    LEARNING_RATE = 1e-5
    PER_DEVICE_TRAIN_BATCH_SIZE = 16
    WARMUP_RATIO = 0.07
    NUM_TRAIN_EPOCHS = 10

@dataclass
class Mamba(DefaultTrainingArguments):
    # Mamba-Original, Mamba-Helsinki, Mamba-Google-Translate
    NAME = "Mamba-Google"
    MODEL_NAME = "state-spaces/mamba-130m-hf"
    # "./mamba_original_sentiment" "./mamba_helsinki_sentiment" "./mamba_google_sentiment"
    OUTPUT_DIR = "./mamba_google_sentiment" 
    LEARNING_RATE = 1e-5
    NUM_TRAIN_EPOCHS = 7 # average of 4-10
    WEIGHT_DECAY = 0.1
    PER_DEVICE_TRAIN_BATCH_SIZE = 16
    FORCE_CUDA = "0"
    FORCE_PYTHON = "1"

class AdamW:
    OPTIMIZER_NAME = "AdamW"
    LR = 1e-5
    BETAS = (0.9, 0.95)
    EPS = 1e-6 if torch.cuda.is_available() else 1e-8
    WEIGHT_DECAY = 0.1
    EXCLUDE_KEYS = [ "EXCLUDE_KEYS", "OPTIMIZER_NAME" ]

class SGD:
    OPTIMIZER_NAME = "SGD"
    LR = 1e-4
    MOMENTUM = 0.95
    NESTEROV = True if MOMENTUM > 0 else False
    WEIGHT_DECAY = 0.01
    EXCLUDE_KEYS = [ "EXCLUDE_KEYS", "OPTIMIZER_NAME" ]

class App:
    ACTION = "TRAIN"
    HAS_GPU = torch.cuda.device_count() > 0
    DEVICE = 0 if HAS_GPU else -1

class Debug:
    CONFIG = False
    DATA = False
    OPTIMIZER = False
    PERFORMANCE = True
    TRAINER = False
    TRACE = False

class Data:
    TEXT_COL = "text"
    LABEL_COL = "label"
    MAX_LENGTH = 128
    DATASET = "./google_translated_reviews.jsonl"
    DATASET_SPLITS_DIR = "./datasets"
    SAVE_MODEL = True

class Translation:
    MODEL = "Helsinki-NLP/opus-mt-tl-en"
    CACHE_DIR = "cache/translations_cache.jsonl"
    BATCH_SIZE = 64
    MAX_NEW_TOKENS = 128

class Aws:
    ENABLED = os.path.isdir("/opt/ml")
    S3_BUCKET_NAME = "s3-ap-southeast-1-533267127131"

__all__ = ["App", "Aws", "Data", "Debug", "Translation", "DefaultTrainingArguments", "Mamba", "MBert", "Xlmr" ]
