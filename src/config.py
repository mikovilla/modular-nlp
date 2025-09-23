import torch

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
    FP16 = torch.cuda.is_available()
    LEARNING_RATE = 2e-5
    GRADIENT_ACCUMULATION_STEPS = 1
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    WARMUP_RATIO = 0.06
    LR_SCHEDULER_TYPE = "linear"
    DATALOADER_NUM_WORKERS = 4
    LOAD_BEST_MODEL_AT_END = True
    METRIC_FOR_BEST_MODEL = "eval_f1_macro"
    GREATER_IS_BETTER = True
    SEED = 42
    EXCLUDE_KEYS = ["MODEL_NAME", "TEMP", "WEIGHT"]

@dataclass
class MBert(DefaultTrainingArguments):
    MODEL_NAME = "bert-base-multilingual-cased"
    OUTPUT_DIR = "./mbert_sentiment"

@dataclass
class Xlmr(DefaultTrainingArguments):
    MODEL_NAME = "xlm-roberta-base"
    OUTPUT_DIR = "./xlmr_sentiment"
    LEARNING_RATE = 1e-5
    PER_DEVICE_TRAIN_BATCH_SIZE = 16
    PER_DEVICE_EVAL_BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = 2
    MAX_GRAD_NORM = 0.95
    WARMUP_RATIO = 0.07

@dataclass
class Mamba(DefaultTrainingArguments):
    MODEL_NAME = "state-spaces/mamba-130m-hf"
    OUTPUT_DIR = "./mamba_sentiment"
    NUM_TRAIN_EPOCHS = 6
    WEIGHT_DECAY = 0.05
    FORCE_CUDA = "0"
    FORCE_PYTHON = "1"

class App:
    ACTION = "ENSEMBLE"
    HAS_GPU = torch.cuda.is_available()
    DEVICE = 0 if HAS_GPU else -1
    DEBUG = False

class Data:
    SHOW_ON_DEBUG = False
    TEXT_COL = "text"
    LABEL_COL = "label"
    MAX_LENGTH = 128
    DATASET = "./miko.jsonl"
    DATASET_SPLITS_DIR = "./datasets"
    SAVE_MODEL = True

class Translation:
    MODEL = "Helsinki-NLP/opus-mt-tl-en"
    CACHE_DIR = "cache/translations_cache.jsonl"
    BATCH_SIZE = 64
    MAX_NEW_TOKENS = 128

__all__ = ["App", "Translation", "MBert", "Xlmr", "Mamba"]