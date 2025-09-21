import numpy as np
import os
import torch
import torch.nn.functional as F

from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import *
from src import (
    helper, 
    mamba, 
    metrics
)

def evaluate(configClasses, temps=None, weights=None):

    val_ds, val_loaded = helper.load_dataset_if_exists("val_ds", None)
    if not val_loaded:
        raise ValueError("Validation dataset not found")
    
    modelConfigs = []
    for configClass in configClasses:
        modelConfigs.append(configClass())
        
    m = len(modelConfigs)
    temps   = temps   or [1.0] * m
    weights = weights or [1.0] * m

    all_logits = []
    y_true = None
    
    for modelConfig in modelConfigs:
        saved_model = mamba.load_saved_model() 
        model = saved_model.model if isinstance(modelConfig, MambaConfig) else AutoModelForSequenceClassification.from_pretrained(modelConfig.OUTPUT_DIR)
        tokenizer = saved_model.tokenizer if isinstance(modelConfig, MambaConfig) else AutoTokenizer.from_pretrained(modelConfig.OUTPUT_DIR)
        texts = [tokenizer.decode(ex['input_ids'], skip_special_tokens=True) for ex in val_ds]
        enc = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=SharedConfig.MAX_LENGTH)
        y_true = torch.tensor(val_ds["label"]).long()
        with torch.no_grad():
            all_logits.append(model(**{k: v.to(model.device) for k, v in enc.items()}).logits)
    
    w = torch.tensor(weights, dtype=all_logits[0].dtype, device=all_logits[0].device)
    w = w / w.sum()

    ensembled_logits = sum(w[j] * (all_logits[j] / all_logits[j]) for j in range(len(all_logits)))

    probs = F.softmax(ensembled_logits, dim=-1)
    preds = probs.argmax(dim=-1)

    if torch.is_tensor(y_true):
        y_np = y_true.cpu().numpy()
    else:
        y_np = np.asarray(y_true)

    y_pred = preds.cpu().numpy()
    p_np   = probs.cpu().numpy()

    nll = F.cross_entropy(ensembled_logits, torch.as_tensor(y_np, device=ensembled_logits.device)).item()
    metrics = {
        "accuracy": accuracy_score(y_np, y_pred),
        "f1_macro": f1_score(y_np, y_pred, average="macro"),
        "f1_micro": f1_score(y_np, y_pred, average="micro"),
        "nll": nll
    }
    return metrics, y_pred, p_np
    