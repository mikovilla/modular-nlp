import numpy as np
import os
import pprint
import torch
import torch.nn.functional as F

from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer, 
    set_seed,
    TrainingArguments,
)

from src.config import *
from src.optimizer import SwitchOptimizerCallback, DebugCallback
from src.trainer import WeightedLossTrainer
from src import (
    helper,
    mamba, 
    metrics,
    translator,
    utility, 
)

def train(context):
    modelConfig = context.modelConfig
    set_seed(SharedConfig.SEED)

    isMamba = isinstance(modelConfig, MambaConfig)
    val_ds, val_loaded = helper.load_dataset_if_exists("val_ds", context.val_ds, ignore=isMamba)
    train_ds, train_loaded = helper.load_dataset_if_exists("train_ds", context.train_ds, ignore=isMamba)
    test_ds, test_loaded = helper.load_dataset_if_exists("test_ds", context.test_ds, ignore=isMamba)
    
    training_args = TrainingArguments(
        output_dir=modelConfig.OUTPUT_DIR,
        per_device_train_batch_size=SharedConfig.BATCH_SIZE,
        per_device_eval_batch_size=SharedConfig.BATCH_SIZE,
        learning_rate=SharedConfig.LR,
        num_train_epochs=SharedConfig.EPOCHS,
        weight_decay=SharedConfig.WEIGHT_DECAY,
        warmup_ratio=SharedConfig.WARMUP_RATIO,
        lr_scheduler_type="constant",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
        fp16=SharedConfig.USE_FP16,
        gradient_accumulation_steps=SharedConfig.GRAD_ACCUM_STEPS,
        save_total_limit=SharedConfig.SAVE_TOTAL_LIMIT,
        logging_steps=50,
        dataloader_num_workers=2,
    )      

    switch_cb = SwitchOptimizerCallback(switch_after_epoch=2, 
                                        opt_class=torch.optim.SGD,
                                        opt_kwargs={"lr":3e-3, "momentum":0.9, "nesterov":True})
    trainer = WeightedLossTrainer(
        model=context.model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=context.tokenizer,
        data_collator=context.data_collator,
        compute_metrics=metrics.compute_metrics,
        class_weights=context.class_weights
    )
    switch_cb.bind(trainer)
    trainer.add_callback(switch_cb)
    
    if AppConfig.DEBUG:
        dbg = DebugCallback(); 
        dbg.bind(trainer)
        trainer.add_callback(dbg)
    
    trainer.train()
    eval_metrics = trainer.evaluate(test_ds)
    helper.print_header(f"{context.modelConfig.MODEL_NAME} evaluation metrics")
    pprint.pprint(eval_metrics)

    if AppConfig.SAVE_MODEL:
        os.makedirs(modelConfig.OUTPUT_DIR, exist_ok=True)
        trainer.model.config.save_pretrained(modelConfig.OUTPUT_DIR)
        trainer.save_model(modelConfig.OUTPUT_DIR)
        context.tokenizer.save_pretrained(modelConfig.OUTPUT_DIR)
        
        context.val_ds.save_to_disk(f"{AppConfig.DATASET_SPLITS_DIR}/val_ds") if not (val_loaded and isMamba) else None
        context.train_ds.save_to_disk(f"{AppConfig.DATASET_SPLITS_DIR}/train_ds") if not (train_loaded and isMamba) else None
        context.test_ds.save_to_disk(f"{AppConfig.DATASET_SPLITS_DIR}/test_ds") if not (test_loaded and isMamba) else None

    return trainer

def infer(texts, configClass):
    modelConfig = configClass()
    saved_model = mamba.load_saved_model() 
    model = saved_model.model if isinstance(modelConfig, MambaConfig) else AutoModelForSequenceClassification.from_pretrained(modelConfig.OUTPUT_DIR)
    tokenizer = saved_model.tokenizer if isinstance(modelConfig, MambaConfig) else AutoTokenizer.from_pretrained(modelConfig.OUTPUT_DIR)

    enc = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=SharedConfig.MAX_LENGTH)
    model.eval()
    with torch.no_grad():
        logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
    helper.print_header("sample predictions")
    print(list(zip(texts, preds)))

def ensemble(configClasses, temps=None, weights=None):

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