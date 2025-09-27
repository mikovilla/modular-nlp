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

from src.config import App, Data, Mamba, MBert
from src.optimizer import SwitchOptimizerCallback, DebugCallback
from src.trainer import WeightedLossTrainer
from src import (
    config,
    helper,
    mamba, 
    metrics,
    translator,
    utility, 
)

def train(context):
    instance = context.instance
    set_seed(instance.SEED)

    isMamba = isinstance(instance, Mamba)
    val_ds, val_loaded = helper.load_dataset_if_exists("val_ds", context.val_ds, ignore=isMamba)
    train_ds, train_loaded = helper.load_dataset_if_exists("train_ds", context.train_ds, ignore=isMamba)
    test_ds, test_loaded = helper.load_dataset_if_exists("test_ds", context.test_ds, ignore=isMamba)

    training_args = helper.to_training_args(context.instance)     

    opt_cls, opt_kwargs = helper.to_optimizer_args(config.AdamW)
    opt_cls_switch, opt_kwargs_switch = helper.to_optimizer_args(config.SGD)
    print(opt_cls.__name__, opt_kwargs)
    print(opt_cls_switch.__name__, opt_kwargs_switch)
    switch_opt = SwitchOptimizerCallback(switch_after_epoch=3, 
                                        opt_class=opt_cls_switch,
                                        opt_kwargs=opt_kwargs_switch)
    trainer = WeightedLossTrainer(
        model=context.model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=context.tokenizer,
        data_collator=context.data_collator,
        compute_metrics=metrics.compute_metrics,
        class_weights=context.class_weights,
        optimizer=opt_cls(context.model.parameters(), **opt_kwargs)
    )
    switch_opt.bind(trainer)
    trainer.add_callback(switch_opt)
    
    if App.DEBUG:
        dbg = DebugCallback(); 
        dbg.bind(trainer)
        trainer.add_callback(dbg)
    
    trainer.train()
    eval_metrics = trainer.evaluate(test_ds)
    helper.print_header(f"{context.instance.MODEL_NAME} evaluation metrics")
    pprint.pprint(eval_metrics)

    if Data.SAVE_MODEL:
        os.makedirs(instance.OUTPUT_DIR, exist_ok=True)
        trainer.model.config.save_pretrained(instance.OUTPUT_DIR)
        trainer.save_model(instance.OUTPUT_DIR)
        context.tokenizer.save_pretrained(instance.OUTPUT_DIR)
        
        context.val_ds.save_to_disk(f"{Data.DATASET_SPLITS_DIR}/val_ds") if not (val_loaded and isMamba) else None
        context.train_ds.save_to_disk(f"{Data.DATASET_SPLITS_DIR}/train_ds") if not (train_loaded and isMamba) else None
        context.test_ds.save_to_disk(f"{Data.DATASET_SPLITS_DIR}/test_ds") if not (test_loaded and isMamba) else None

    return trainer

def infer(texts, instance_cls):
    instance = instance_cls()
    saved_model = mamba.load_saved_model() 
    model = saved_model.model if isinstance(instance, Mamba) else AutoModelForSequenceClassification.from_pretrained(instance.OUTPUT_DIR)
    tokenizer = saved_model.tokenizer if isinstance(instance, Mamba) else AutoTokenizer.from_pretrained(instance.OUTPUT_DIR)

    enc = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=Data.MAX_LENGTH)
    model.eval()
    with torch.no_grad():
        logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
    helper.print_header("sample predictions")
    print(list(zip(texts, preds)))

def ensemble(instance_classes, temps=None, weights=None):

    val_ds = load_from_disk(f"{Data.DATASET_SPLITS_DIR}/val_ds")
    
    instances = []
    for instance_cls in instance_classes:
        instances.append(instance_cls())
        
    m = len(instances)
    temps   = temps   or [1.0] * m
    weights = weights or [1.0] * m

    all_logits = []
    y_true = None
    
    for instance in instances:
        saved_model = mamba.load_saved_model() 
        model = saved_model.model if isinstance(instance, Mamba) else AutoModelForSequenceClassification.from_pretrained(instance.OUTPUT_DIR)
        tokenizer = saved_model.tokenizer if isinstance(instance, Mamba) else AutoTokenizer.from_pretrained(instance.OUTPUT_DIR)
        texts = [tokenizer.decode(ex['input_ids'], skip_special_tokens=True) for ex in val_ds]
        enc = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=Data.MAX_LENGTH)
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