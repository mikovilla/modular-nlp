import os
import numpy as np
import torch
import pprint

from pathlib import Path
from transformers import (
    set_seed,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments
)

from src import utility, metrics, translator, helper
from src.overrides import WeightedLossTrainer
from src.config import SharedConfig, AppConfig

def train(configClass, require_translation: bool = False):
    modelConfig = configClass()
    set_seed(SharedConfig.SEED)
    
    jsonl = ""
    if not require_translation:
        jsonl = helper.read_jsonl_as_string(Path(AppConfig.DATASET))
    else:
        jsonl = translator.from_jsonl(AppConfig.DATASET)
        
    train_ds, val_ds, test_ds, label2id, id2label = utility.load_split_dataset(jsonl)
    num_labels = len(id2label)

    tokenizer, tok_fn = utility.make_tokenizer(modelConfig.MODEL_NAME)
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=[SharedConfig.TEXT_COL])
    val_ds   = val_ds.map(tok_fn, batched=True, remove_columns=[SharedConfig.TEXT_COL])
    test_ds  = test_ds.map(tok_fn, batched=True, remove_columns=[SharedConfig.TEXT_COL])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    y_train = np.array(train_ds["label"])
    counts = np.bincount(y_train, minlength=num_labels).astype(np.float32)
    weights = counts.sum() / (counts + 1e-9)
    class_weights = torch.tensor(weights / weights.mean())

    model = AutoModelForSequenceClassification.from_pretrained(
        modelConfig.MODEL_NAME,
        num_labels=num_labels,
        id2label={i: str(i) for i in range(num_labels)} if not isinstance(id2label, dict) else id2label,
        label2id={str(v): k for k, v in ({v: k for k, v in id2label.items()}).items()} if isinstance(id2label, dict) else None,
        problem_type="single_label_classification",
    )

    training_args = TrainingArguments(
        output_dir=modelConfig.OUTPUT_DIR,
        per_device_train_batch_size=SharedConfig.BATCH_SIZE,
        per_device_eval_batch_size=SharedConfig.BATCH_SIZE,
        learning_rate=SharedConfig.LR,
        num_train_epochs=SharedConfig.EPOCHS,
        weight_decay=SharedConfig.WEIGHT_DECAY,
        warmup_ratio=SharedConfig.WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
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

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics.compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    eval_metrics = trainer.evaluate(eval_dataset=test_ds)
    print("\r\nTest metrics:")
    pprint.pprint(eval_metrics)

    if AppConfig.SAVE_MODEL:
        os.makedirs(modelConfig.OUTPUT_DIR, exist_ok=True)
        trainer.save_model(modelConfig.OUTPUT_DIR)
        tokenizer.save_pretrained(modelConfig.OUTPUT_DIR)

    return trainer

def infer(texts, tokenizer, model):
    enc = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=SharedConfig.MAX_LENGTH)
    model.eval()
    with torch.no_grad():
        logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
    print("Sample predictions:", list(zip(texts, preds)))