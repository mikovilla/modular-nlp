import os
import torch
import pprint

from transformers import (
    set_seed,
    TrainingArguments
)

from src import utility, metrics, translator, helper
from src.overrides import WeightedLossTrainer
from src.config import SharedConfig, AppConfig

def train(context):
    modelConfig = context.modelConfig
    set_seed(SharedConfig.SEED)

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
        model=context.model,
        args=training_args,
        train_dataset=context.train_ds,
        eval_dataset=context.val_ds,
        tokenizer=context.tokenizer,
        data_collator=context.data_collator,
        compute_metrics=metrics.compute_metrics,
        class_weights=context.class_weights,
    )

    trainer.train()
    eval_metrics = trainer.evaluate(eval_dataset=context.test_ds)
    print("\r\nTest metrics:")
    pprint.pprint(eval_metrics)

    if AppConfig.SAVE_MODEL:
        os.makedirs(modelConfig.OUTPUT_DIR, exist_ok=True)
        trainer.save_model(modelConfig.OUTPUT_DIR)
        context.tokenizer.save_pretrained(modelConfig.OUTPUT_DIR)

    return trainer

def infer(texts, tokenizer, model):
    enc = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=SharedConfig.MAX_LENGTH)
    model.eval()
    with torch.no_grad():
        logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
    print("Sample predictions:", list(zip(texts, preds)))