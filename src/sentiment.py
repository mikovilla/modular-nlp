import os
import torch
import pprint

from transformers import (
    set_seed,
    TrainingArguments
)

from src import utility, metrics, translator, helper
from src.trainer import WeightedLossTrainer
from src.config import SharedConfig, AppConfig
from src.optimizer import SwitchOptimizerCallback, DebugCallback

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
        lr_scheduler_type="constant",
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

    switch_cb = SwitchOptimizerCallback(switch_after_epoch=2, 
                                        opt_class=torch.optim.SGD,
                                        opt_kwargs={"lr":3e-3, "momentum":0.9, "nesterov":True})
    trainer = WeightedLossTrainer(
        model=context.model,
        args=training_args,
        train_dataset=context.train_ds,
        eval_dataset=context.val_ds,
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
    eval_metrics = trainer.evaluate(eval_dataset=context.test_ds)
    helper.print_header(f"{context.modelConfig.MODEL_NAME} evaluation metrics")
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
    helper.print_header("sample predictions")
    print( list(zip(texts, preds)))