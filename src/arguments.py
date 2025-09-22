from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="runs/baseline",
    evaluation_strategy="epoch",      # or "steps" if your dataset is large
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,

    num_train_epochs=3,               # 3–5 is typical for classification
    per_device_train_batch_size=16,   # adjust to fit memory
    per_device_eval_batch_size=32,

    learning_rate=2e-5,               # works well for base-sized encoders
    weight_decay=0.01,
    adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
    max_grad_norm=1.0,
    warmup_ratio=0.06,                # ~6% warmup is a solid default
    lr_scheduler_type="linear",

    gradient_accumulation_steps=1,    # bump this if you need a bigger effective batch
    fp16=True,                        # if you have NVIDIA w/ AMP
    dataloader_num_workers=4,
    load_best_model_at_end=True,
    metric_for_best_model="f1",       # change to your metric (e.g., "accuracy")
    greater_is_better=True,
    seed=42
)


mbert_args = TrainingArguments(
    output_dir="runs/mbert_tuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,                 # try 3–5; add early stopping
    per_device_train_batch_size=32,     # mBERT fits larger batches
    per_device_eval_batch_size=64,
    learning_rate=3e-5,                 # mBERT often likes 2e-5 ~ 5e-5
    weight_decay=0.01,
    warmup_ratio=0.06,
    lr_scheduler_type="linear",
    max_grad_norm=1.0,
    fp16=True,
    gradient_accumulation_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    seed=42
)


xlmr_base_args = TrainingArguments(
    output_dir="runs/xlmr_base_tuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,     # usually needs smaller batch than mBERT
    per_device_eval_batch_size=32,
    learning_rate=2e-5,                 # try 1e-5 ~ 2e-5
    weight_decay=0.01,
    warmup_ratio=0.08,                  # slightly longer warmup often stabilizes
    lr_scheduler_type="linear",
    max_grad_norm=1.0,
    fp16=True,
    gradient_accumulation_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    seed=42
)

xlmr_large_args = TrainingArguments(
    output_dir="runs/xlmr_large_tuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,      # may need 4–8 depending on GPU
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,      # boosts effective batch without OOM
    learning_rate=1e-5,                 # 7e-6 ~ 1.5e-5 typical
    weight_decay=0.01,
    warmup_ratio=0.1,                    # longer warmup for stability
    lr_scheduler_type="linear",
    max_grad_norm=0.8,                   # slightly tighter clipping can help
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    seed=42
)


# less than 1b
from transformers import TrainingArguments

mamba_cls_full_args = TrainingArguments(
    output_dir="runs/mamba_cls_full",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,

    num_train_epochs=4,                 # 3–5 typical
    per_device_train_batch_size=16,     # drop if seq_len > 512
    per_device_eval_batch_size=32,

    learning_rate=2e-5,                 # 1e-5–3e-5 for full FT
    weight_decay=0.05,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",

    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    fp16=True,                          # or bf16=True if supported
    gradient_checkpointing=True,

    load_best_model_at_end=True,
    metric_for_best_model="f1",         # or "accuracy"
    greater_is_better=True,
    seed=42
)

#LoRa
from transformers import TrainingArguments

mamba_cls_lora_args = TrainingArguments(
    output_dir="runs/mamba_cls_lora",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,

    num_train_epochs=4,
    per_device_train_batch_size=32,     # LoRA is lighter; can bump batch
    per_device_eval_batch_size=64,

    learning_rate=1e-4,                 # 5e-5–3e-4 typical for LoRA
    weight_decay=0.0,                    # often 0 for adapters
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",

    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    fp16=True,
    gradient_checkpointing=True,

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    seed=42
)


from transformers import TrainingArguments

mamba_causal_lm_args = TrainingArguments(
    output_dir="runs/mamba_causal_lm",
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    logging_steps=50,

    num_train_epochs=2,                 # focus on total tokens/steps instead
    per_device_train_batch_size=4,      # long seqs → smaller batches
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,      # aim for effective batch ≥ 128 tokens*examples

    learning_rate=1e-4 if "lora" else 2e-5,
    weight_decay=0.01 if not "lora" else 0.0,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",

    fp16=True,
    gradient_checkpointing=True,

    max_grad_norm=1.0,
    save_total_limit=3,
    seed=42
)
