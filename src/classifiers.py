import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import (
    AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
)
from transformers.modeling_outputs import SequenceClassifierOutput

@dataclass
class MambaLoadResult:
    tokenizer: object
    model: nn.Module

class MambaClassifier(nn.Module):
    def __init__(self, base_model: nn.Module, hidden_size: int, num_labels: int,
                 id2label=None, label2id=None, dropout_p: float = 0.1, config = None):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.id2label = id2label
        self.label2id = label2id
        self.config = config

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs.last_hidden_state  # [B, T, H]

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1) - 1
            lengths = torch.clamp(lengths, min=0)
            pooled = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), lengths]
        else:
            pooled = hidden_states[:, -1, :]

        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)

def ensure_tokens_and_sync(tokenizer, model, cfg):
    if tokenizer.pad_token_id is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": "<s>"})

    model.config = cfg

    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.bos_token_id = tokenizer.bos_token_id

    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.pad_token_id = tokenizer.pad_token_id
        gen_cfg.eos_token_id = tokenizer.eos_token_id
        gen_cfg.bos_token_id = tokenizer.bos_token_id

    if hasattr(model, "resize_token_embeddings"):
        if len(tokenizer) != model.get_input_embeddings().weight.size(0):
            model.resize_token_embeddings(len(tokenizer))

def load_mamba_for_classification(model_name: str, num_labels: int, id2label, label2id) -> MambaLoadResult:
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    cfg.num_labels = num_labels
    cfg.id2label = id2label
    cfg.label2id = label2id
    cfg.problem_type = "single_label_classification"

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=False,
        )
    except ValueError as e:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=False,
        )

    if tokenizer.pad_token_id is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=cfg,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
    except Exception:
        base = AutoModel.from_pretrained(model_name, config=cfg, trust_remote_code=True)

        hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
        if hidden_size is None:
            hidden_size = base.get_input_embeddings().weight.shape[1]

        model = MambaClassifier(base, hidden_size, num_labels, id2label=id2label, label2id=label2id)

    ensure_tokens_and_sync(tokenizer, model, cfg)
    
    if hasattr(model, "resize_token_embeddings"):
        if len(tokenizer) != model.get_input_embeddings().weight.size(0):
            model.resize_token_embeddings(len(tokenizer))

    return MambaLoadResult(tokenizer, model)