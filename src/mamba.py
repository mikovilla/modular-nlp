import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import (
    AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
)
from transformers.modeling_outputs import SequenceClassifierOutput

from src import utility
from src.config import MambaConfig

@dataclass
class MambaLoadResult:
    tokenizer: object
    model: nn.Module

class MambaClassifier(nn.Module):
    def __init__(self, base_model: nn.Module, hidden_size: int, num_labels: int,
                 id2label=None, label2id=None, dropout_p: float = 0.1, config=None):
        super().__init__()
        self.backbone = base_model
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.id2label = id2label
        self.label2id = label2id
        self.config = config

    @property
    def base_model(self):  # compatibility alias (not registered twice)
        return self.backbone

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 👇 swallow labels so they don’t reach the backbone
        kwargs.pop("labels", None)
        kwargs.pop("label", None)

        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hs = out.last_hidden_state  # [B, T, H]

        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            pooled = (hs * m).sum(1) / m.sum(1).clamp_min(1e-9)   # [B, H]
        else:
            pooled = hs.mean(1)

        logits = self.classifier(self.dropout(pooled))             # [B, C]
        return SequenceClassifierOutput(logits=logits) 

    @property
    def device(self):
        return next(self.parameters()).device

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

        
def load_state(folder: str):
    state = None
    safep = os.path.join(folder, "model.safetensors")
    binp  = os.path.join(folder, "pytorch_model.bin")
    try:
        if os.path.isfile(safep):
            from safetensors.torch import load_file
            state = load_file(safep)
        elif os.path.isfile(binp):
            state = torch.load(binp, map_location="cpu")
    except Exception:
        state = None
    return state

def load_wrapper_state_with_embedding_fix(model, state, prefix_choices=("backbone", "base_model")):
    import torch
    # find the correct prefix used in the checkpoint
    emb_key = None
    for p in prefix_choices:
        k = f"{p}.embeddings.weight"
        if k in state:
            emb_key = k
            break

    if emb_key is not None:
        ckpt_emb = state[emb_key]                           # [V_ckpt, H]
        model_emb = model.backbone.get_input_embeddings().weight  # [V_model, H]
        if ckpt_emb.shape != model_emb.shape:
            # partial copy the overlapping rows
            with torch.no_grad():
                n = min(model_emb.shape[0], ckpt_emb.shape[0])
                model_emb[:n].copy_(ckpt_emb[:n])
            # remove the mismatched key so load_state_dict won't error
            state = dict(state)  # shallow copy
            state.pop(emb_key)

    # load the rest (classifier, layers, etc.)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


def load_model_for_classification(model_name, num_labels, id2label, label2id, backbone_name=None):
    import copy, os, torch
    from transformers import AutoConfig, AutoTokenizer, AutoModel

    # 0) Try to read a local state dict (wrapper ckpt) — ok if None
    state = load_state(model_name)
    has_wrapper = isinstance(state, dict) and any(
        k.startswith(("backbone.", "base_model.", "classifier.")) for k in state.keys()
    )

    # 1) Pick backbone source & config FROM THE SAME SOURCE
    src = backbone_name if has_wrapper else model_name
    backbone_cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)

    # 2) Tokenizer (don’t resize model yet)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    except Exception:
        tokenizer = utility.make_tokenizer(MambaConfig)
    if tokenizer.pad_token_id is None:
        if getattr(tokenizer, "eos_token", None):
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # 3) Backbone
    backbone = AutoModel.from_pretrained(src, trust_remote_code=True)

    # 4) Build wrapper (NOW `model` exists)
    hidden = getattr(backbone_cfg, "hidden_size", None) or getattr(backbone_cfg, "d_model", None) \
             or backbone.get_input_embeddings().weight.shape[1]
    task_cfg = copy.deepcopy(backbone_cfg)
    task_cfg.num_labels = num_labels
    task_cfg.id2label = id2label
    task_cfg.label2id = label2id
    task_cfg.problem_type = "single_label_classification"

    model = MambaClassifier(backbone, hidden, num_labels, id2label=id2label, label2id=label2id, config=task_cfg)

    # 5) Load wrapper state (handle vocab mismatch inside helper)
    if isinstance(state, dict):
        # optional: align old prefix
        if any(k.startswith("base_model.") for k in state):
            state = {k.replace("base_model.", "backbone.", 1): v for k, v in state.items()}
        _ = load_wrapper_state_with_embedding_fix(model, state)

    # 6) Finally, resize embeddings to match tokenizer (AFTER loading)
    emb_mod = model.backbone
    if hasattr(emb_mod, "resize_token_embeddings"):
        if len(tokenizer) != emb_mod.get_input_embeddings().weight.size(0):
            emb_mod.resize_token_embeddings(len(tokenizer))

    return MambaLoadResult(tokenizer, model)
