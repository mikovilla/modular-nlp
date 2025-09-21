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
    def __init__(self, model: nn.Module, hidden_size: int, config, dropout_p: float = 0.1):
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, config.num_labels)
        self.config = config

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop("labels", None)
        kwargs.pop("label", None)

        out = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hs = out.last_hidden_state  # [B, T, H]

        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            pooled = (hs * m).sum(1) / m.sum(1).clamp_min(1e-9)
        else:
            pooled = hs.mean(1)

        logits = self.classifier(self.dropout(pooled))
        return SequenceClassifierOutput(logits=logits) 

    @property
    def device(self):
        return next(self.parameters()).device

def load_model_for_classification(model_name, num_labels, id2label, label2id):
    tokenizer = utility.make_tokenizer(MambaConfig)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )
    
    hidden = getattr(config, "hidden_size", None) or getattr(config, "d_model", None) or model.get_input_embeddings().weight.shape[1]
    model = MambaClassifier(model, hidden, config)
    return MambaLoadResult(tokenizer, model)

def load_saved_model():
    src = MambaConfig.OUTPUT_DIR
    state = load_state(src)
    config = AutoConfig.from_pretrained(src, trust_remote_code=True)
    tokenizer = utility.make_tokenizer(MambaConfig)
    
    backbone = AutoModel.from_pretrained(MambaConfig.MODEL_NAME, trust_remote_code=True)
    hidden = getattr(config, "hidden_size", None) or getattr(config, "d_model", None) or model.get_input_embeddings().weight.shape[1]
    model = MambaClassifier(backbone, hidden, config)

    if isinstance(state, dict):
        if any(k.startswith("base_model.") for k in state):
            state = {k.replace("base_model.", "backbone.", 1): v for k, v in state.items()}
        _ = load_wrapper_state_with_embedding_fix(model, state)

    if hasattr(model, "resize_token_embeddings"):
        if len(tokenizer) != model.get_input_embeddings().weight.size(0):
            model.resize_token_embeddings(len(tokenizer))
    
    return MambaLoadResult(tokenizer, model)
     
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
    emb_key = None
    for p in prefix_choices:
        k = f"{p}.embeddings.weight"
        if k in state:
            emb_key = k
            break

    if emb_key is not None:
        ckpt_emb = state[emb_key]
        model_emb = model.backbone.get_input_embeddings().weight
        if ckpt_emb.shape != model_emb.shape:
            with torch.no_grad():
                n = min(model_emb.shape[0], ckpt_emb.shape[0])
                model_emb[:n].copy_(ckpt_emb[:n])
            state = dict(state)
            state.pop(emb_key)

    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected
    


