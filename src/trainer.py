import torch
import torch.nn as nn

from torch.optim import AdamW
from transformers import Trainer

from src.config import App

class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self._dbg = False

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                betas=(0.9, 0.999),
                eps = 1e-6 if App.HAS_GPU else 1e-8,
                weight_decay=self.args.weight_decay
            )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = getattr(outputs, "logits", None)

        if labels is not None and labels.dtype != torch.long:
            labels = labels.long()

        if not self._dbg:
            self._dbg = True

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None

        if logits.dim() == 2:
            loss = nn.CrossEntropyLoss(weight=weight)(logits, labels)

        elif logits.dim() == 3:
            if labels is not None and labels.dim() == 2:
                loss = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)(
                    logits.view(-1, logits.size(-1)), labels.view(-1)
                )
            else:
                attn = inputs.get("attention_mask")
                if attn is not None and attn.dim() == 2:
                    mask = attn.unsqueeze(-1).float()
                    pooled = (logits * mask).sum(1) / mask.sum(1).clamp_min(1e-9)  # [B, C]
                else:
                    pooled = logits.mean(dim=1)
                loss = nn.CrossEntropyLoss(weight=weight)(pooled, labels)
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")

        return (loss, outputs) if return_outputs else loss
