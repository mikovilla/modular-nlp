import inspect
import time
import torch
import torch.nn as nn
from transformers import Trainer

class InputsForwardingTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        t0 = time.perf_counter()
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )
        dt = time.perf_counter() - t0

        for cb in list(getattr(self.callback_handler, "callbacks", [])):
            on_pred = getattr(cb, "on_prediction_step", None)
            if on_pred is None:
                continue
            try:
                on_pred(self.args, self.state, self.control,
                        inputs=inputs, model=model, optimizer=self.optimizer, dt=dt)
            except TypeError:
                on_pred(self.args, self.state, self.control)
        return loss, logits, labels

class WeightedLossTrainer(InputsForwardingTrainer):
    def __init__(self, *args, class_weights=None, optimizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.class_weights = class_weights
        self._dbg = False

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.get("labels", None)

        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = getattr(outputs, "logits", None)

        if logits is None:
            raise ValueError("Model outputs have no `.logits`; cannot compute CrossEntropyLoss.")

        if labels is not None and labels.dtype != torch.long:
            labels = labels.long()

        if not self._dbg:
            self._dbg = True

        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(device=logits.device, dtype=logits.dtype)

        if labels is None:
            return (torch.tensor(0.0, device=logits.device, dtype=logits.dtype), outputs) if return_outputs else torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        if logits.dim() == 2:
            loss = nn.CrossEntropyLoss(weight=weight)(logits, labels)

        elif logits.dim() == 3:
            if labels.dim() == 2:
                loss = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
            else:
                attn = inputs.get("attention_mask", None)
                if attn is not None and attn.dim() == 2:
                    mask = attn.unsqueeze(-1).to(dtype=logits.dtype)
                    pooled = (logits * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
                else:
                    pooled = logits.mean(dim=1)
                loss = nn.CrossEntropyLoss(weight=weight)(pooled, labels)
        else:
            raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

        return (loss, outputs) if return_outputs else loss
