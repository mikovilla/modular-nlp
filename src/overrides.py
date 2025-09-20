from transformers import Trainer
import torch
import torch.nn as nn

class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits", outputs[0])

        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device, dtype=logits.dtype)
            loss_fct = nn.CrossEntropyLoss(weight=cw)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss