import torch
from transformers import Trainer

class WeightedLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.class_weights = kwargs.pop("class_weights", None)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss