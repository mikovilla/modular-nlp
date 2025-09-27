import torch.optim as optim

from transformers import TrainerCallback

from src.config import Debug

class SwitchOptimizerCallback(TrainerCallback):
    def __init__(self, switch_after_epoch:int, opt_class, opt_kwargs: dict = None):
        self.switch_after_epoch = switch_after_epoch
        self.opt_class = opt_class
        self.opt_kwargs = opt_kwargs or {}
        self.trainer = None

    def bind(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if state.epoch is None or int(state.epoch) != self.switch_after_epoch:
            return control

        if Debug.OPTIMIZER:
            print(f"Switching optimizer to {self.opt_class.__name__} right after epoch {int(state.epoch)} "
                  f"(next epoch will print {self.opt_class.__name__} in your debug)")

        base_opt = self.opt_class(model.parameters(), **self.opt_kwargs)

        accel = getattr(self.trainer, "accelerator", None)
        if accel is not None:
            new_opt = accel.prepare_optimizer(base_opt)
            self.trainer.optimizer = new_opt
            if hasattr(accel, "_optimizers"):
                accel._optimizers = [new_opt]
        else:
            self.trainer.optimizer = base_opt

        if getattr(self.trainer, "scaler", None) is not None:
            new_scaler = GradScaler()
            self.trainer.scaler = new_scaler
            if accel is not None:
                accel.scaler = new_scaler

        steps = self.trainer.state.max_steps or self.trainer.get_num_training_steps(self.trainer.get_train_dataloader())
        self.trainer.create_scheduler(num_training_steps=steps, optimizer=self.trainer.optimizer)

        base = getattr(self.trainer.optimizer, "optimizer", self.trainer.optimizer)
        if Debug.OPTIMIZER:
            print(f"[OPTIMIZER_SWITCH] base={type(base).__name__} lr={self.trainer.optimizer.param_groups[0]['lr']}")
        
        return control