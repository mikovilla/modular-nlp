import torch.optim as optim

from transformers import TrainerCallback

from src.config import App

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

        if App.DEBUG:
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
        if App.DEBUG:
            print(f"[SWITCHED] base={type(base).__name__} lr={self.trainer.optimizer.param_groups[0]['lr']}")
        
        return control

class DebugCallback(TrainerCallback):
    def __init__(self): 
        self.trainer = None

    def bind(self, trainer): 
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kw):
        opt = self.trainer.optimizer
        wrapped = type(opt).__name__
        base = getattr(opt, "optimizer", opt)

        if opt and hasattr(opt, "param_groups") and opt.param_groups:
            group = opt.param_groups[0]
            # Print common optimizer settings
            hyperparams = {k: v for k, v in group.items() if k != "params"}
        else:
            hyperparams = {}

        print(
            f"[DBG-LIVE] epoch_start={state.epoch} global_step={state.global_step} "
            f"wrapped={wrapped} base={type(base).__name__} "
            f"lr={hyperparams.get('lr')} id={id(base)} "
            f"hyperparams={hyperparams}"
        )


    #def on_epoch_end(self, args, state, control, model=None, optimizer=None, **kw):
        #print(f"[DBG] epoch_end={state.epoch} global_step={state.global_step}")

    #def on_step_end(self, args, state, control, **kw):
        #print(f"[DBG] step_end global_step={state.global_step}")