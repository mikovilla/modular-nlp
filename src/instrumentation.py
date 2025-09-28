from __future__ import annotations

import torch

import time, math, inspect
from dataclasses import dataclass
from typing import Optional, Dict, Any
from transformers import TrainerCallback

from src.config import App

try:
    from fvcore.nn import FlopCountAnalysis
    _HAS_FVCORE = True
except Exception:
    _HAS_FVCORE = False

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _HAS_NVML = True
except Exception:
    _NVML_HANDLE = None
    _HAS_NVML = False

@dataclass
class CostModel:
    gpu_hourly_usd: float = 0.0
    include_energy: bool = False
    gpu_watts: float = 0.0
    energy_usd_per_kwh: float = 0.0

    input_token_usd_per_1k: float = 0.0
    output_token_usd_per_1k: float = 0.0

class PerfCallback(TrainerCallback):
    def __init__(
        self,
        seq_key: str = "input_ids",
        mask_key: str = "attention_mask",
        label_key: str = "labels",
        estimate_flops_seq_len: Optional[int] = None,
        flops_batch_size: int = 8,
        with_gpu_util: bool = True,
        with_grad_norm: bool = True,
        with_lr_log: bool = True,
        with_padding_stats: bool = True,
        with_examples_sec: bool = True,
        cost: Optional[CostModel] = None,
        dbg: bool = False,
    ):
        self.seq_key = seq_key
        self.mask_key = mask_key
        self.label_key = label_key
        self.estimate_flops_seq_len = estimate_flops_seq_len
        self.flops_batch_size = flops_batch_size
        self.with_gpu_util = with_gpu_util and _HAS_NVML
        self.with_grad_norm = with_grad_norm
        self.with_lr_log = with_lr_log
        self.with_padding_stats = with_padding_stats
        self.with_examples_sec = with_examples_sec
        self.cost = cost or CostModel()
        self.dbg = dbg

        self._trainer = None
        self._step_t0: Optional[float] = None
        self._epoch_tokens = 0
        self._epoch_examples = 0
        self._epoch_time = 0.0
        self._peak_alloc = 0
        self._flops_per_fwd: Optional[int] = None
        self._wall_t0: Optional[float] = None
        self._eval_tokens = 0
        self._eval_examples = 0
        self._eval_time = 0.0
        self._last_eval_accuracy: Optional[float] = None
        self._padding_total = 0
        self._tokens_total = 0
        self._last_lr: Optional[float] = None
        self._grad_norm: Optional[float] = None

    def bind(self, trainer):
        self._trainer = trainer
    def _get_model(self, **kw):
        return (self._trainer.model if self._trainer else kw.get("model", None))
    def _get_optimizer(self, **kw):
        return (self._trainer.optimizer if self._trainer else kw.get("optimizer", None))

    @staticmethod
    def _is_rank0(trainer) -> bool:
        acc = getattr(trainer, "accelerator", None)
        if acc is not None:
            return acc.process_index == 0
        return getattr(trainer, "is_world_process_zero", True)

    @staticmethod
    def _count_tokens_and_examples(inputs: Dict[str, Any], seq_key: str, mask_key: str):
        if not isinstance(inputs, dict):
            return 0, 0, 0, 0
        ids = inputs.get(seq_key)
        mask = inputs.get(mask_key)
        if isinstance(ids, torch.Tensor):
            B, L = ids.shape[:2]
            examples = int(B)
            if isinstance(mask, torch.Tensor):
                real = int(mask.sum().item())
            else:
                real = int((ids != 0).sum().item())
            total = B * L
            padding = total - real
            return real, examples, padding, total
        return 0, 0, 0, 0

    @staticmethod
    def _gpu_peak_alloc_bytes() -> int:
        if App.HAS_GPU:
            return torch.cuda.max_memory_allocated()
        return 0

    @staticmethod
    def _bytes_to_gib(n: int) -> float:
        return n / (1024 ** 3)

    def _estimate_flops_once(self, trainer):
        if self._flops_per_fwd is not None or not _HAS_FVCORE or self.estimate_flops_seq_len is None:
            return
        model = trainer.model
        model.eval()
        bs = self.flops_batch_size
        L = self.estimate_flops_seq_len
        device = next(model.parameters()).device
        with torch.no_grad():
            dummy = {
                self.seq_key: torch.randint(100, (bs, L), device=device),
                self.mask_key: torch.ones((bs, L), device=device, dtype=torch.long),
            }
            try:
                self._flops_per_fwd = int(FlopCountAnalysis(model, (dummy,)).total())
            except Exception:
                self._flops_per_fwd = None

    def _gpu_util_str(self) -> str:
        if not self.with_gpu_util or not _HAS_NVML:
            return ""
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
            return f" gpu_util={util.gpu}% mem_util={util.memory}%"
        except Exception:
            return ""

    def _cost_usd_for_seconds(self, seconds: float, input_tokens: int = 0, output_tokens: int = 0) -> float:
        cost = 0.0
        if self.cost.gpu_hourly_usd > 0:
            cost += self.cost.gpu_hourly_usd * (seconds / 3600.0)
        if self.cost.include_energy and self.cost.gpu_watts > 0 and self.cost.energy_usd_per_kwh > 0:
            kwh = (self.cost.gpu_watts / 1000.0) * (seconds / 3600.0)
            cost += kwh * self.cost.energy_usd_per_kwh
        if self.cost.input_token_usd_per_1k > 0 or self.cost.output_token_usd_per_1k > 0:
            cost += (input_tokens / 1000.0) * self.cost.input_token_usd_per_1k
            cost += (output_tokens / 1000.0) * self.cost.output_token_usd_per_1k
        return cost

    def on_train_begin(self, args, state, control, **kw):
        self._trainer = kw.get("trainer", self._trainer)
        if App.HAS_GPU:
            torch.cuda.reset_peak_memory_stats()
        self._peak_alloc = 0
        self._epoch_tokens = self._epoch_examples = 0
        self._epoch_time = 0.0
        self._padding_total = self._tokens_total = 0
        self._wall_t0 = time.perf_counter()
        self._estimate_flops_once(self._trainer)

    def on_step_begin(self, args, state, control, **kw):
        self._step_t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        dt = time.perf_counter() - (self._step_t0 or time.perf_counter())
        self._epoch_time += dt

        inputs = kw.get("inputs", {})
        tok, ex, pad, tot = self._count_tokens_and_examples(inputs, self.seq_key, self.mask_key)
        self._epoch_tokens += tok
        self._epoch_examples += ex
        if self.with_padding_stats:
            self._padding_total += pad
            self._tokens_total += tot

        self._peak_alloc = max(self._peak_alloc, self._gpu_peak_alloc_bytes())

        if self.with_lr_log:
            opt = self._trainer.optimizer if self._trainer is not None else None
            if opt and getattr(opt, "param_groups", None):
                lrs = [g.get("lr", 0.0) for g in opt.param_groups]
                self._last_lr = float(sum(lrs) / len(lrs)) if lrs else None

        if self.with_grad_norm:
            total_sq = 0.0
            for p in self._trainer.model.parameters():
                if p.grad is not None:
                    gn = p.grad.data.norm(2).item()
                    total_sq += gn * gn
            self._grad_norm = math.sqrt(total_sq) if total_sq > 0 else None

        return control

    def on_epoch_end(self, args, state, control, **kw):
        trainer = self._trainer or kw.get("trainer")
        if not trainer or not self._is_rank0(trainer):
            return control

        tok = self._epoch_tokens
        ex = self._epoch_examples
        sec = max(self._epoch_time, 1e-9)
        tps = tok / sec
        eps = ex / sec if sec > 0 else 0.0
        gib = self._bytes_to_gib(self._peak_alloc)

        msg = (f"[TRAINING_PERFORMANCE] epoch={state.epoch:.1f} tokens={tok} "
               f"time={sec:.2f}s tok/s={tps:.1f}")
        if self.with_examples_sec:
            msg += f" ex/s={eps:.2f}"
        msg += f" peak_mem={gib:.2f}GiB"
        if self._flops_per_fwd is not None:
            msg += f" ~FLOPs/forward(batch)={self._flops_per_fwd/1e12:.2f} TFLOPs"
        if self.with_lr_log and self._last_lr is not None:
            msg += f" lr={self._last_lr:.2e}"
        if self.with_grad_norm and self._grad_norm is not None:
            msg += f" grad_norm={self._grad_norm:.2f}"
        if self.with_padding_stats and self._tokens_total > 0:
            pad_pct = 100.0 * self._padding_total / self._tokens_total
            msg += f" pad%={pad_pct:.1f}"
        msg += self._gpu_util_str()
        print(msg)

        self._epoch_tokens = self._epoch_examples = 0
        self._epoch_time = 0.0
        self._padding_total = self._tokens_total = 0
        self._peak_alloc = 0
        if App.HAS_GPU:
            torch.cuda.reset_peak_memory_stats()
        return control

    @staticmethod
    def _first_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                t = PerfCallback._first_tensor(v)
                if t is not None: return t
        if isinstance(obj, (list, tuple)):
            for v in obj:
                t = PerfCallback._first_tensor(v)
                if t is not None: return t
        return None

    def _count_tokens_examples_any(self, inputs, seq_key, mask_key, args):
        if isinstance(inputs, dict):
            ids = inputs.get(seq_key, None)
            mask = inputs.get(mask_key, None)
            if isinstance(ids, torch.Tensor):
                B, L = ids.shape[:2]
                ex = int(B)
                if isinstance(mask, torch.Tensor):
                    tok = int(mask.sum().item())
                else:
                    tok = int((ids != 0).sum().item())
                return tok, ex

        t = self._first_tensor(inputs)
        if t is not None and t.dim() >= 2:
            B, L = int(t.size(0)), int(t.size(1))
            return B * L, B

        bs = getattr(args, "per_device_eval_batch_size", 0) or 0
        L  = getattr(self, "_eval_seq_len_hint", None)
        if L is None:
            L = getattr(args, "max_seq_length", 0) or 128
            self._eval_seq_len_hint = int(L)
        return int(bs * L), int(bs)

    def on_prediction_step(self, args, state, control, **kw):
        inputs = kw.get("inputs")
        tok, ex = self._count_tokens_examples_any(inputs, self.seq_key, self.mask_key, args)

        dt = float(kw.get("dt") or 0.0)
        if dt <= 0.0:
            dt = 1e-3

        self._eval_tokens += int(tok)
        self._eval_examples += int(ex)
        self._eval_time += dt
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        trainer = self._trainer or kw.get("trainer")
        if metrics and isinstance(metrics, dict):
            for k in ("eval_accuracy", "accuracy"):
                if k in metrics:
                    self._last_eval_accuracy = float(metrics[k])
                    break

        if trainer and self._is_rank0(trainer):
            if self._eval_time > 0:
                tps = self._eval_tokens / self._eval_time
                eps = self._eval_examples / self._eval_time if self.with_examples_sec else None
                msg = f"[EVALUATION_PERFORMANCE] tokens={self._eval_tokens} time={self._eval_time:.2f}s tok/s={tps:.1f}"
                if eps is not None:
                    msg += f" ex/s={eps:.2f}"
                print(msg)
            else:
                tps = eps = 0.0

            if self._last_eval_accuracy is not None and self._wall_t0 is not None:
                wall = time.perf_counter() - self._wall_t0
                usd = self._cost_usd_for_seconds(wall)
                if usd > 0:
                    cna = self._last_eval_accuracy / usd
                    print(f"[EVALUATION_PERFORMANCE] accuracy={self._last_eval_accuracy:.4f} "
                          f"cost=${usd:.2f} accuracy_per_$={cna:.2f}")

        self._eval_tokens = self._eval_examples = 0
        self._eval_time = 0.0
        return control

class DebugCallback(TrainerCallback):
    def __init__(
        self,
        dump_all_groups: bool = True,
        show_scheduler: bool = True,
        show_param_counts: bool = True,
    ):
        self.trainer = None
        self.dump_all_groups = dump_all_groups
        self.show_scheduler = show_scheduler
        self.show_param_counts = show_param_counts

    def bind(self, trainer):
        self.trainer = trainer

    def _get_trainer(self, **kw):
        return self.trainer or kw.get("trainer", None)

    @staticmethod
    def _unwrap_optimizer(opt):
        return getattr(opt, "optimizer", opt)

    @staticmethod
    def _count_params(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def on_epoch_begin(self, args, state, control, **kw):
        trainer = self._get_trainer(**kw)
        if trainer is None:
            return control

        opt = getattr(trainer, "optimizer", None)
        wrapped = type(opt).__name__ if opt is not None else "None"
        base = self._unwrap_optimizer(opt)
        base_name = type(base).__name__ if base is not None else "None"

        if opt and hasattr(opt, "param_groups") and opt.param_groups:
            group0 = opt.param_groups[0]
            hyperparams0 = {k: v for k, v in group0.items() if k != "params"}
            lr0 = hyperparams0.get("lr")
        else:
            hyperparams0 = {}
            lr0 = None

        print(
            f"[OPTIMIZER] epoch_start={state.epoch} global_step={state.global_step} "
            f"wrapped={wrapped} base={base_name} "
            f"lr={lr0} id={id(base) if base is not None else 'NA'} "
            f"hyperparams={hyperparams0}"
        )

        if self.dump_all_groups and opt and getattr(opt, "param_groups", None):
            for i, g in enumerate(opt.param_groups):
                hp = {k: v for k, v in g.items() if k != "params"}
                print(f"[OPTIMIZER_GROUP]   group[{i}] {hp}")

        if self.show_scheduler:
            sched = getattr(trainer, "lr_scheduler", None)
            if sched is not None:
                try:
                    lrs = sched.get_last_lr()
                    print(f"[SCHEDULER]   scheduler_last_lr={lrs}")
                except Exception:
                    pass

        if self.show_param_counts:
            model = getattr(trainer, "model", None)
            if model is not None:
                total, trainable = self._count_params(model)
                print(f"[PARAMS]   params_total={total:,} params_trainable={trainable:,}")

        return control



    #def on_epoch_end(self, args, state, control, model=None, optimizer=None, **kw):
        #print(f"[DBG] epoch_end={state.epoch} global_step={state.global_step}")

    #def on_step_end(self, args, state, control, **kw):
        #print(f"[DBG] step_end global_step={state.global_step}")
