"""Cloud-only FP32 pretrained continuation, audited updates, EMA and full-state resume."""

from .runtime import require_cloud_workspace

require_cloud_workspace()

import json
import os
from pathlib import Path
import time

import mindspore as ms
from mindspore import nn, ops
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.ops import functional as F
import numpy as np

from .evaluation import file_sha256, metrics, to_bgr_uint8


_reference_adam_update = ops.MultitypeFuncGraph("drct_reference_adam_update")


@_reference_adam_update.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                                 "Tensor", "Tensor", "Tensor", "Tensor")
def _apply_reference_adam(primitive, beta1_power, beta2_power, lr, beta1, beta2, epsilon, parameter, m, v, gradient):
    return ops.depend(True, primitive(parameter, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, gradient))


class ReferenceAdam(nn.Adam):
    """Dense FP32 Adam with PyTorch's epsilon after bias correction.

    MindSpore 2.7.2's fused Adam kernel divides by sqrt(v) + epsilon and scales
    the numerator by sqrt(1-beta2**t). Pass epsilon*sqrt(1-beta2**t) to obtain
    sqrt(v/(1-beta2**t)) + epsilon, as in the official PyTorch optimizer.
    Reuses the pinned nn.Adam state/counters/schedule, overriding only its dense
    update dispatch. Sparse/groups/offload/AMSGrad are outside this adapter.
    """
    def _apply_adam(self, params, beta1_power, beta2_power, moment1, moment2, lr, gradients):
        epsilon = self.eps * ops.sqrt(1.0 - beta2_power)
        return self.hyper_map(F.partial(_reference_adam_update, self.opt, beta1_power, beta2_power,
                                        lr, self.beta1, self.beta2, epsilon),
                              params, moment1, moment2, gradients)


class MultiStepLR(LearningRateSchedule):
    """Step is the number of completed updates; milestone decay precedes the next update."""
    def __init__(self, initial, milestones, gamma):
        super().__init__()
        if sorted(set(milestones)) != milestones or any(m <= 0 for m in milestones):
            raise ValueError("Milestones must be positive and strictly increasing")
        self.initial = ms.Tensor(initial, ms.float32)
        self.milestones = tuple(milestones)
        self.gamma = ms.Tensor(gamma, ms.float32)

    def construct(self, global_step):
        value = self.initial
        for milestone in self.milestones:
            value = value * ops.where(global_step >= milestone, self.gamma, ops.ones_like(self.gamma))
        return value


class Trainer:
    def __init__(self, model, config):
        self.model, self.config = model, config
        self.schedule = MultiStepLR(config["learning_rate"], config["milestones"], config["gamma"])
        if config["weight_decay"] != 0.0:
            raise ValueError("The reference Adam adapter currently implements the released zero-weight-decay recipe")
        self.optimizer = ReferenceAdam(model.trainable_params(), learning_rate=self.schedule,
                                 beta1=config["betas"][0], beta2=config["betas"][1],
                                 eps=config["eps"], weight_decay=config["weight_decay"])
        self.ema = {name: ms.Parameter(ms.Tensor(p.asnumpy()), name=f"ema_{i}", requires_grad=False)
                    for i, (name, p) in enumerate(model.parameters_dict().items())}
        self.step = 0

        def forward_loss(lr, hr):
            return ops.mean(ops.abs(self.model(lr) - hr))

        self.grad_fn = ms.value_and_grad(forward_loss, grad_position=None, weights=self.optimizer.parameters)

    def update(self, lr, hr):
        self.model.set_train(True)
        before = {name: p.asnumpy().copy() for name, p in self.model.parameters_dict().items()}
        rate = float(self.schedule(ms.Tensor(self.step, ms.int32)).asnumpy().reshape(-1)[0])
        ms.runtime.synchronize()
        started = time.perf_counter()
        loss, grads = self.grad_fn(ms.Tensor(lr, ms.float32), ms.Tensor(hr, ms.float32))
        if len(grads) != len(self.optimizer.parameters):
            raise AssertionError("Missing parameter gradients")
        for parameter, grad in zip(self.optimizer.parameters, grads):
            if grad.shape != parameter.shape or grad.dtype != ms.float32:
                raise AssertionError(f"Gradient layout/dtype failed: {parameter.name}")
        finite = bool(ops.all(ops.stack([ops.all(ops.isfinite(g)) for g in grads])).asnumpy())
        loss_value = float(loss.asnumpy())
        if not finite or not np.isfinite(loss_value):
            raise FloatingPointError("Nonfinite loss/gradient; optimizer was not called")
        self.optimizer(grads)
        decay = self.config["ema_decay"]
        for name, parameter in self.model.parameters_dict().items():
            ops.assign(self.ema[name], self.ema[name] * decay + parameter * (1.0 - decay))
        ms.runtime.synchronize()
        elapsed = time.perf_counter() - started
        self.step += 1
        if int(self.optimizer.global_step.asnumpy().reshape(-1)[0]) != self.step:
            raise AssertionError("Optimizer and sampler step counters diverged")
        changed, maximum = 0, 0.0
        for name, parameter in self.model.parameters_dict().items():
            value = parameter.asnumpy()
            if not np.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite updated parameter: {name}")
            delta = float(np.max(np.abs(value - before[name])))
            changed += int(delta > 0)
            maximum = max(maximum, delta)
        if changed == 0:
            raise AssertionError("No parameters changed")
        return {"step": self.step, "loss_l1": loss_value, "learning_rate": rate,
                "all_gradients_finite": finite, "gradient_tensors": len(grads),
                "changed_parameter_tensors": changed, "max_parameter_abs_update": maximum,
                "audited_update_seconds": elapsed,
                "peak_allocated_bytes": ms.runtime.max_memory_allocated(),
                "peak_reserved_bytes": ms.runtime.max_memory_reserved()}

    def state_objects(self):
        model = self.model.parameters_dict()
        model_ids = {id(p) for p in model.values()}
        state = {"model." + name: p for name, p in model.items()}
        state.update({"optimizer." + name: p for name, p in self.optimizer.parameters_dict().items()
                      if id(p) not in model_ids})
        state.update({"ema." + name: p for name, p in self.ema.items()})
        return state

    def save(self, path, identity):
        path = Path(path)
        if path.exists() or path.with_suffix(".json").exists():
            raise FileExistsError("Training checkpoints are immutable; choose a new step/path")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + ".partial.ckpt")
        state = self.state_objects()
        ms.save_checkpoint([{"name": name, "data": p} for name, p in state.items()], str(temporary))
        os.replace(temporary, path)
        metadata = {"format": 1, "completed_steps": self.step, "next_sample": self.step * self.config["batch_size"],
                    "config": self.config, "identity": identity, "checkpoint_sha256": file_sha256(path),
                    "state_tensors": len(state), "rng_resume": "sampler exact; stochastic DropPath stream not guaranteed"}
        sidecar = path.with_suffix(".json")
        temporary_json = sidecar.with_suffix(".partial.json")
        temporary_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_json, sidecar)
        return metadata

    def restore(self, path, identity):
        path = Path(path)
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (metadata["format"] != 1 or metadata["config"] != self.config or metadata["identity"] != identity
                or metadata["checkpoint_sha256"] != file_sha256(path)
                or metadata["next_sample"] != metadata["completed_steps"] * self.config["batch_size"]):
            raise ValueError("Resume identity/config/data/checkpoint integrity mismatch")
        supplied, required = ms.load_checkpoint(str(path)), self.state_objects()
        if set(supplied) != set(required):
            raise ValueError("Resume must cover every model, optimizer and EMA tensor exactly")
        # Validate all entries before mutating any state. No suffix matching or casts.
        for name, parameter in required.items():
            value = supplied[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError(f"Resume shape/dtype mismatch: {name}")
            if not np.isfinite(value.asnumpy()).all():
                raise ValueError(f"Nonfinite checkpoint state: {name}")
        for name, parameter in required.items():
            parameter.set_data(supplied[name])
        for name, parameter in required.items():
            if not np.array_equal(parameter.asnumpy(), supplied[name].asnumpy()):
                raise AssertionError(f"Restored state differs: {name}")
        self.step = metadata["completed_steps"]
        if int(self.optimizer.global_step.asnumpy().reshape(-1)[0]) != self.step:
            raise AssertionError("Restored optimizer step mismatch")
        return {"state_tensors_exactly_restored": len(required), "completed_steps": self.step,
                "next_sample": metadata["next_sample"], "checkpoint_sha256": metadata["checkpoint_sha256"]}

    def validate(self, dataset, use_ema=False):
        saved = None
        if use_ema:
            saved = {name: p.asnumpy().copy() for name, p in self.model.parameters_dict().items()}
            for name, p in self.model.parameters_dict().items():
                p.set_data(ms.Tensor(self.ema[name].asnumpy()))
        self.model.set_train(False)
        rows = []
        try:
            for i in range(len(dataset.pairs)):
                lr, hr, audit = dataset.sample(i)
                output = self.model(ms.Tensor(lr[None], ms.float32)).asnumpy()
                target = to_bgr_uint8(hr[None])
                scores = metrics(to_bgr_uint8(output), target)
                if not all(np.isfinite(v) for v in scores.values()):
                    raise FloatingPointError("Nonfinite validation metrics")
                rows.append({"id": audit["id"], **scores, "output_sha256": __import__("hashlib").sha256(output.tobytes()).hexdigest()})
        finally:
            if saved is not None:
                for name, p in self.model.parameters_dict().items():
                    p.set_data(ms.Tensor(saved[name]))
            self.model.set_train(True)
        return {"step": self.step, "weights": "ema" if use_ema else "live", "scope": "fixed center LR patches",
                "lr_patch_size": dataset.patch, "protocol": "uint8 BT.601 Y / HR shave4",
                "per_image": rows, "mean_psnr": float(np.mean([r["psnr"] for r in rows])),
                "mean_ssim": float(np.mean([r["ssim"] for r in rows]))}
