"""Cloud-only functional training checks, separate from full DRCT-L resource smoke."""

import json

import mindspore as ms
from mindspore import nn
import numpy as np
import pytest

from tools.cloud_guard import ROOT
from drct_ms.training import MultiStepLR, Trainer
from drct_ms.training_data import PairedDIV2K, paired_crop


def configuration():
    return json.loads((ROOT / "configs/train_smoke.json").read_text(encoding="utf-8"))


def test_multistep_boundaries():
    schedule = MultiStepLR(1e-5, [2, 4], 0.5)
    actual = [float(schedule(ms.Tensor(i, ms.int32)).asnumpy()) for i in range(6)]
    np.testing.assert_allclose(actual, [1e-5, 1e-5, 5e-6, 5e-6, 2.5e-6, 2.5e-6], rtol=1e-7)


def test_paired_crop_and_augmentation():
    lr = np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3)
    hr = np.repeat(np.repeat(lr, 4, axis=0), 4, axis=1)
    for seed in range(16):
        a, b, audit = paired_crop(lr, hr, 16, np.random.default_rng(seed))
        np.testing.assert_array_equal(b, np.repeat(np.repeat(a, 4, axis=1), 4, axis=2))
        assert 0 <= audit["lr_top"] <= 16 and 0 <= audit["lr_left"] <= 32


def test_div2k_split_and_data_position():
    train = PairedDIV2K(ROOT / "data/DIV2K_smoke", "train")
    valid = PairedDIV2K(ROOT / "data/DIV2K_smoke", "valid")
    assert {p["id"] for p in train.pairs}.isdisjoint({p["id"] for p in valid.pairs})
    restored = PairedDIV2K(ROOT / "data/DIV2K_smoke", "train")
    for index in (0, 1, 4, 7):
        a, b, metadata = train.sample(index)
        c, d, other = restored.sample(index)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
        assert metadata == other
    assert len({train.sample(i)[2]["id"] for i in range(4)}) == 4


def test_adam_ema_and_full_state_resume(tmp_path):
    config = configuration()
    model = nn.Dense(2, 1, weight_init="ones", bias_init="zeros")
    trainer = Trainer(model, config)
    x, y = np.array([[1.0, 2.0]], np.float32), np.array([[0.0]], np.float32)
    before = {name: p.asnumpy().copy() for name, p in model.parameters_dict().items()}
    row = trainer.update(x, y)
    assert row["all_gradients_finite"] and row["changed_parameter_tensors"] == 2
    # Positive L1 gradients [1,2] and [1]: first Adam update has known closed form.
    for name, parameter in model.parameters_dict().items():
        gradient = np.array([[1.0, 2.0]], np.float32) if name == "weight" else np.array([1.0], np.float32)
        expected = before[name] - config["learning_rate"] * gradient / (np.abs(gradient) + config["eps"])
        np.testing.assert_allclose(parameter.asnumpy(), expected, atol=2e-7, rtol=0)
        expected_ema = before[name] * config["ema_decay"] + parameter.asnumpy() * (1-config["ema_decay"])
        np.testing.assert_allclose(trainer.ema[name].asnumpy(), expected_ema, atol=1e-7, rtol=0)
    path = tmp_path / "resume.ckpt"
    identity = {"dataset": "training-test"}
    trainer.save(path, identity)
    resumed = Trainer(nn.Dense(2, 1, weight_init="zeros", bias_init="ones"), config)
    audit = resumed.restore(path, identity)
    assert audit["completed_steps"] == 1
    assert audit["state_tensors_exactly_restored"] > 4
    for name, parameter in trainer.state_objects().items():
        np.testing.assert_array_equal(parameter.asnumpy(), resumed.state_objects()[name].asnumpy())
    trainer.update(x, y)
    resumed.update(x, y)
    for name, parameter in trainer.state_objects().items():
        np.testing.assert_array_equal(parameter.asnumpy(), resumed.state_objects()[name].asnumpy())
    with pytest.raises(ValueError, match="identity"):
        resumed.restore(path, {"dataset": "wrong-data"})


def test_adam_epsilon_with_tiny_gradients():
    config = configuration()
    model = nn.Dense(2, 1, weight_init="zeros", bias_init="ones")
    trainer = Trainer(model, config)
    expected, m, v = (np.zeros((1, 2), np.float64) for _ in range(3))
    beta1, beta2 = config["betas"]
    for step, gradient in enumerate(([[1e-9, 1e-7]], [[-2e-8, 3e-9]], [[0.0, 1e-10]]), 1):
        gradient = np.array(gradient, np.float32)
        trainer.update(gradient, np.zeros((1, 1), np.float32))
        m = beta1 * m + (1-beta1) * gradient.astype(np.float64)
        v = beta2 * v + (1-beta2) * gradient.astype(np.float64)**2
        expected -= config["learning_rate"] * (m / (1-beta1**step)) / (np.sqrt(v / (1-beta2**step)) + config["eps"])
        np.testing.assert_allclose(model.weight.asnumpy(), expected, atol=2e-11, rtol=2e-5)


def test_optimizer_uses_milestone_before_next_update():
    config = configuration()
    config["milestones"] = [1]
    model = nn.Dense(2, 1, weight_init="zeros", bias_init="ones")
    trainer = Trainer(model, config)
    x, y = np.ones((1, 2), np.float32), np.zeros((1, 1), np.float32)
    trainer.update(x, y)
    second = trainer.update(x, y)
    np.testing.assert_allclose(second["learning_rate"], 5e-6, rtol=1e-7)
    np.testing.assert_allclose(model.weight.asnumpy(), -1.5e-5, atol=2e-10, rtol=0)
