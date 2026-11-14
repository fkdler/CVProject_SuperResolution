"""Ascend-only structural and operator checks; no inference frameworks on local hosts."""

import json
from pathlib import Path

import numpy as np
import pytest

ms = pytest.importorskip("mindspore")
from mindspore import nn

from drct_ms.checkpoint import source_parameter_name
from drct_ms.models.drct import DRCT, DropPath, SwinTransformerBlock, WindowAttention
from drct_ms.models.convolution import MatmulConv2d
from drct_ms.models.geometry import shifted_window_mask
from drct_ms.models.primitives import cyclic_shift_bhwc

pytestmark = pytest.mark.mindspore
ROOT = Path(__file__).resolve().parents[1]


def test_full_large_parameter_shapes():
    config = json.loads((ROOT / "configs/drct_l_x4.json").read_text())["network_g"]
    audit = json.loads((ROOT / "docs/verification/reference_audit.json").read_text())["models"]["large"]
    net = DRCT(**config)
    actual = {source_parameter_name(k): list(v.shape) for k, v in net.parameters_dict().items()}
    expected = {k: v for k, v in audit["state_dict"].items()
                if not k.endswith("attn_mask") and not k.endswith("relative_position_index")}
    assert actual == expected
    assert sum(p.size for p in net.get_parameters()) == 27580719
    assert len(net.layers) == 12
    for group in net.layers:
        blocks = [getattr(group, f"swin{i}") for i in range(1, 6)]
        assert [b.dim for b in blocks] == [180, 212, 244, 276, 308]
        assert [b.num_heads for b in blocks] == [6, 4, 2, 6, 4]
        assert [b.shift_size for b in blocks] == [0, 8, 0, 8, 0]
        assert [b.mlp_ratio for b in blocks] == [2, 2, 2, 1, 1]
        assert len({b.drop_path.drop_prob for b in blocks}) == 1


@pytest.mark.parametrize("shift", [0, 8])
def test_changed_resolution_mask(shift):
    block = SwinTransformerBlock(180, (64, 64), 6, window_size=16, shift_size=shift)
    actual = block.calculate_mask((32, 48)).asnumpy()
    np.testing.assert_array_equal(actual, shifted_window_mask(32, 48, 16, shift))


@pytest.mark.parametrize("shift", [-8, 8, 0])
def test_cyclic_shift_exact_order(shift):
    x = np.arange(2 * 32 * 48 * 3, dtype=np.float32).reshape(2, 32, 48, 3)
    actual = cyclic_shift_bhwc(ms.Tensor(x), shift, shift).asnumpy()
    np.testing.assert_array_equal(actual, np.roll(x, (shift, shift), (1, 2)))


def test_attention_uniform_values_and_mask():
    attention = WindowAttention(12, 2, 3)
    attention.set_train(False)
    # Q=K=0, V=identity, projection=identity, no positional bias.
    weight = np.zeros((36, 12), dtype=np.float32)
    weight[24:] = np.eye(12, dtype=np.float32)
    attention.qkv.weight.set_data(ms.Tensor(weight))
    attention.qkv.bias.set_data(ms.Tensor(np.zeros(36, dtype=np.float32)))
    attention.proj.weight.set_data(ms.Tensor(np.eye(12, dtype=np.float32)))
    attention.proj.bias.set_data(ms.Tensor(np.zeros(12, dtype=np.float32)))
    attention.relative_position_bias_table.set_data(ms.Tensor(np.zeros((9, 3), dtype=np.float32)))
    x = np.arange(2 * 4 * 12, dtype=np.float32).reshape(2, 4, 12)
    mask = np.zeros((2, 4, 4), dtype=np.float32)
    mask[:, :, 2:] = -100.0
    result = attention(ms.Tensor(x), ms.Tensor(mask)).asnumpy()
    expected = np.broadcast_to(x[:, :2].mean(axis=1, keepdims=True), x.shape)
    np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-6)


def test_drop_path_eval_and_per_sample_broadcast():
    layer = DropPath(0.5)
    x = ms.Tensor(np.ones((64, 8, 12), dtype=np.float32))
    layer.set_train(False)
    np.testing.assert_array_equal(layer(x).asnumpy(), x.asnumpy())
    layer.set_train(True)
    actual = layer(x).asnumpy()
    assert set(np.unique(actual)) <= {0.0, 2.0}
    np.testing.assert_array_equal(actual, np.broadcast_to(actual[:, :1, :1], actual.shape))


class ShiftedSwin(nn.Cell):
    def __init__(self):
        super().__init__()
        self.block = SwinTransformerBlock(180, (64, 64), 6, window_size=16, shift_size=8)

    def construct(self, x):
        return self.block(x, (32, 48))


def test_swin_finite_input_gradient():
    # Component gradient smoke only; no optimizer or training loop is launched.
    net = ShiftedSwin()
    net.set_train(False)
    x = ms.Tensor(np.random.default_rng(3).normal(0, 0.1, (1, 32 * 48, 180)).astype(np.float32))
    value = net(x)
    assert value.shape == x.shape
    def loss(a):
        return net(a).sum()
    gradient = ms.grad(loss)(x).asnumpy()
    assert np.isfinite(gradient).all()
    assert np.count_nonzero(gradient) > 0


@pytest.mark.parametrize("kernel", [1, 3])
def test_equivalent_convolution_output_and_gradient(kernel):
    rng = np.random.default_rng(18)
    x = rng.normal(0, 0.2, (2, 3, 5, 7)).astype(np.float32)
    weight = rng.normal(0, 0.1, (4, 3, kernel, kernel)).astype(np.float32)
    bias = rng.normal(0, 0.1, 4).astype(np.float32)
    net = MatmulConv2d(3, 4, kernel)
    net.weight.set_data(ms.Tensor(weight))
    net.bias.set_data(ms.Tensor(bias))
    padding = kernel // 2
    padded = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    expected = np.broadcast_to(bias.reshape(1, 4, 1, 1), (2, 4, 5, 7)).astype(np.float64).copy()
    expected_grad = np.zeros_like(padded)
    for row in range(kernel):
        for col in range(kernel):
            expected += np.einsum("bchw,oc->bohw", padded[:, :, row:row+5, col:col+7], weight[:, :, row, col])
            expected_grad[:, :, row:row+5, col:col+7] += weight[:, :, row, col].sum(axis=0).reshape(1, 3, 1, 1)
    if padding:
        expected_grad = expected_grad[:, :, padding:-padding, padding:-padding]
    np.testing.assert_allclose(net(ms.Tensor(x)).asnumpy(), expected, atol=2e-6, rtol=2e-5)
    def loss(a):
        return net(a).sum()
    np.testing.assert_allclose(ms.grad(loss)(ms.Tensor(x)).asnumpy(), expected_grad, atol=2e-6, rtol=2e-5)
