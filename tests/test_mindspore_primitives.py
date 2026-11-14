"""Compare MindSpore Cells to official-PyTorch fixtures without installing torch."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ms = pytest.importorskip("mindspore")
from drct_ms.models.primitives import Mlp, PatchEmbed, PatchUnEmbed, PixelShuffle, window_partition, window_reverse

pytestmark = pytest.mark.mindspore


@pytest.fixture(scope="module")
def golden():
    # Device is configured by the caller/environment; never silently switch Ascend to CPU.
    assert ms.__version__ == "2.7.2", f"Target runtime is MindSpore 2.7.2, got {ms.__version__}"
    ms.set_context(mode=ms.PYNATIVE_MODE)
    path = Path(__file__).parent / "fixtures/primitive_reference.npz"
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["artifact_sha256"]
    with np.load(path, allow_pickle=False) as archive:
        yield dict(archive)


def test_window_order_and_roundtrip(golden):
    x = golden["window_x"]
    windows = window_partition(ms.Tensor(x), 16)
    np.testing.assert_array_equal(windows.asnumpy(), golden["window_y"])
    np.testing.assert_array_equal(window_reverse(windows, 16, 32, 48).asnumpy(), x)


def test_patch_layout_and_layernorm(golden):
    # A near-constant input makes an incorrect LayerNorm epsilon observable.
    x = golden["patch_x"]
    bare = PatchEmbed(12, patch_norm=False)(ms.Tensor(x))
    np.testing.assert_array_equal(PatchUnEmbed()(bare, (8, 12)).asnumpy(), x)
    net = PatchEmbed(12, patch_norm=True)
    net.norm.gamma.set_data(ms.Tensor(golden["patch_gamma"]))
    net.norm.beta.set_data(ms.Tensor(golden["patch_beta"]))
    np.testing.assert_allclose(net(ms.Tensor(x)).asnumpy(), golden["patch_y"], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("scale", [2, 3])
def test_pixel_shuffle_channel_order(golden, scale):
    x = golden[f"shuffle{scale}_x"]
    np.testing.assert_array_equal(PixelShuffle(scale)(ms.Tensor(x)).asnumpy(), golden[f"shuffle{scale}_y"])


def test_mlp_shared_weights_and_input_gradients(golden):
    net = Mlp(12, 24, 12, drop=0)
    net.set_train(False)
    for layer in ("fc1", "fc2"):
        for field in ("weight", "bias"):
            getattr(getattr(net, layer), field).set_data(ms.Tensor(golden[f"mlp_{layer}.{field}"]))
    x = ms.Tensor(golden["mlp_x"])
    np.testing.assert_allclose(net(x).asnumpy(), golden["mlp_y"], atol=2e-6, rtol=2e-5)
    gradient = ms.grad(lambda a: net(a).sum())(x).asnumpy()
    np.testing.assert_allclose(gradient, golden["mlp_input_grad"], atol=3e-6, rtol=3e-5)
