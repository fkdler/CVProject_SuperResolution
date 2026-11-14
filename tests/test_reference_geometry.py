import numpy as np
import pytest

torch = pytest.importorskip("torch")
from drct_ms.models.geometry import relative_position_index, shifted_window_mask
from tools.reference_loader import load_reference

pytestmark = pytest.mark.reference


@pytest.fixture(scope="module")
def ref():
    return load_reference()


@pytest.mark.parametrize("window", [2, 4, 16])
def test_index_matches_official(ref, window):
    attention = ref["WindowAttention"](12, (window, window), 3)
    np.testing.assert_array_equal(relative_position_index(window), attention.relative_position_index.numpy())


@pytest.mark.parametrize("height,width,window,shift", [(32, 48, 16, 8), (32, 48, 16, 0), (8, 12, 4, 2)])
def test_mask_matches_official(ref, height, width, window, shift):
    block = ref["SwinTransformerBlock"](12, (64, 64), 3, window_size=window, shift_size=shift)
    expected = block.calculate_mask((height, width)).numpy()
    np.testing.assert_array_equal(shifted_window_mask(height, width, window, shift), expected)
