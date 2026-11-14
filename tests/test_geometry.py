import numpy as np
import pytest

from drct_ms.models.geometry import relative_position_index, shifted_window_mask


def test_relative_index_orientation():
    # Independent explicit coordinate example: horizontal and vertical offsets differ.
    index = relative_position_index(2)
    np.testing.assert_array_equal(index, [[4, 3, 1, 0], [5, 4, 2, 1], [7, 6, 4, 3], [8, 7, 5, 4]])


def test_shift_mask_boundary_connectivity():
    mask = shifted_window_mask(8, 12, 4, 2)
    assert mask.shape == (6, 16, 16)
    assert set(np.unique(mask)) == {-100.0, 0.0}
    np.testing.assert_array_equal(mask, mask.transpose(0, 2, 1))
    assert np.all(np.diagonal(mask, axis1=1, axis2=2) == 0)
    # Interior window is unrestricted, while the bottom-right wrapped window is split.
    assert np.all(mask[0] == 0)
    assert np.count_nonzero(mask[-1] == -100.0) == 192


def test_zero_shift_dynamic_mask():
    assert np.all(shifted_window_mask(32, 48, 16, 0) == 0)


@pytest.mark.parametrize("args", [(7, 8, 4, 2), (8, 8, 4, 4), (0, 8, 4, 2)])
def test_invalid_mask_geometry(args):
    with pytest.raises(ValueError):
        shifted_window_mask(*args)


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_window_size(size):
    with pytest.raises(ValueError):
        relative_position_index(size)
