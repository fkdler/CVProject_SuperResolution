"""Initial real MindSpore Cells; target runtime is MindSpore 2.7.2.

Derived from ming053l/DRCT, MIT, Copyright (c) 2024 Chia-Ming Lee.
See THIRD_PARTY_NOTICES.md. Runtime equivalence is tested only when MindSpore
is available; these Cells are not a completed DRCT-L network.
"""

import math
from mindspore import nn, ops


def window_partition(x, window_size: int):
    """BHWC -> [B*nW, window, window, C], preserving upstream token order."""
    b, h, w, c = x.shape
    if window_size <= 0 or h % window_size or w % window_size:
        raise ValueError("spatial sizes must be divisible by a positive window_size")
    x = x.reshape((b, h // window_size, window_size, w // window_size, window_size, c))
    return ops.transpose(x, (0, 1, 3, 2, 4, 5)).reshape((-1, window_size, window_size, c))


def window_reverse(windows, window_size: int, height: int, width: int):
    """Inverse of window_partition; use integer arithmetic for batch size."""
    if window_size <= 0 or height <= 0 or width <= 0 or height % window_size or width % window_size:
        raise ValueError("spatial sizes must be positive multiples of window_size")
    per_image = (height // window_size) * (width // window_size)
    if windows.shape[0] % per_image:
        raise ValueError("window count is incompatible with spatial size")
    b = windows.shape[0] // per_image
    c = windows.shape[-1]
    x = windows.reshape((b, height // window_size, width // window_size, window_size, window_size, c))
    return ops.transpose(x, (0, 1, 3, 2, 4, 5)).reshape((b, height, width, c))


class PatchEmbed(nn.Cell):
    """NCHW -> BLC. In this model patch_size=1: no convolution or downsampling."""

    def __init__(self, embed_dim: int, patch_norm: bool = False):
        super().__init__()
        # PyTorch LayerNorm defaults to 1e-5; MindSpore defaults to 1e-7.
        self.norm = nn.LayerNorm((embed_dim,), epsilon=1e-5) if patch_norm else None

    def construct(self, x):
        b, c, h, w = x.shape
        x = ops.transpose(x.reshape((b, c, h * w)), (0, 2, 1))
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Cell):
    """BLC -> NCHW, with channels inferred (RDG has growing channel widths)."""

    def construct(self, x, x_size):
        b, _, c = x.shape
        return ops.transpose(x, (0, 2, 1)).reshape((b, c, x_size[0], x_size[1]))


class ExactGELU(nn.Cell):
    """erf definition of PyTorch GELU, avoiding backend approximation defaults."""

    def construct(self, x):
        return 0.5 * x * (1.0 + ops.erf(x / math.sqrt(2.0)))


class Mlp(nn.Cell):
    """Linear -> exact GELU -> dropout -> linear -> dropout.

    Explicitly use the erf definition instead of the default approximation. Shared weights
    are loaded for comparisons. Full-network initialization remains pending.
    """

    def __init__(self, in_features: int, hidden_features=None, out_features=None, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Dense(in_features, hidden_features or in_features)
        self.act = ExactGELU()
        self.fc2 = nn.Dense(hidden_features or in_features, out_features or in_features)
        self.drop = nn.Dropout(p=drop)

    def construct(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class PixelShuffle(nn.Cell):
    """PyTorch channel ordering via reshape/transpose, avoiding DepthToSpace mode ambiguity."""

    def __init__(self, scale: int):
        super().__init__()
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.scale = scale

    def construct(self, x):
        b, channels, h, w = x.shape
        r = self.scale
        if channels % (r * r):
            raise ValueError("channels must be divisible by scale squared")
        c = channels // (r * r)
        x = x.reshape((b, c, r, r, h, w))
        return ops.transpose(x, (0, 1, 4, 2, 5, 3)).reshape((b, c, h * r, w * r))
