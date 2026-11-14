"""Faithful DRCT network definition for MindSpore 2.7.2 / Ascend.

Derived from ming053l/DRCT c0a8b51, MIT, Copyright (c) 2024 Chia-Ming Lee.
See THIRD_PARTY_NOTICES.md. This implementation requires cloud validation.
"""

import math

import mindspore as ms
import numpy as np
from mindspore import nn, ops, Tensor, Parameter

from .geometry import relative_position_index, shifted_window_mask
from .convolution import MatmulConv2d
from .primitives import (Mlp, PatchEmbed, PatchUnEmbed, PixelShuffle, window_partition,
                         window_reverse, cyclic_shift_bhwc)


def conv2d(in_channels, out_channels, kernel_size, backend="matmul"):
    if backend == "matmul":
        return MatmulConv2d(in_channels, out_channels, kernel_size)
    if backend != "native":
        raise ValueError("Unknown convolution backend")
    # Both padding and bias differ from MindSpore defaults.
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, pad_mode="pad",
                     padding=kernel_size // 2, has_bias=True)


class DropPath(nn.Cell):
    """Per-sample stochastic depth, invoked independently on both residual branches."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        if not 0 <= drop_prob < 1:
            raise ValueError("drop_prob must be in [0, 1)")
        self.drop_prob = float(drop_prob)
        self.uniform = ops.UniformReal()

    def construct(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (len(x.shape) - 1)
        mask = ops.floor(ops.cast(self.uniform(shape), x.dtype) + keep)
        return (x / keep) * mask


class WindowAttention(nn.Cell):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if isinstance(window_size, (tuple, list)):
            if len(window_size) != 2 or window_size[0] != window_size[1]:
                raise ValueError("This DRCT port supports the official square windows")
            window_size = window_size[0]
        if num_heads <= 0 or dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.window_size = int(window_size)
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.relative_position_index = Tensor(relative_position_index(self.window_size), ms.int32)
        # Checkpoint loading overwrites all learned values. See initialization note in docs.
        self.relative_position_bias_table = Parameter(
            ops.zeros(((2 * self.window_size - 1) ** 2, num_heads), ms.float32),
            name="relative_position_bias_table")
        self.qkv = nn.Dense(dim, 3 * dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x, mask=None):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape((b, n, 3, self.num_heads, c // self.num_heads))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attention = ops.matmul(q * self.scale, ops.transpose(k, (0, 1, 3, 2)))
        bias = ops.gather(self.relative_position_bias_table, self.relative_position_index.reshape((-1,)), 0)
        bias = ops.transpose(bias.reshape((n, n, self.num_heads)), (2, 0, 1))
        attention = attention + ops.expand_dims(bias, 0)
        if mask is not None:
            nw = mask.shape[0]
            attention = attention.reshape((b // nw, nw, self.num_heads, n, n))
            attention = attention + mask.reshape((1, nw, 1, n, n))
            attention = attention.reshape((b, self.num_heads, n, n))
        attention = self.attn_drop(self.softmax(attention))
        x = ops.transpose(ops.matmul(attention, v), (0, 2, 1, 3)).reshape((b, n, c))
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Cell):
    def __init__(self, dim, input_resolution, num_heads, window_size=16, shift_size=0,
                 mlp_ratio=2.0, qkv_bias=True, qk_scale=None, drop=0.0,
                 attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = tuple(input_resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.window_size = min(self.input_resolution)
            self.shift_size = 0
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError("shift must be in [0, window_size)")
        self.norm1 = nn.LayerNorm((dim,), epsilon=1e-5)
        self.attn = WindowAttention(dim, self.window_size, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm((dim,), epsilon=1e-5)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.attn_mask = None
        if self.shift_size:
            self.attn_mask = Tensor(shifted_window_mask(*self.input_resolution, self.window_size, self.shift_size))

    def calculate_mask(self, x_size):
        """Device tensor geometry supports changed static shapes without NumPy in construct."""
        h, w = x_size
        ws, shift = self.window_size, self.shift_size
        if shift == 0:
            return ops.zeros(((h // ws) * (w // ws), ws * ws, ws * ws), ms.float32)
        rows = ops.arange(0, h, 1, dtype=ms.int32)
        cols = ops.arange(0, w, 1, dtype=ms.int32)
        rows = ops.cast(rows >= h - ws, ms.int32) + ops.cast(rows >= h - shift, ms.int32)
        cols = ops.cast(cols >= w - ws, ms.int32) + ops.cast(cols >= w - shift, ms.int32)
        labels = (rows.reshape((h, 1)) * 3 + cols.reshape((1, w))).reshape((1, h, w, 1))
        windows = window_partition(labels, ws).reshape((-1, ws * ws))
        delta = ops.expand_dims(windows, 1) - ops.expand_dims(windows, 2)
        return ops.cast(delta != 0, ms.float32) * -100.0

    def construct(self, x, x_size):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x).reshape((b, h, w, c))
        if self.shift_size:
            x = cyclic_shift_bhwc(x, -self.shift_size, -self.shift_size)
        windows = window_partition(x, self.window_size).reshape((-1, self.window_size ** 2, c))
        if x_size == self.input_resolution:
            mask = self.attn_mask
        else:
            mask = self.calculate_mask(x_size)
        windows = self.attn(windows, mask).reshape((-1, self.window_size, self.window_size, c))
        x = window_reverse(windows, self.window_size, h, w)
        if self.shift_size:
            x = cyclic_shift_bhwc(x, self.shift_size, self.shift_size)
        x = shortcut + self.drop_path(x.reshape((b, h * w, c)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class RDG(nn.Cell):
    def __init__(self, dim, input_resolution, num_heads, window_size=16, mlp_ratio=2.0,
                 qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, gc=32,
                 conv_backend="matmul"):
        super().__init__()
        # Keep original names for strict checkpoint name/shape auditing.
        for i in range(5):
            channels = dim + i * gc
            heads = num_heads if i == 0 else num_heads - channels % num_heads
            block = SwinTransformerBlock(
                channels, input_resolution, heads, window_size, 0 if i % 2 == 0 else window_size // 2,
                mlp_ratio if i < 3 else 1, qkv_bias, qk_scale, drop, attn_drop, drop_path)
            setattr(self, f"swin{i + 1}", block)
            setattr(self, f"adjust{i + 1}", conv2d(channels, gc if i < 4 else dim, 1, conv_backend))
        self.lrelu = nn.LeakyReLU(alpha=0.2)
        self.pe = PatchEmbed(dim, patch_norm=False)
        self.pue = PatchUnEmbed()

    def construct(self, x, x_size):
        x1 = self.pe(self.lrelu(self.adjust1(self.pue(self.swin1(x, x_size), x_size))))
        x2 = self.pe(self.lrelu(self.adjust2(self.pue(self.swin2(ops.concat((x, x1), -1), x_size), x_size))))
        x3 = self.pe(self.lrelu(self.adjust3(self.pue(self.swin3(ops.concat((x, x1, x2), -1), x_size), x_size))))
        x4 = self.pe(self.lrelu(self.adjust4(self.pue(self.swin4(ops.concat((x, x1, x2, x3), -1), x_size), x_size))))
        x5 = self.pe(self.adjust5(self.pue(self.swin5(ops.concat((x, x1, x2, x3, x4), -1), x_size), x_size)))
        return x5 * 0.2 + x


class Upsample(nn.SequentialCell):
    def __init__(self, scale, num_feat=64, conv_backend="matmul"):
        cells = []
        if scale > 0 and scale & (scale - 1) == 0:
            for _ in range(int(math.log2(scale))):
                cells.extend([conv2d(num_feat, 4 * num_feat, 3, conv_backend), PixelShuffle(2)])
        elif scale == 3:
            cells.extend([conv2d(num_feat, 9 * num_feat, 3, conv_backend), PixelShuffle(3)])
        else:
            raise ValueError("Unsupported upsampling scale")
        super().__init__(cells)


class DRCT(nn.Cell):
    """Complete official computation graph; pretrained-only initialization policy.

    Random initialization is NOT a reproduction of the official training recipe.
    Conversion and inference runners require a completely loaded checkpoint.
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=3, embed_dim=180,
                 depths=(6,) * 12, num_heads=(6,) * 12, window_size=16, gc=32,
                 mlp_ratio=2.0, qkv_bias=True, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0.0, drop_path_rate=0.1, ape=False, patch_norm=True,
                 upscale=4, img_range=1.0, upsampler="pixelshuffle", resi_connection="1conv",
                 compress_ratio=3, squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5,
                 conv_backend="matmul"):
        super().__init__()
        if patch_size != 1 or ape:
            raise ValueError("Official DRCT-L baseline requires patch_size=1 and ape=False")
        if upsampler != "pixelshuffle" or resi_connection not in {"1conv", "identity"}:
            raise ValueError("Unsupported reconstruction configuration")
        if not depths or len(depths) != len(num_heads) or any(d <= 0 for d in depths):
            raise ValueError("depths and num_heads must have matching positive lengths")
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        if any(s % window_size for s in self.img_size):
            raise ValueError("Nominal resolution must be divisible by window_size")
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.upscale = upscale
        self.in_chans = in_chans
        self.img_range = float(img_range)
        self.conv_backend = conv_backend
        if self.img_range <= 0:
            raise ValueError("img_range must be positive")
        mean = (0.4488, 0.4371, 0.4040) if in_chans == 3 else (0.0,) * in_chans
        self.mean = Tensor(np.asarray(mean, dtype=np.float32).reshape(1, in_chans, 1, 1))
        self.conv_first = conv2d(in_chans, embed_dim, 3, conv_backend)
        self.patch_embed = PatchEmbed(embed_dim, patch_norm=patch_norm)
        self.patch_unembed = PatchUnEmbed()
        self.pos_drop = nn.Dropout(p=drop_rate)
        # float32 linspace values are audit-visible; random masks are not shared across frameworks.
        rates = np.linspace(0, drop_path_rate, sum(depths), dtype=np.float32)
        layers, offset = [], 0
        for depth, heads in zip(depths, num_heads):
            layers.append(RDG(embed_dim, self.img_size, heads, window_size, mlp_ratio, qkv_bias,
                              qk_scale, drop_rate, attn_drop_rate, float(rates[offset]), gc, conv_backend))
            offset += depth
        self.layers = nn.CellList(layers)
        self.norm = nn.LayerNorm((embed_dim,), epsilon=1e-5)
        self.conv_after_body = conv2d(embed_dim, embed_dim, 3, conv_backend) if resi_connection == "1conv" else nn.Identity()
        self.conv_before_upsample = nn.SequentialCell([conv2d(embed_dim, 64, 3, conv_backend), nn.LeakyReLU(alpha=0.01)])
        self.upsample = Upsample(upscale, 64, conv_backend)
        self.conv_last = conv2d(64, in_chans, 3, conv_backend)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.pos_drop(self.patch_embed(x))
        for layer in self.layers:
            x = layer(x, x_size)
        return self.patch_unembed(self.norm(x), x_size)

    def construct(self, x):
        mean = ops.cast(self.mean, x.dtype)
        x = self.conv_first((x - mean) * self.img_range)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_before_upsample(x)
        return self.conv_last(self.upsample(x)) / self.img_range + mean
