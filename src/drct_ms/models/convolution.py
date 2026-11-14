"""Explicit FP32 cross-correlation via spatial slices and GEMM.

Same OIHW weights/bias and zero padding as nn.Conv2d, stride=1/group=1.
Default validated backend; native Conv is retained separately for diagnostics.
"""

import mindspore as ms
from mindspore import nn, ops, Parameter


class MatmulConv2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        padding = kernel_size // 2
        self.pad = ops.Pad(((0, 0), (0, 0), (padding, padding), (padding, padding)))
        self.weight = Parameter(ops.zeros((out_channels, in_channels, kernel_size, kernel_size), ms.float32), name="weight")
        self.bias = Parameter(ops.zeros((out_channels,), ms.float32), name="bias")

    def construct(self, x):
        b, c, h, w = x.shape
        k = self.kernel_size
        if k == 1:
            patches = ops.transpose(x, (0, 2, 3, 1)).reshape((-1, c))
        else:
            padded = self.pad(x)
            pieces = ()
            for row in range(self.kernel_size):
                for col in range(self.kernel_size):
                    pieces += (padded[:, :, row:row + h, col:col + w],)
            # [B,C,K*K,H,W] -> [B,H,W,C,K*K], matching OIHW flatten order.
            stacked = ops.stack(pieces, axis=2)
            patches = ops.transpose(stacked, (0, 3, 4, 1, 2)).reshape((-1, c * k * k))
        weight = ops.transpose(self.weight.reshape((self.out_channels, c * k * k)), (1, 0))
        y = ops.matmul(patches, weight) + self.bias
        return ops.transpose(y.reshape((b, h, w, self.out_channels)), (0, 3, 1, 2))
