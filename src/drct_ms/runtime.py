"""Cloud-only execution policy. This module uses the standard library only."""

import platform
import sys
from pathlib import Path


def require_cloud_workspace():
    root = Path(__file__).resolve().parents[2]
    allowed = Path("/home/ma-user/work")
    if sys.platform != "linux" or platform.machine().lower() not in {"aarch64", "arm64"}:
        raise RuntimeError("Model execution is cloud-only: use the ModelArts Linux ARM64 Notebook")
    if not root.is_relative_to(allowed):
        raise RuntimeError("Cloud project and outputs must be under /home/ma-user/work")
    return root


def configure_ascend(mode="pynative", seed=0):
    require_cloud_workspace()
    import mindspore as ms

    if ms.__version__ != "2.7.2":
        raise RuntimeError(f"Expected MindSpore 2.7.2, found {ms.__version__}")
    if mode not in {"pynative", "graph"}:
        raise ValueError("mode must be pynative or graph")
    ms.set_device("Ascend")
    ms.set_context(mode=ms.PYNATIVE_MODE if mode == "pynative" else ms.GRAPH_MODE)
    precision = ms.device_context.ascend.op_precision
    precision.precision_mode("must_keep_origin_dtype")
    precision.matmul_allow_hf32(False)
    precision.conv_allow_hf32(False)
    if mode == "graph":
        # Explicit kernel-by-kernel graph execution avoids costly GE whole-graph tuning.
        ms.set_context(jit_config={"jit_level": "O0"})
    ms.set_seed(seed)
    return ms
