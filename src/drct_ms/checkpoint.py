"""Strict parameter mapping from cloud-exported NumPy bundles to MindSpore.

No torch dependency. Learned parameters and deterministic geometry buffers have
separate, explicit coverage checks; unexpected entries are never silently dropped.
"""

import hashlib
import json
from pathlib import Path

from .runtime import require_cloud_workspace

require_cloud_workspace()

import mindspore as ms
import numpy as np


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_parameter_name(target_name):
    """Only LayerNorm gamma/beta are renamed; Dense/Conv weights stay unchanged."""
    if target_name.endswith(".gamma"):
        return target_name[:-6] + ".weight"
    if target_name.endswith(".beta"):
        return target_name[:-5] + ".bias"
    return target_name


def geometry_buffers(net):
    buffers = {}
    for prefix, cell in net.cells_and_names():
        base = prefix + "." if prefix else ""
        if hasattr(cell, "relative_position_index"):
            buffers[base + "relative_position_index"] = cell.relative_position_index.asnumpy()
        if hasattr(cell, "attn_mask") and cell.attn_mask is not None:
            buffers[base + "attn_mask"] = cell.attn_mask.asnumpy()
    return buffers


def load_numpy_state(net, state, parameter_names):
    target = net.parameters_dict()
    mapping = {name: source_parameter_name(name) for name in target}
    required = set(mapping.values())
    supplied = set(parameter_names)
    buffers = geometry_buffers(net)
    missing = sorted((required - supplied) | (required - set(state)))
    unexpected_parameters = sorted(supplied - required)
    unexpected_entries = sorted(set(state) - required - set(buffers))
    missing_buffers = sorted(set(buffers) - set(state))
    shape_mismatches, dtype_mismatches, bad_values, buffer_mismatches = [], [], [], []
    for target_name, source_name in mapping.items():
        if source_name not in state:
            continue
        value = state[source_name]
        if tuple(value.shape) != tuple(target[target_name].shape):
            shape_mismatches.append(source_name)
        if value.dtype != np.float32:
            dtype_mismatches.append(source_name)
        if not np.isfinite(value).all():
            bad_values.append(source_name)
    for name, expected in buffers.items():
        if name in state and not np.array_equal(state[name], expected):
            buffer_mismatches.append(name)
    report = {
        "target_parameter_tensors": len(target), "source_parameter_tensors": len(supplied),
        "target_parameter_elements": sum(p.size for p in target.values()),
        "mapping": mapping, "geometry_buffers_checked": len(buffers),
        "missing_parameters": missing, "unexpected_parameters": unexpected_parameters,
        "unexpected_entries": unexpected_entries, "missing_buffers": missing_buffers,
        "shape_mismatches": shape_mismatches, "dtype_mismatches": dtype_mismatches,
        "nonfinite_parameters": bad_values, "geometry_mismatches": buffer_mismatches,
    }
    if any((missing, unexpected_parameters, unexpected_entries, missing_buffers,
            shape_mismatches, dtype_mismatches, bad_values, buffer_mismatches)):
        raise ValueError("State mapping failed: " + json.dumps({k: v for k, v in report.items() if k != "mapping"}))
    parameters = {name: ms.Parameter(ms.Tensor(state[source], ms.float32), name=name)
                  for name, source in mapping.items()}
    missing_net, missing_ckpt = ms.load_param_into_net(net, parameters, strict_load=True)
    if missing_net or missing_ckpt:
        raise RuntimeError(f"Strict load failed: {missing_net}, {missing_ckpt}")
    report["coverage"] = 1.0
    report["strict_load"] = "passed"
    return report


def load_bundle(net, bundle):
    bundle = Path(bundle)
    metadata = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    path = bundle / "state.npz"
    if sha256(path) != metadata["state_sha256"]:
        raise ValueError("Reference state bundle hash mismatch")
    with np.load(path, allow_pickle=False) as state:
        report = load_numpy_state(net, state, metadata["parameter_names"])
    report.update(source_checkpoint_sha256=metadata["checkpoint_sha256"], branch=metadata["branch"],
                  state_sha256=metadata["state_sha256"])
    return report


def load_checkpoint_strict(net, checkpoint):
    """Reject absent/extra/different-dtype parameters before MindSpore loading."""
    state = ms.load_checkpoint(str(checkpoint))
    target = net.parameters_dict()
    if set(state) != set(target):
        raise ValueError(f"Checkpoint name mismatch: missing={set(target)-set(state)}, extra={set(state)-set(target)}")
    for name, parameter in target.items():
        if state[name].shape != parameter.shape or state[name].dtype != parameter.dtype:
            raise ValueError(f"Checkpoint shape/dtype mismatch: {name}")
    missing_net, missing_ckpt = ms.load_param_into_net(net, state, strict_load=True)
    if missing_net or missing_ckpt:
        raise ValueError(f"Strict load failed: {missing_net}, {missing_ckpt}")
