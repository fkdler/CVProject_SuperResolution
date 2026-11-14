"""Cloud-only, bounded full DRCT-L FP32 numerical comparison against official arrays."""

import argparse
import json
from pathlib import Path
import time

from tools.cloud_guard import ROOT, require_cloud_workspace


def errors(actual, expected):
    import numpy as np
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Invalid shapes or nonfinite comparison arrays")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {"max_abs": float(np.max(np.abs(delta))), "mean_abs": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta ** 2))),
            "relative_l2": float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(expected.ravel()), 1e-12))}


def trace_network(net, x):
    """Separate eager trace to locate divergence; final construct is always checked too."""
    from mindspore import ops
    trace = {}
    image = (x - net.mean) * net.img_range
    shallow = net.conv_first(image)
    trace["conv_first"] = shallow.asnumpy()
    tokens = net.pos_drop(net.patch_embed(shallow))
    trace["patch_embed"] = tokens.asnumpy()
    size = (shallow.shape[2], shallow.shape[3])
    # Inspect first RDG sublayers using the exact block instances and shared weights.
    block = net.layers[0]
    features = [tokens]
    for i in range(1, 6):
        swin = getattr(block, f"swin{i}")(ops.concat(tuple(features), -1), size)
        trace[f"swin{i}"] = swin.asnumpy()
        adjusted = getattr(block, f"adjust{i}")(block.pue(swin, size))
        if i < 5:
            adjusted = block.lrelu(adjusted)
        features.append(block.pe(adjusted))
    tokens = features[0] + 0.2 * features[5]
    trace["rdg0"] = tokens.asnumpy()
    for layer in net.layers[1:]:
        tokens = layer(tokens, size)
    trace["rdg11"] = tokens.asnumpy()
    tokens = net.norm(tokens)
    trace["norm"] = tokens.asnumpy()
    deep = net.conv_after_body(net.patch_unembed(tokens, size))
    trace["conv_after_body"] = deep.asnumpy()
    return trace


def main():
    require_cloud_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT / "experiments/reference_bundle")
    parser.add_argument("--mode", choices=["pynative", "graph"], default="pynative")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", action="store_true", help="Extra eager trace for the rectangular case")
    parser.add_argument("--conv-backend", choices=["native", "matmul"], default="matmul")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.trace and args.mode != "pynative":
        raise ValueError("Intermediate trace is supported only in pynative; graph checks full construct")
    if not args.report.resolve().is_relative_to(Path("/home/ma-user/work")):
        raise ValueError("Reports must stay under /home/ma-user/work")
    from drct_ms.runtime import configure_ascend
    ms = configure_ascend(args.mode)
    import numpy as np
    from drct_ms.models.drct import DRCT
    from drct_ms.checkpoint import load_checkpoint_strict, sha256

    metadata = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
    conversion = json.loads(args.checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    if (conversion["checkpoint_sha256"] != sha256(args.checkpoint)
            or conversion["source_checkpoint_sha256"] != metadata["checkpoint_sha256"]
            or conversion["branch"] != metadata["branch"]
            or conversion["state_sha256"] != metadata["state_sha256"]
            or conversion["config"] != metadata["config"]):
        raise ValueError("Checkpoint and reference bundle provenance differ")
    net = DRCT(**metadata["config"], conv_backend=args.conv_backend)
    load_checkpoint_strict(net, args.checkpoint)
    net.set_train(False)
    report = {"mindspore": ms.__version__, "device": "Ascend", "mode": args.mode,
              "jit_level": "O0" if args.mode == "graph" else None,
              "conv_backend": args.conv_backend,
              "precision_mode": "must_keep_origin_dtype", "hf32": False,
              "checkpoint_sha256": sha256(args.checkpoint), "branch": metadata["branch"],
              "atol": args.atol, "rtol": args.rtol, "cases": [], "passed": True}
    for case in metadata["cases"]:
        path = args.bundle / case["file"]
        if sha256(path) != case["sha256"]:
            raise ValueError("Golden case hash mismatch")
        with np.load(path, allow_pickle=False) as golden:
            x = ms.Tensor(golden["input"], ms.float32)
            start = time.perf_counter()
            actual = net(x).asnumpy()  # Host materialization synchronizes the result.
            elapsed = time.perf_counter() - start
            result = {"name": case["name"], "shape": list(x.shape), "seconds_including_compile": elapsed,
                      "output": errors(actual, golden["output"]),
                      "passed": bool(np.allclose(actual, golden["output"], atol=args.atol, rtol=args.rtol))}
            if args.trace and case["name"] == "rectangular":
                result["trace"] = {}
                for name, value in trace_network(net, x).items():
                    result["trace"][name] = errors(value, golden[name])
                    result["passed"] &= bool(np.allclose(value, golden[name], atol=args.atol, rtol=args.rtol))
            report["cases"].append(result)
            report["passed"] &= result["passed"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
