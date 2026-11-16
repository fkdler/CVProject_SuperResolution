"""Cloud-only parity check against actual pinned BasicSR metric and tensor2img code."""

import argparse
import ast
import json
from pathlib import Path

from tools.cloud_guard import ROOT, require_cloud_workspace


def official_functions():
    require_cloud_workspace()
    import cv2
    import numpy as np
    import torch
    from tools.reference_loader import checked_source

    namespace = {"cv2": cv2, "np": np, "torch": torch}
    sources = {
        "reference/BasicSR/basicsr/utils/matlab_functions.py":
            {"bgr2ycbcr", "_convert_input_type_range", "_convert_output_type_range"},
        "reference/BasicSR/basicsr/metrics/metric_util.py": {"reorder_image", "to_y_channel"},
        "reference/BasicSR/basicsr/metrics/psnr_ssim.py": {"calculate_psnr", "calculate_ssim", "_ssim"},
        "reference/BasicSR/basicsr/utils/img_util.py": {"tensor2img"},
    }
    for relative, names in sources.items():
        path = checked_source(relative)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in functions} != names:
            raise ValueError("Pinned reference function is missing")
        for node in functions:
            node.decorator_list = []
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def main():
    require_cloud_workspace()
    import numpy as np
    import torch
    from drct_ms.evaluation import (metrics, reflect_pad_to_window, to_bgr_uint8,
                                    input_from_bgr, load_dataset, read_bgr)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/Set14")
    parser.add_argument("--report", type=Path, default=ROOT / "experiments/stage3/evaluation_parity.json")
    args = parser.parse_args()
    if not args.report.resolve().is_relative_to(Path("/home/ma-user/work")):
        raise ValueError("Report must be on the persistent work volume")
    functions = official_functions()
    rng = np.random.default_rng(20261116)
    checks = []
    for shape in [(37, 43, 3), (64, 80, 3)]:
        a = rng.integers(0, 256, shape, dtype=np.uint8)
        b = np.clip(a.astype(np.int32) + rng.integers(-9, 10, shape), 0, 255).astype(np.uint8)
        for protocol, border, y in [("y_shave4", 4, True), ("rgb_shave8", 8, False)]:
            actual = metrics(a, b, protocol)
            expected = {
                "psnr": float(functions["calculate_psnr"](a, b, border, test_y_channel=y)),
                "ssim": float(functions["calculate_ssim"](a, b, border, test_y_channel=y)),
            }
            delta = {name: abs(actual[name] - expected[name]) for name in actual}
            if max(delta.values()) > 1e-12:
                raise AssertionError(f"BasicSR metric mismatch: {protocol} {delta}")
            checks.append({"shape": list(shape), "protocol": protocol, "absolute_errors": delta})
    for shape in [(1, 3, 17, 31), (1, 3, 32, 48), (1, 3, 97, 146)]:
        value = rng.uniform(-0.3, 1.3, shape).astype(np.float32)
        np.testing.assert_array_equal(to_bgr_uint8(value), functions["tensor2img"](torch.from_numpy(value.copy())))
        padded, _ = reflect_pad_to_window(value)
        ph, pw = (-shape[2]) % 16, (-shape[3]) % 16
        expected = torch.nn.functional.pad(torch.from_numpy(value), (0, pw, 0, ph), mode="reflect").numpy()
        np.testing.assert_array_equal(padded, expected)
    ties = np.tile((np.arange(256, dtype=np.float32) + 0.5) / 255.0, (1, 3, 1, 1))
    np.testing.assert_array_equal(to_bgr_uint8(ties), functions["tensor2img"](torch.from_numpy(ties.copy())))
    dataset, dataset_hash = load_dataset(args.dataset)
    for pair in dataset["pairs"]:
        for role in ("lr", "hr"):
            image = read_bgr(args.dataset / pair[role]["file"])
            value = input_from_bgr(image)
            np.testing.assert_array_equal(to_bgr_uint8(value), image)
    report = {"passed": True, "torch": torch.__version__, "dataset_manifest_sha256": dataset_hash,
              "metrics": checks, "quantization_cases": 4, "reflection_cases": 3,
              "decoded_scale_validated_pairs": 14, "input_color_roundtrip_images": 28,
              "reference": "BasicSR 1.3.4.9 pinned source functions loaded without body edits"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
