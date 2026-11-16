"""Cloud-only paired raw-output and metric comparison of complete Set14 runs."""

import argparse
import csv
import json
from pathlib import Path

from tools.cloud_guard import require_cloud_workspace


def main():
    require_cloud_workspace()
    import numpy as np
    from drct_ms.evaluation import read_bgr, file_sha256
    from tools.verify_alignment import errors
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-run", type=Path, required=True)
    parser.add_argument("--ms-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path("/home/ma-user/work")):
        raise ValueError("Comparison output must be on persistent work volume")
    reference = json.loads((args.torch_run / "report.json").read_text())
    actual = json.loads((args.ms_run / "report.json").read_text())
    if not reference["complete"] or not actual["complete"] or len(reference["rows"]) != 14 or len(actual["rows"]) != 14:
        raise ValueError("Both evaluations must contain all 14 images")
    for field in ["dataset_manifest_sha256", "source_checkpoint_sha256", "branch", "network", "protocols",
                  "preprocessing", "quantization", "tile", "self_ensemble", "source_sha256"]:
        if reference[field] != actual[field]:
            raise ValueError(f"Evaluation settings differ: {field}")
    if reference["framework"] != "torch" or actual["framework"] != "mindspore":
        raise ValueError("Expected official PyTorch and target MindSpore reports")
    rows = []
    passed = True
    for first, second in zip(reference["rows"], actual["rows"]):
        if first["id"] != second["id"] or first["input_sha256"] != second["input_sha256"] or first["hr_sha256"] != second["hr_sha256"]:
            raise ValueError("Image ordering or paired data identity mismatch")
        label = first["id"]
        for folder, row in [(args.torch_run, first), (args.ms_run, second)]:
            if file_sha256(folder / (label + "_raw.npy")) != row["raw_sha256"]:
                raise ValueError("Saved raw output hash mismatch")
            if file_sha256(folder / (label + "_SR.png")) != row["sr_sha256"]:
                raise ValueError("Saved SR PNG hash mismatch")
        target = np.load(args.torch_run / (label + "_raw.npy"), allow_pickle=False)
        prediction = np.load(args.ms_run / (label + "_raw.npy"), allow_pickle=False)
        raw = errors(prediction, target)
        close = bool(np.allclose(prediction, target, atol=1e-4, rtol=1e-4))
        image_delta = (read_bgr(args.ms_run / (label + "_SR.png")).astype(np.int16)
                       - read_bgr(args.torch_run / (label + "_SR.png")).astype(np.int16))
        delta = {p: {metric: second["metrics"][p][metric] - first["metrics"][p][metric]
                     for metric in ("psnr", "ssim")} for p in reference["protocols"]}
        metric_close = all(abs(v["psnr"]) <= 0.01 and abs(v["ssim"]) <= 1e-4 for v in delta.values())
        row = {"id": label, "raw": raw, "allclose": close, "metric_deltas_ms_minus_torch": delta,
               "uint8_changed_values": int(np.count_nonzero(image_delta)),
               "uint8_total_values": int(image_delta.size), "uint8_max_abs_delta": int(np.abs(image_delta).max())}
        rows.append(row)
        passed &= close and metric_close
    report = {"passed": passed, "complete_pairs": 14, "dataset_manifest_sha256": actual["dataset_manifest_sha256"],
              "branch": actual["branch"], "source_checkpoint_sha256": actual["source_checkpoint_sha256"],
              "mindspore_checkpoint_sha256": actual["checkpoint_sha256"], "mindspore_mode": actual["mode"],
              "conv_backend": actual["conv_backend"], "raw_allclose_atol": 1e-4, "raw_allclose_rtol": 1e-4,
              "per_image_psnr_delta_limit_db": 0.01, "per_image_ssim_delta_limit": 1e-4,
              "torch_average": reference["average"], "mindspore_average": actual["average"],
              "average_deltas_ms_minus_torch": {p: {m: actual["average"][p][m] - reference["average"][p][m]
                                                    for m in ("psnr", "ssim")} for p in actual["protocols"]},
              "paper_reference_psnr_db": 29.54,
              "mindspore_y4_minus_paper_reference_db": actual["average"]["y_shave4"]["psnr"] - 29.54,
              "maximum_raw_absolute_error": max(row["raw"]["max_abs"] for row in rows),
              "npu_peak_tensor_allocated_bytes": max(r["memory"].get("tensor_peak_allocated_bytes", 0) for r in actual["rows"]),
              "npu_peak_tensor_pool_reserved_bytes": max(r["memory"].get("tensor_pool_peak_reserved_bytes", 0) for r in actual["rows"]),
              "mindspore_total_forward_seconds_including_shape_compile": sum(r["forward_seconds"] for r in actual["rows"]),
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "max_abs", "mean_abs", "rmse", "y_psnr_delta", "y_ssim_delta", "rgb_psnr_delta", "rgb_ssim_delta", "uint8_changed_values"])
        for row in rows:
            d = row["metric_deltas_ms_minus_torch"]
            writer.writerow([row["id"], row["raw"]["max_abs"], row["raw"]["mean_abs"], row["raw"]["rmse"],
                             d["y_shave4"]["psnr"], d["y_shave4"]["ssim"], d["rgb_shave8"]["psnr"], d["rgb_shave8"]["ssim"], row["uint8_changed_values"]])
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
