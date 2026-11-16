"""Cloud-only full-image Set14 x4 evaluation with identical PT/MS preprocessing and metrics."""

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

from tools.cloud_guard import ROOT, require_cloud_workspace


def main():
    require_cloud_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=["torch", "mindspore"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/Set14")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["pynative", "graph"], default="pynative")
    parser.add_argument("--branch", choices=["params_ema", "params"], default="params_ema")
    parser.add_argument("--alignment-report", type=Path, default=ROOT / "experiments/alignment_pynative_matmul.json")
    parser.add_argument("--parity-report", type=Path, default=ROOT / "experiments/stage3/evaluation_parity.json")
    parser.add_argument("--ids", nargs="+", help="Explicit resource smoke subset; not a complete benchmark")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path("/home/ma-user/work")):
        raise ValueError("Outputs must stay on the persistent work volume")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new output directory; existing experiments are preserved")
    output.mkdir(parents=True, exist_ok=True)
    import cv2
    import numpy as np
    from drct_ms.evaluation import (PROTOCOLS, file_sha256, load_dataset, read_bgr, input_from_bgr,
                                    reflect_pad_to_window, unpad_output, to_bgr_uint8, metrics)
    dataset, data_hash = load_dataset(args.dataset)
    parity = json.loads(args.parity_report.read_text(encoding="utf-8"))
    if not parity["passed"] or parity["dataset_manifest_sha256"] != data_hash:
        raise ValueError("Pinned metric/preprocessing parity checks must pass before evaluation")
    config = json.loads((ROOT / "configs/drct_l_x4.json").read_text(encoding="utf-8"))["network_g"]
    source = json.loads((ROOT / "reference/manifest.json").read_text(encoding="utf-8"))
    checkpoint_hash = file_sha256(args.checkpoint)
    if args.framework == "torch":
        import torch
        from tools.reference_loader import load_reference
        if checkpoint_hash != source["checkpoint"]["sha256"]:
            raise ValueError("Official checkpoint identity mismatch")
        torch.set_num_threads(args.threads)
        torch.set_float32_matmul_precision("highest")
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model = load_reference()["DRCT"](**config).eval()
        model.load_state_dict(state[args.branch], strict=True)
        del state
        version, device = torch.__version__, "cloud CPU"
        def predict(value):
            with torch.no_grad():
                return model(torch.from_numpy(value)).numpy()
        def memory():
            return {"host_peak_rss_cumulative_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        def reset_memory():
            pass
    else:
        from drct_ms.runtime import configure_ascend
        ms = configure_ascend(args.mode)
        from drct_ms.models.drct import DRCT
        from drct_ms.checkpoint import load_checkpoint_strict
        provenance = json.loads(args.checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
        alignment = json.loads(args.alignment_report.read_text(encoding="utf-8"))
        if (provenance["checkpoint_sha256"] != checkpoint_hash
                or provenance["source_checkpoint_sha256"] != source["checkpoint"]["sha256"]
                or provenance["branch"] != args.branch or provenance["config"] != config
                or not alignment["passed"] or alignment["checkpoint_sha256"] != checkpoint_hash
                or alignment["conv_backend"] != "matmul"):
            raise ValueError("MindSpore checkpoint provenance/numerical prerequisite failed")
        model = DRCT(**config, conv_backend="matmul")
        load_checkpoint_strict(model, args.checkpoint)
        model.set_train(False)
        version, device = ms.__version__, "Ascend"
        def predict(value):
            return model(ms.Tensor(value, ms.float32)).asnumpy()
        def memory():
            return {"tensor_peak_allocated_bytes": ms.runtime.max_memory_allocated(),
                    "tensor_pool_peak_reserved_bytes": ms.runtime.max_memory_reserved(),
                    "host_peak_rss_cumulative_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        def reset_memory():
            ms.runtime.synchronize()
            ms.runtime.reset_peak_memory_stats()

    selected = dataset["pairs"] if not args.ids else [p for p in dataset["pairs"] if p["id"] in args.ids]
    if args.ids and (len(args.ids) != len(set(args.ids)) or {p["id"] for p in selected} != set(args.ids)):
        raise ValueError("Unknown or duplicate dataset ID")
    try:
        code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        code_commit = "source bundle (see source hashes)"
    report = {"schema": 1, "dataset": "Set14", "scale": 4, "dataset_manifest_sha256": data_hash,
              "dataset_source": dataset["source"], "framework": args.framework, "version": version,
              "device": device, "mode": args.mode if args.framework == "mindspore" else "eager",
              "conv_backend": "matmul" if args.framework == "mindspore" else "official",
              "checkpoint_sha256": checkpoint_hash, "source_checkpoint_sha256": source["checkpoint"]["sha256"],
              "branch": args.branch, "network": config, "protocols": PROTOCOLS,
              "preprocessing": "OpenCV color decode, BGR->RGB, float32/255, right/bottom reflect to 16",
              "quantization": "clamp [0,1], float32 *255, numpy round, uint8 before metrics",
              "tile": False, "self_ensemble": False, "timing": "one full forward plus host materialization; includes first-shape compilation",
              "code_commit": code_commit, "python": platform.python_version(), "rows": [], "complete": False,
              "source_sha256": {name: file_sha256(ROOT / name) for name in [
                  "src/drct_ms/evaluation.py", "src/drct_ms/models/drct.py", "src/drct_ms/models/convolution.py",
                  "src/drct_ms/runtime.py", "tools/evaluate_set14.py"]}}

    def write_report():
        temporary = output / "report.partial.json"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, output / "report.json")

    write_report()
    try:
        for pair in selected:
            started = time.perf_counter()
            lr, hr = read_bgr(args.dataset / pair["lr"]["file"]), read_bgr(args.dataset / pair["hr"]["file"])
            value, size = reflect_pad_to_window(input_from_bgr(lr))
            reset_memory()
            before = time.perf_counter()
            prediction = predict(value)
            elapsed = time.perf_counter() - before
            prediction = unpad_output(prediction, size)
            if prediction.shape != (1, 3, hr.shape[0], hr.shape[1]):
                raise ValueError("Model returned unexpected upscaled image size")
            row = {"id": pair["id"], "lr_shape": list(lr.shape), "padded_input_shape": list(value.shape),
                   "output_shape": list(prediction.shape), "forward_seconds": elapsed, "memory": memory(),
                   "input_sha256": pair["lr"]["sha256"], "hr_sha256": pair["hr"]["sha256"]}
            quantized = to_bgr_uint8(prediction)
            np.save(output / (pair["id"] + "_raw.npy"), prediction, allow_pickle=False)
            if not cv2.imwrite(str(output / (pair["id"] + "_SR.png")), quantized):
                raise OSError("Failed to save SR image")
            row["raw_sha256"] = file_sha256(output / (pair["id"] + "_raw.npy"))
            row["sr_sha256"] = file_sha256(output / (pair["id"] + "_SR.png"))
            row["metrics"] = {protocol: metrics(quantized, hr, protocol) for protocol in PROTOCOLS}
            row["total_seconds"] = time.perf_counter() - started
            report["rows"].append(row)
            write_report()
            print(json.dumps(row, allow_nan=False), flush=True)
            del prediction, quantized, value, lr, hr
            gc.collect()
        report["complete"] = len(report["rows"]) == 14
        report["average"] = {protocol: {metric: float(np.mean([r["metrics"][protocol][metric] for r in report["rows"]]))
                                       for metric in ("psnr", "ssim")} for protocol in PROTOCOLS}
        write_report()
        with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "y_psnr", "y_ssim", "rgb8_psnr", "rgb8_ssim", "forward_seconds"])
            for row in report["rows"]:
                writer.writerow([row["id"], row["metrics"]["y_shave4"]["psnr"], row["metrics"]["y_shave4"]["ssim"],
                                 row["metrics"]["rgb_shave8"]["psnr"], row["metrics"]["rgb_shave8"]["ssim"], row["forward_seconds"]])
        print(json.dumps({"complete": report["complete"], "average": report["average"]}), flush=True)
    except BaseException as error:
        report["error"] = repr(error)
        write_report()
        raise


if __name__ == "__main__":
    main()
