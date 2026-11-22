"""Train, evaluate and benchmark the MindSpore DRCT-L comparison baseline on ModelArts."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time

from tools.cloud_guard import ROOT, require_cloud_workspace

WORK = Path("/home/ma-user/work")
PLAN = ROOT / "configs/experiment_baseline.json"
NETWORK = ROOT / "configs/drct_l_x4.json"
OFFICIAL = ROOT / "weights/drct_l_x4_params_ema.ckpt"
DATASET = ROOT / "data/DIV2K_baseline"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def create_output(path):
    path = path.resolve()
    if not path.is_relative_to(WORK) or path == WORK:
        raise ValueError("Use a new output directory inside the persistent work volume")
    path.mkdir(parents=True, exist_ok=False)
    return path


def source_hashes():
    from drct_ms.evaluation import file_sha256

    paths = sorted((ROOT / "src/drct_ms").rglob("*.py"))
    paths.extend(ROOT / "tools" / name for name in ("experiment.py", "experiment_data.py"))
    return {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in paths}


def checkpoint_info(path, network):
    from drct_ms.evaluation import file_sha256

    info = read_json(path.with_suffix(".json"))
    official_source = read_json(ROOT / "reference/manifest.json")["checkpoint"]["sha256"]
    if (info["checkpoint_sha256"] != file_sha256(path) or info["config"] != network
            or info["branch"] != "params_ema" or info["source_checkpoint_sha256"] != official_source):
        raise ValueError("Checkpoint identity, architecture, source or EMA branch mismatch")
    kind = info.get("kind", "official_conversion")
    if kind == "experiment_ema":
        if (info["step"] < 1 or not info["training_checkpoint_sha256"]
                or not info["identity"]["dataset_manifest_sha256"]):
            raise ValueError("Experiment EMA must carry its training and dataset provenance")
    elif kind != "official_conversion" or info["checkpoint_sha256"] != read_json(
        ROOT / "docs/verification/ascend_stage2/drct_l_x4_params_ema.json"
    )["checkpoint_sha256"]:
        raise ValueError("Expected the audited official conversion or an experiment EMA")
    return {**info, "kind": kind}


def load_model(path, mode, seed):
    from drct_ms.runtime import configure_ascend

    network = read_json(NETWORK)["network_g"]
    info = checkpoint_info(path, network)
    ms = configure_ascend(mode, seed)
    from drct_ms.checkpoint import load_checkpoint_strict
    from drct_ms.models.drct import DRCT

    model = DRCT(**network, conv_backend="matmul")
    load_checkpoint_strict(model, path)
    model.set_train(False)
    return ms, model, info


def div2k_dataset(path, plan):
    from tools.experiment_data import ExperimentDIV2K

    return ExperimentDIV2K(path, read_json(ROOT / plan["dataset_selection"]))


def environment(ms, mode):
    return {
        "mindspore": ms.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "device": "Ascend",
        "mode": mode,
        "precision": "FP32 / HF32 off",
        "conv_backend": "matmul",
    }


def batch_hash(batch):
    return hashlib.sha256(batch[0].tobytes() + batch[1].tobytes()).hexdigest()


def export_ema(ms, trainer, path, saved, info, identity):
    from drct_ms.evaluation import file_sha256

    temporary = path.with_name("ema.partial.ckpt")
    ms.save_checkpoint([{"name": name, "data": value} for name, value in trainer.ema.items()], str(temporary))
    temporary.replace(path)
    exported = {
        "kind": "experiment_ema",
        "branch": "params_ema",
        "checkpoint_sha256": file_sha256(path),
        "source_checkpoint_sha256": info["source_checkpoint_sha256"],
        "config": info["config"],
        "step": trainer.step,
        "training_checkpoint_sha256": saved["checkpoint_sha256"],
        "identity": identity,
    }
    write_json(path.with_suffix(".json"), exported)
    return exported


def train(args):
    plan = read_json(args.plan)
    config = plan["training"]
    target = args.until_step if args.until_step is not None else plan["total_steps"]
    if not 1 <= target <= plan["total_steps"]:
        raise ValueError("Target step must be within the predeclared experiment plan")
    patch = config["lr_patch_size"]
    window = read_json(NETWORK)["network_g"]["window_size"]
    if patch < window or patch % window or config["batch_size"] < 1 or config["seed"] < 0:
        raise ValueError("Training requires a positive batch, nonnegative seed and a window-aligned patch")
    dataset = div2k_dataset(args.dataset, plan)
    ms, model, info = load_model(args.pretrained, config["mode"], config["seed"])
    if info["kind"] != "official_conversion":
        raise ValueError("Initialize the original baseline from the official EMA; use --resume to continue")
    from drct_ms.training import Trainer

    identity = {
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "pretrained_sha256": info["checkpoint_sha256"],
        "model_config": info["config"],
        "conv_backend": "matmul",
        "total_steps": plan["total_steps"],
        "source_sha256": source_hashes(),
    }
    trainer = Trainer(model, config)
    restored = trainer.restore(args.resume, identity) if args.resume else None
    if trainer.step >= target:
        raise ValueError("Target step must be after the restored step")
    output = create_output(args.output)
    first_batch = dataset.batch(trainer.step, config)
    initial_step = trainer.step
    report = {
        "environment": environment(ms, config["mode"]),
        "plan": plan,
        "identity": identity,
        "train_ids": dataset.train_ids,
        "valid_ids": [p["id"] for p in dataset.valid_pairs],
        "resume": restored,
        "initial_step": initial_step,
        "initial_batch_sha256": batch_hash(first_batch),
        "parameter_elements": sum(p.size for p in model.trainable_params()),
    }
    ms.runtime.reset_peak_memory_stats()
    started = time.perf_counter()
    elapsed_steps = []
    with (output / "train.jsonl").open("w", encoding="utf-8") as log:
        while trainer.step < target:
            lr, hr, samples = first_batch if trainer.step == initial_step else dataset.batch(trainer.step, config)
            row = trainer.update(lr, hr)
            row["samples"] = samples
            elapsed_steps.append(row["audited_update_seconds"])
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps(row, allow_nan=False), flush=True)
    report["training_wall_seconds"] = time.perf_counter() - started
    report["first_update_seconds"] = elapsed_steps[0]
    report["warm_update_median_seconds"] = statistics.median(elapsed_steps[1:]) if len(elapsed_steps) > 1 else None
    report["warm_update_count"] = len(elapsed_steps) - 1
    report["training_allocator_peak_bytes"] = ms.runtime.max_memory_allocated()
    report["training_pool_peak_reserved_bytes"] = ms.runtime.max_memory_reserved()
    saved = trainer.save(output / f"step_{trainer.step:06d}.ckpt", identity)
    report["saved_checkpoint"] = saved
    report["ema"] = export_ema(ms, trainer, output / "ema.ckpt", saved, info, identity)
    report["next_batch_sha256"] = batch_hash(dataset.batch(trainer.step, config))
    write_json(output / "report.json", report)


def evaluate(args):
    from drct_ms.evaluation import (PROTOCOLS, input_from_bgr, load_dataset,
                                    metrics, read_bgr, reflect_pad_to_window, to_bgr_uint8, unpad_output)

    plan = read_json(args.plan)
    if args.split == "div2k-valid":
        from tools.fetch_set14 import png_header

        dataset = div2k_dataset(args.dataset or DATASET, plan)
        pairs, manifest_hash = dataset.valid_pairs, dataset.manifest_sha256
        read_pair = dataset.read_pair
        window = read_json(NETWORK)["network_g"]["window_size"]

        def padded_area(pair):
            png = png_header(dataset.paths[pair["id"]][0])
            return ((png["height"] + window - 1) // window) * ((png["width"] + window - 1) // window)

        # Start with the largest allocation so earlier shapes cannot fragment the pool.
        execution_pairs = sorted(pairs, key=padded_area, reverse=True)
    else:
        directory = args.dataset or ROOT / "data/Set14"
        metadata, manifest_hash = load_dataset(directory)
        pairs = metadata["pairs"]
        execution_pairs = pairs
        records = {p["id"]: p for p in pairs}

        def read_pair(image_id):
            pair = records[image_id]
            return read_bgr(directory / pair["lr"]["file"]), read_bgr(directory / pair["hr"]["file"]), {
                "bottom_hr": 0, "right_hr": 0,
            }

    config = plan["training"]
    ms, model, info = load_model(args.checkpoint, config["mode"], config["seed"])
    if (args.split == "div2k-valid" and info["kind"] == "experiment_ema"
            and info["identity"]["dataset_manifest_sha256"] != manifest_hash):
        raise ValueError("Experiment EMA and the DIV2K train/validation manifest differ")
    output = create_output(args.output)
    rows = []
    report = {
        "split": args.split, "scope": "full-image", "protocol": PROTOCOLS["y_shave4"],
        "quantization": "uint8 before metrics", "padding": "right/bottom reflect to window",
        "dataset_manifest_sha256": manifest_hash, "checkpoint": info, "rows": rows,
        "environment": environment(ms, config["mode"]), "source_sha256": source_hashes(),
        "evaluation_order": [pair["id"] for pair in execution_pairs],
        "complete": False,
    }
    write_json(output / "report.json", report)
    for pair in execution_pairs:
        image_id = pair["id"]
        lr, hr, trim = read_pair(image_id)
        value, original_size = reflect_pad_to_window(input_from_bgr(lr), info["config"]["window_size"])
        sr = model(ms.Tensor(value, ms.float32)).asnumpy()
        sr = unpad_output(sr, original_size)
        if sr.shape != (1, 3, hr.shape[0], hr.shape[1]):
            raise ValueError(f"Unexpected output dimensions for {image_id}")
        row = {"id": image_id, "lr_shape": list(lr.shape[:2]), **trim,
               "lr_sha256": pair["lr"]["sha256"], "hr_sha256": pair["hr"]["sha256"],
               **metrics(to_bgr_uint8(sr), hr, "y_shave4")}
        rows.append(row)
        write_json(output / "report.json", report)
        print(json.dumps(row, allow_nan=False), flush=True)
    rows.sort(key=lambda row: row["id"])
    report["average"] = {key: statistics.mean(row[key] for row in rows) for key in ("psnr", "ssim")}
    if args.split == "set14":
        report["reference_parity_manifest_matches"] = manifest_hash == read_json(
            ROOT / "docs/verification/stage3/evaluation_parity.json"
        )["dataset_manifest_sha256"]
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report["complete"] = True
    write_json(output / "report.json", report)


def benchmark(args):
    config = read_json(args.plan)["benchmark"]
    height, width = args.shape
    if list(args.shape) not in config["shapes_lr"] or config["warmup"] < 1 or config["repeats"] < 2:
        raise ValueError("Use a predeclared shape, at least one warmup and at least two repetitions")
    ms, model, info = load_model(args.checkpoint, config["mode"], config["seed"])
    window = info["config"]["window_size"]
    if min(height, width) < window or height % window or width % window:
        raise ValueError("Benchmark dimensions must be positive multiples of the model window")
    output = create_output(args.output)
    import numpy as np

    input_host = np.random.default_rng(config["seed"]).random((1, 3, height, width), dtype=np.float32)
    value = ms.Tensor(input_host)

    def timed_forward():
        ms.runtime.synchronize()
        started = time.perf_counter_ns()
        result = model(value)
        ms.runtime.synchronize()
        return (time.perf_counter_ns() - started) / 1e6, result

    first_ms, first_output = timed_forward()
    expected = (1, 3, height * 4, width * 4)
    materialized = first_output.asnumpy()
    if first_output.shape != expected or not np.isfinite(materialized).all():
        raise ValueError("Nonfinite or wrong-shaped benchmark output")
    del first_output
    warmup_ms = []
    for _ in range(config["warmup"]):
        duration, result = timed_forward()
        del result
        warmup_ms.append(duration)
    ms.runtime.synchronize()
    ms.runtime.reset_peak_memory_stats()
    prior_allocated = ms.runtime.memory_allocated()
    prior_reserved = ms.runtime.memory_reserved()
    samples = []
    for _ in range(config["repeats"]):
        duration, result = timed_forward()
        del result
        samples.append(duration)
    allocated = ms.runtime.max_memory_allocated()
    reserved = ms.runtime.max_memory_reserved()
    snapshot = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=20)
    (output / "npu_smi.log").write_text(snapshot.stdout + snapshot.stderr, encoding="utf-8")
    report = {
        "environment": environment(ms, config["mode"]), "checkpoint": info,
        "input_shape": list(value.shape), "output_shape": list(expected),
        "input_sha256": hashlib.sha256(input_host.tobytes()).hexdigest(),
        "output_sha256": hashlib.sha256(materialized.tobytes()).hexdigest(),
        "parameter_elements": sum(p.size for p in model.trainable_params()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "first_forward_ms": first_ms, "warmup_ms": warmup_ms, "samples_ms": samples,
        "p50_ms": statistics.median(samples), "p95_ms": statistics.quantiles(samples, n=100, method="inclusive")[94],
        "allocator_before_bytes": prior_allocated, "allocator_peak_bytes": allocated,
        "pool_before_bytes": prior_reserved, "pool_peak_reserved_bytes": reserved,
        "npu_smi_exit_code": snapshot.returncode,
        "timing_scope": "host wall clock, synchronized forward, precreated input; no output copy or metrics",
        "source_sha256": source_hashes(),
    }
    write_json(output / "report.json", report)
    print(json.dumps({key: report[key] for key in ("input_shape", "p50_ms", "p95_ms", "allocator_peak_bytes")}))


def audit(args):
    """Link the baseline's independent processes and check the measured artifacts."""
    run = args.run_dir.resolve()
    if not run.is_relative_to(WORK):
        raise ValueError("Audit the reports on the persistent work volume")
    first, resumed, official, trained, small, large = (
        read_json(run / part / "report.json") for part in
        ("first", "resumed", "div2k_official", "div2k_step64", "bench_64", "bench_256")
    )
    logs = []
    for part in ("first", "resumed"):
        with (run / part / "train.jsonl").open(encoding="utf-8") as stream:
            logs.extend(json.loads(line) for line in stream)
    plan = first["plan"]
    identity = first["identity"]
    split_ids = first["valid_ids"]
    pretrained = identity["pretrained_sha256"]
    target = plan["total_steps"]
    checks = {
        "same_plan_source_and_data": first["plan"] == resumed["plan"] and identity == resumed["identity"],
        "optimizer_step_and_state_restored": (
            first["saved_checkpoint"]["completed_steps"] == resumed["initial_step"]
            == resumed["resume"]["completed_steps"] == 32
            and first["saved_checkpoint"]["state_tensors"] == resumed["resume"]["state_tensors_exactly_restored"] == 3667
            and first["saved_checkpoint"]["checkpoint_sha256"] == resumed["resume"]["checkpoint_sha256"]
        ),
        "next_batch_identical": first["next_batch_sha256"] == resumed["initial_batch_sha256"],
        "full_model_updated_through_target": (
            target == resumed["saved_checkpoint"]["completed_steps"] == trained["checkpoint"]["step"] == 64
            and [row["step"] for row in logs] == list(range(1, target + 1))
            and all(row["all_gradients_finite"] and row["gradient_tensors"] == 916
                    and row["changed_parameter_tensors"] > 0 and math.isfinite(row["loss_l1"]) for row in logs)
        ),
        "training_and_validation_are_disjoint": (
            {sample["id"] for row in logs for sample in row["samples"]} == set(first["train_ids"])
            and not set(first["train_ids"]).intersection(split_ids)
        ),
        "exported_ema_link": (
            trained["checkpoint"] == resumed["ema"]
            and trained["checkpoint"]["training_checkpoint_sha256"] == resumed["saved_checkpoint"]["checkpoint_sha256"]
        ),
        "quality_has_same_full_images_and_protocol": all(
            r["complete"] and r["scope"] == "full-image" and r["split"] == "div2k-valid"
            and r["dataset_manifest_sha256"] == identity["dataset_manifest_sha256"]
            and [row["id"] for row in r["rows"]] == split_ids
            and r["protocol"] == official["protocol"] == {"channel": "BT.601 Y", "crop_border_hr": 4}
            and r["source_sha256"] == trained["source_sha256"]
            and all(r["source_sha256"][name] == digest for name, digest in identity["source_sha256"].items()
                    if name.startswith("src/"))
            and r["evaluation_order"] == trained["evaluation_order"]
            for r in (official, trained)
        ) and official["checkpoint"]["checkpoint_sha256"] == pretrained,
        "benchmark_measured_both_shapes_with_official_ema": all(
            r["input_shape"] == [1, 3, *shape]
            and r["checkpoint"]["checkpoint_sha256"] == pretrained
            and r["source_sha256"] == identity["source_sha256"]
            and len(r["warmup_ms"]) == plan["benchmark"]["warmup"]
            and len(r["samples_ms"]) == plan["benchmark"]["repeats"]
            and all(math.isfinite(t) and t > 0 for t in r["samples_ms"])
            and 0 < r["allocator_peak_bytes"] <= r["pool_peak_reserved_bytes"]
            and r["parameter_elements"] == 27580719 and r["npu_smi_exit_code"] == 0
            for r, shape in zip((small, large), plan["benchmark"]["shapes_lr"], strict=True)
        ),
    }
    report = {
        "passed": all(checks.values()), "checks": checks,
        "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
        "official_ema_sha256": pretrained, "step64_ema_sha256": trained["checkpoint"]["checkpoint_sha256"],
        "div2k_full_image_official": official["average"], "div2k_full_image_step64": trained["average"],
        "training_wall_seconds": sum(r["training_wall_seconds"] for r in (first, resumed)),
        "median_training_update_seconds": [r["warm_update_median_seconds"] for r in (first, resumed)],
        "benchmark": [
            {key: r[key] for key in ("input_shape", "p50_ms", "p95_ms", "allocator_peak_bytes",
                                     "pool_peak_reserved_bytes", "parameter_elements")}
            for r in (small, large)
        ],
    }
    write_json(run / "acceptance.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["passed"]:
        raise SystemExit("Baseline artifacts failed one or more consistency checks")


def main():
    require_cloud_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)
    training = subparsers.add_parser("train", help="Fine-tune the original model or continue a saved segment")
    training.add_argument("--dataset", type=Path, default=DATASET)
    training.add_argument("--pretrained", type=Path, default=OFFICIAL)
    training.add_argument("--resume", type=Path)
    training.add_argument("--until-step", type=int)
    training.set_defaults(run=train)
    evaluation = subparsers.add_parser("evaluate", help="Full-image Y/shave4 scores for an official or experiment EMA")
    evaluation.add_argument("--split", choices=("div2k-valid", "set14"), default="div2k-valid")
    evaluation.add_argument("--dataset", type=Path)
    evaluation.add_argument("--checkpoint", type=Path, default=OFFICIAL)
    evaluation.set_defaults(run=evaluate)
    speed = subparsers.add_parser("benchmark", help="Warm synchronized inference on one fixed shape")
    speed.add_argument("--shape", type=int, nargs=2, metavar=("H", "W"), required=True)
    speed.add_argument("--checkpoint", type=Path, default=OFFICIAL)
    speed.set_defaults(run=benchmark)
    for command in (training, evaluation, speed):
        command.add_argument("--plan", type=Path, default=PLAN)
        command.add_argument("--output", type=Path, required=True)
    acceptance = subparsers.add_parser("audit", help="Verify a complete 32+32-step baseline run and summarize it")
    acceptance.add_argument("--run-dir", type=Path, required=True)
    acceptance.set_defaults(run=audit)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
