"""Bounded cloud-only full DRCT-L pretrained training smoke and resumable continuation."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

from tools.cloud_guard import ROOT, require_cloud_workspace


def main():
    require_cloud_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_smoke.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/DIV2K_smoke")
    parser.add_argument("--pretrained", type=Path, default=ROOT / "weights/drct_l_x4_params_ema.ckpt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=2, help="Additional updates, max 20 across a smoke chain")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path("/home/ma-user/work")) or (output.exists() and any(output.iterdir())):
        raise ValueError("Choose a new output directory on the persistent work volume")
    if not 1 <= args.steps <= 20:
        raise ValueError("This entry point is bounded to 20 smoke updates; long training requires measured budget approval")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    from drct_ms.runtime import configure_ascend
    ms = configure_ascend(config["mode"], config["seed"])
    from drct_ms.models.drct import DRCT
    from drct_ms.checkpoint import load_checkpoint_strict
    from drct_ms.evaluation import file_sha256
    from drct_ms.training import Trainer
    from drct_ms.training_data import PairedDIV2K
    model_config = json.loads((ROOT / "configs/drct_l_x4.json").read_text(encoding="utf-8"))["network_g"]
    pretrained_hash = file_sha256(args.pretrained)
    provenance = json.loads(args.pretrained.with_suffix(".json").read_text(encoding="utf-8"))
    if (pretrained_hash != provenance["checkpoint_sha256"] or provenance["branch"] != "params_ema"
            or provenance["config"] != model_config):
        raise ValueError("Pretrained checkpoint provenance mismatch")
    train = PairedDIV2K(args.dataset, "train", config["lr_patch_size"], config["seed"], config["hflip"], config["rotation"])
    valid = PairedDIV2K(args.dataset, "valid", config["lr_patch_size"], config["seed"])
    identity = {"dataset_manifest_sha256": train.manifest_sha256, "pretrained_sha256": pretrained_hash,
                "model_config": model_config, "conv_backend": "matmul",
                "reference_manifest_sha256": file_sha256(ROOT / "reference/manifest.json"),
                "source_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted((ROOT / "src/drct_ms").rglob("*.py"))}}
    model = DRCT(**model_config, conv_backend="matmul")
    load_checkpoint_strict(model, args.pretrained)
    if len(model.trainable_params()) != 916 or sum(p.size for p in model.trainable_params()) != 27580719:
        raise AssertionError("Full DRCT-L parameter audit failed")
    trainer = Trainer(model, config)
    report = {"purpose": config["purpose"], "environment": {"mindspore": ms.__version__, "python": platform.python_version(),
              "machine": platform.machine(), "mode": config["mode"], "device": "Ascend", "precision": "FP32 / HF32 off"},
              "config": config, "identity": identity, "train_ids": [p["id"] for p in train.pairs],
              "valid_ids": [p["id"] for p in valid.pairs], "modcrop_hr": train.modcrops + valid.modcrops,
              "updates": [], "parameter_tensors": 916, "parameter_elements": 27580719,
              "rng_resume": "deterministic data position; stochastic DropPath stream is not guaranteed across process restarts"}
    if args.resume:
        report["restore"] = trainer.restore(args.resume, identity)
    if trainer.step + args.steps > 20:
        raise ValueError("Smoke chain exceeds 20 total updates")
    report["initial_validation"] = trainer.validate(valid)
    sample = train.batch(trainer.step, config["batch_size"])
    report["next_batch_sha256"] = hashlib.sha256(sample[0].tobytes() + sample[1].tobytes()).hexdigest()
    ms.runtime.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.steps):
        lr, hr, audit = train.batch(trainer.step, config["batch_size"])
        row = trainer.update(lr, hr)
        row["samples"] = audit
        report["updates"].append(row)
        with (output / "train.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    report["training_wall_seconds"] = time.perf_counter() - start
    report["final_validation"] = trainer.validate(valid)
    report["ema_validation"] = trainer.validate(valid, use_ema=True)
    checkpoint = output / f"step_{trainer.step:06d}.ckpt"
    report["saved_checkpoint"] = trainer.save(checkpoint, identity)
    # Roundtrip audit before exit, followed by an independent resumed process in Stage4.
    report["roundtrip"] = trainer.restore(checkpoint, identity)
    restored_validation = trainer.validate(valid)
    if restored_validation != report["final_validation"]:
        raise AssertionError("Checkpoint roundtrip changed deterministic live validation")
    report["roundtrip_validation_exact"] = True
    next_batch = train.batch(trainer.step, config["batch_size"])
    report["saved_next_batch_sha256"] = hashlib.sha256(next_batch[0].tobytes() + next_batch[1].tobytes()).hexdigest()
    report["passed"] = True
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "completed_steps": trainer.step, "report": str(output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
