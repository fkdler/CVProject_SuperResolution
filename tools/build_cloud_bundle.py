"""Package source files for transfer. Standard library/file I/O only; no ML execution."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/cloud_source.tar")
    args = parser.parse_args()
    selected = set()
    for directory in ("src", "tools", "tests", "configs"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json", ".npz"}:
                selected.add(path)
    selected.update((ROOT / "docs/verification/experiment_baseline").glob("*.json"))
    for relative in [
        "AGENTS.md", "README.md", "THIRD_PARTY_NOTICES.md", "pyproject.toml", "reference/manifest.json",
        "licenses/BasicSR-Apache-2.0.txt",
        "docs/verification/reference_audit.json", "docs/cloud_validation.md", "requirements/reference.txt",
        "docs/progress.md", "docs/migration.md", "docs/source_analysis.md",
        "docs/verification/checkpoint_audit.json", "data/README.md", "weights/README.md",
        "docs/verification/stage2.md",
        "docs/stage4_training.md",
        "docs/experiment_baseline.md",
        "docs/verification/stage4.md",
        "docs/verification/stage4/acceptance.json",
        "docs/verification/stage4/dataset_manifest.json",
        "docs/verification/stage4/environment.json",
        "docs/verification/stage4/first_report.json",
        "docs/verification/stage4/resumed_report.json",
        "docs/verification/stage4/resource_observation.json",
        "docs/verification/stage4/pytest.log",
        "docs/verification/stage4/static_check.json",
        "docs/verification/stage4/npu_snapshot.log",
        "docs/verification/stage4/native_adam_epsilon_failure.log",
        "docs/verification/stage4/native_adam_first_report.json",
        "docs/verification/stage4/native_adam_resumed_report.json",
        "docs/stage3_evaluation.md",
        "docs/verification/stage3.md",
        "docs/verification/stage3/dataset_manifest.json",
        "docs/verification/stage3/evaluation_parity.json",
        "docs/verification/stage3/comparison.json",
        "docs/verification/stage3/comparison.csv",
        "docs/verification/stage3/mindspore_report.json",
        "docs/verification/stage3/mindspore_per_image.csv",
        "docs/verification/stage3/torch_report.json",
        "docs/verification/stage3/torch_per_image.csv",
        "docs/verification/stage3/params_diagnostic_report.json",
        "docs/verification/stage3/params_alignment.json",
        "docs/verification/stage3/params_conversion.json",
        "docs/verification/ascend_stage2/alignment_pynative_matmul.json",
        "docs/verification/ascend_stage2/alignment_graph_matmul.json",
        "docs/verification/ascend_stage2/drct_l_x4_params_ema.json",
        "reference/DRCT/drct/archs/DRCT_arch.py", "reference/DRCT/LICENSE",
        "reference/DRCT/options/test/DRCT_SRx4.yml",
        "reference/DRCT/options/test/DRCT-L_SRx4_ImageNet-pretrain.yml",
        "reference/DRCT/options/train/train_DRCT-L_SRx4_finetune_from_ImageNet_pretrain.yml",
        "reference/BasicSR/basicsr/archs/arch_util.py", "reference/BasicSR/LICENSE/LICENSE",
        "reference/BasicSR/LICENSE/README.md", "reference/BasicSR/LICENSE/LICENSE_pytorch-image-models",
        "reference/BasicSR/basicsr/metrics/psnr_ssim.py",
        "reference/BasicSR/basicsr/metrics/metric_util.py",
        "reference/BasicSR/basicsr/utils/matlab_functions.py",
        "reference/BasicSR/basicsr/utils/img_util.py",
    ]:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        selected.add(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w") as archive:
        for path in sorted(selected):
            archive.add(path, arcname="drct_ms/" + path.relative_to(ROOT).as_posix(), recursive=False)
    with args.output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"artifact": args.output.name, "sha256": digest, "bytes": args.output.stat().st_size,
              "includes_weights": False, "includes_frameworks": False,
              "files": [p.relative_to(ROOT).as_posix() for p in sorted(selected)]}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
