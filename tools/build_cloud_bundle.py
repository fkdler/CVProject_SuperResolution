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
    for relative in [
        "AGENTS.md", "README.md", "THIRD_PARTY_NOTICES.md", "pyproject.toml", "reference/manifest.json",
        "docs/verification/reference_audit.json", "docs/cloud_validation.md", "requirements/reference.txt",
        "docs/progress.md", "docs/migration.md", "docs/source_analysis.md",
        "docs/verification/checkpoint_audit.json", "data/README.md", "weights/README.md",
        "docs/verification/stage2.md",
        "docs/verification/ascend_stage2/alignment_pynative_matmul.json",
        "docs/verification/ascend_stage2/alignment_graph_matmul.json",
        "reference/DRCT/drct/archs/DRCT_arch.py", "reference/DRCT/LICENSE",
        "reference/DRCT/options/test/DRCT_SRx4.yml",
        "reference/DRCT/options/test/DRCT-L_SRx4_ImageNet-pretrain.yml",
        "reference/BasicSR/basicsr/archs/arch_util.py", "reference/BasicSR/LICENSE/LICENSE",
        "reference/BasicSR/LICENSE/README.md", "reference/BasicSR/LICENSE/LICENSE_pytorch-image-models",
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
