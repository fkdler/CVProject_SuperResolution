"""Cloud Ascend environment: strictly convert a reference NumPy bundle to .ckpt."""

import argparse
import json
import os
from pathlib import Path

from tools.cloud_guard import ROOT, require_cloud_workspace


def main():
    require_cloud_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "experiments/reference_bundle")
    parser.add_argument("--output", type=Path, default=ROOT / "weights/drct_l_x4_params_ema.ckpt")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path("/home/ma-user/work")) or output.suffix != ".ckpt":
        raise ValueError("Output must be a .ckpt under /home/ma-user/work")
    if output.exists():
        raise FileExistsError("Choose a new checkpoint output; existing checkpoints are not overwritten")
    from drct_ms.runtime import configure_ascend
    ms = configure_ascend()
    import numpy as np
    from drct_ms.models.drct import DRCT
    from drct_ms.checkpoint import load_bundle, load_checkpoint_strict, sha256

    metadata = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
    pinned = json.loads((ROOT / "reference/manifest.json").read_text(encoding="utf-8"))
    expected_config = json.loads((ROOT / "configs/drct_l_x4.json").read_text(encoding="utf-8"))["network_g"]
    if metadata["checkpoint_sha256"] != pinned["checkpoint"]["sha256"]:
        raise ValueError("Bundle is not from the pinned official checkpoint")
    if metadata["config"] != expected_config or metadata["source_commit"] != pinned["repositories"][0]["commit"]:
        raise ValueError("Bundle architecture/source differs from the pinned baseline")
    net = DRCT(**metadata["config"])
    report = load_bundle(net, args.bundle)
    if report["target_parameter_elements"] != 27580719:
        raise ValueError("Unexpected DRCT-L parameter count")
    net.set_train(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial.ckpt")
    ms.save_checkpoint(net, str(temporary))
    # Strict reloading also checks the saved serialization names and types.
    fresh = DRCT(**metadata["config"])
    load_checkpoint_strict(fresh, temporary)
    for name, value in net.parameters_dict().items():
        if not np.array_equal(value.asnumpy(), fresh.parameters_dict()[name].asnumpy()):
            raise ValueError(f"Saved checkpoint changed parameter {name}")
    os.replace(temporary, output)
    report.update(checkpoint=str(output), checkpoint_sha256=sha256(output), mindspore=ms.__version__,
                  config=metadata["config"], source_commit=metadata["source_commit"],
                  serialization_roundtrip="passed", numeric_forward_validation="pending")
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "mapping"}, indent=2))


if __name__ == "__main__":
    main()
