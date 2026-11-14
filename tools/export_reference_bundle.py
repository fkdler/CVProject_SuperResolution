"""Cloud reference environment: export official weights and bounded FP32 comparisons.

Writes NumPy files so MindSpore and PyTorch can use separate Python environments.
Default inference inputs are small synthetic LR arrays, not Set14 measurements.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

from tools.cloud_guard import ROOT, require_cloud_workspace


def main():
    require_cloud_workspace()
    import numpy as np
    import torch

    from tools.reference_loader import load_reference

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/DRCT-L_X4.pth")
    parser.add_argument("--branch", choices=["params", "params_ema"], default="params_ema")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/reference_bundle")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path("/home/ma-user/work")):
        raise ValueError("Outputs must remain on the persistent work volume")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory; an existing bundle will not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    source = json.loads((ROOT / "reference/manifest.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "configs/drct_l_x4.json").read_text(encoding="utf-8"))["network_g"]
    with args.checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != source["checkpoint"]["sha256"]:
        raise ValueError("Official checkpoint hash mismatch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.branch not in checkpoint:
        raise ValueError("Requested checkpoint branch is absent; no automatic fallback")
    model = load_reference()["DRCT"](**config).eval()
    model.load_state_dict(checkpoint[args.branch], strict=True)
    parameter_names = [name for name, _ in model.named_parameters()]
    np.savez(output / "state.npz", **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
    rng = np.random.default_rng(20261114)
    cases = []
    # Cover changed spatial shape and nominal resolution, never a reduced-size network.
    for label, shape in [("rectangular", (1, 3, 32, 48)), ("nominal", (1, 3, 64, 64))]:
        value = rng.uniform(0, 1, shape).astype(np.float32)
        tensors = {"input": value}
        handles = []
        for name, module in [("conv_first", model.conv_first), ("patch_embed", model.patch_embed),
                             ("rdg0", model.layers[0]), ("rdg11", model.layers[11]),
                             ("norm", model.norm), ("conv_after_body", model.conv_after_body),
                             ("swin1", model.layers[0].swin1), ("swin2", model.layers[0].swin2),
                             ("swin3", model.layers[0].swin3), ("swin4", model.layers[0].swin4),
                             ("swin5", model.layers[0].swin5)]:
            def capture(_module, _inputs, result, key=name):
                tensors[key] = result.detach().cpu().numpy().copy()
            handles.append(module.register_forward_hook(capture))
        start = time.perf_counter()
        try:
            with torch.no_grad():
                result = model(torch.from_numpy(value))
            tensors["output"] = result.numpy()
        finally:
            for handle in handles:
                handle.remove()
        if not all(np.isfinite(array).all() for array in tensors.values()):
            raise ValueError("Nonfinite reference values")
        filename = label + ".npz"
        np.savez(output / filename, **tensors)
        with (output / filename).open("rb") as stream:
            case_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        cases.append({"name": label, "file": filename, "shape": list(shape), "sha256": case_hash,
                      "seconds": time.perf_counter() - start})
    with (output / "state.npz").open("rb") as stream:
        state_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {"schema": 1, "checkpoint_sha256": digest, "branch": args.branch,
                "source_commit": source["repositories"][0]["commit"], "config": config,
                "torch": torch.__version__, "device": "cloud host CPU", "threads": args.threads,
                "seed": 20261114, "parameter_names": parameter_names, "state_sha256": state_hash,
                "cases": cases}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in metadata.items() if k != "parameter_names"}, indent=2))


if __name__ == "__main__":
    main()
