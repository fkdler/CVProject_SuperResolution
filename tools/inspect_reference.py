"""Instantiate upstream models on CPU and write an audit (no training)."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

from tools.reference_loader import ROOT, load_reference


def main():
    import torch
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/verification/reference_audit.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    ref = load_reference()
    audit = {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU", "models": {}}
    for name, config in [("base", "DRCT_SRx4.yml"), ("large", "DRCT-L_SRx4_ImageNet-pretrain.yml")]:
        path = ROOT / "reference/DRCT/options/test" / config
        options = yaml.safe_load(path.read_text(encoding="utf-8"))["network_g"]
        options.pop("type")
        model = ref["DRCT"](**options).eval()
        group = model.layers[0]
        blocks = [getattr(group, f"swin{i}") for i in range(1, 6)]
        audit["models"][name] = {
            "config": path.relative_to(ROOT).as_posix(),
            "config_sha256_lf": hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "state_dict_entries": len(model.state_dict()),
            "rdg_count": len(model.layers), "swin_count": len(model.layers) * len(blocks),
            "group_channels": [block.dim for block in blocks],
            "group_heads": [block.num_heads for block in blocks],
            "group_shifts": [block.shift_size for block in blocks],
            "group_mlp_ratios": [block.mlp_ratio for block in blocks],
            "drop_path_per_group": [getattr(layer.swin1.drop_path, "drop_prob", 0.0) for layer in model.layers],
            "state_dict": {k: list(v.shape) for k, v in model.state_dict().items()},
        }
        print(name, {k: v for k, v in audit["models"][name].items() if k != "state_dict"})
        del model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
