"""Audit the official checkpoint safely on CPU; never fall back from EMA silently."""

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

from tools.reference_loader import ROOT, load_reference


def main():
    from tools.cloud_guard import require_cloud_workspace
    require_cloud_workspace()
    import torch
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/verification/checkpoint_audit.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    options = yaml.safe_load((ROOT / "reference/DRCT/options/test/DRCT-L_SRx4_ImageNet-pretrain.yml").read_text())
    network = options["network_g"]
    network.pop("type")
    model = load_reference()["DRCT"](**network).eval()
    expected = model.state_dict()
    with args.checkpoint.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = json.loads((ROOT / "reference/manifest.json").read_text(encoding="utf-8"))
    known_hash = manifest["checkpoint"].get("sha256")
    if known_hash and sha256 != known_hash:
        raise RuntimeError("Checkpoint SHA-256 differs from the pinned official artifact")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Expected a checkpoint mapping")
    report = {
        "checkpoint": args.checkpoint.name, "bytes": args.checkpoint.stat().st_size, "sha256": sha256,
        "source_url": manifest["checkpoint"]["url"],
        "source_commit": manifest["repositories"][0]["commit"],
        "torch": torch.__version__, "top_level_keys": list(checkpoint),
        "yaml_requested_key": options["path"]["param_key_g"], "branches": {},
    }
    for key in ("params", "params_ema"):
        if key not in checkpoint:
            continue
        state = checkpoint[key]
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        mismatched = {k: {"expected": list(expected[k].shape), "actual": list(state[k].shape)}
                      for k in expected.keys() & state.keys() if expected[k].shape != state[k].shape}
        buffers = {k: not torch.equal(state[k], expected[k]) for k in expected.keys() & state.keys()
                   if (k.endswith("relative_position_index") or k.endswith("attn_mask")) and k not in mismatched}
        finite = all(bool(torch.isfinite(v).all()) for v in state.values() if v.is_floating_point())
        branch = {"entries": len(state), "missing": missing, "unexpected": unexpected,
                  "shape_mismatches": mismatched, "geometry_buffer_mismatches": [k for k, differs in buffers.items() if differs],
                  "all_floating_tensors_finite": finite}
        if not missing and not unexpected and not mismatched:
            model.load_state_dict(state, strict=True)
            branch["strict_load"] = "passed"
        else:
            branch["strict_load"] = "not attempted: mismatch"
        report["branches"][key] = branch
    report["yaml_requested_key_available"] = report["yaml_requested_key"] in checkpoint
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["branches"] or any(b["strict_load"] != "passed" or not b["all_floating_tensors_finite"]
                                     or b["geometry_buffer_mismatches"] for b in report["branches"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
