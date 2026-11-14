"""Generate official-PyTorch golden arrays on the cloud reference environment."""

import hashlib
import json

from tools.reference_loader import ROOT, load_reference


def main():
    from tools.cloud_guard import require_cloud_workspace
    require_cloud_workspace()
    import numpy as np
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(0)
    ref = load_reference()
    arrays = {}
    x = np.arange(2 * 32 * 48 * 3, dtype=np.float32).reshape(2, 32, 48, 3)
    arrays["window_x"] = x
    arrays["window_y"] = ref["window_partition"](torch.from_numpy(x), 16).numpy()
    rng = np.random.default_rng(42)
    x = (1 + 0.001 * rng.standard_normal((2, 12, 8, 12))).astype(np.float32)
    gamma = np.linspace(0.5, 1.5, 12, dtype=np.float32)
    beta = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
    patch = ref["PatchEmbed"](embed_dim=12, norm_layer=torch.nn.LayerNorm)
    with torch.no_grad():
        patch.norm.weight.copy_(torch.from_numpy(gamma))
        patch.norm.bias.copy_(torch.from_numpy(beta))
        arrays.update(patch_x=x, patch_gamma=gamma, patch_beta=beta, patch_y=patch(torch.from_numpy(x)).numpy())
    for scale in (2, 3):
        x = np.arange(2 * 3 * scale * scale * 4 * 5, dtype=np.float32).reshape(2, 3 * scale * scale, 4, 5)
        arrays[f"shuffle{scale}_x"] = x
        arrays[f"shuffle{scale}_y"] = torch.nn.PixelShuffle(scale)(torch.from_numpy(x)).numpy()
    rng = np.random.default_rng(7)
    mlp = ref["Mlp"](12, 24, 12, drop=0).eval()
    for name, parameter in mlp.named_parameters():
        value = rng.normal(0, 0.15, tuple(parameter.shape)).astype(np.float32)
        arrays[f"mlp_{name}"] = value
        with torch.no_grad():
            parameter.copy_(torch.from_numpy(value))
    x = rng.normal(0, 1, (2, 16, 12)).astype(np.float32)
    tx = torch.tensor(x, requires_grad=True)
    y = mlp(tx)
    y.sum().backward()
    arrays.update(mlp_x=x, mlp_y=y.detach().numpy(), mlp_input_grad=tx.grad.numpy())
    target = ROOT / "tests/fixtures/primitive_reference.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    manifest = json.loads((ROOT / "reference/manifest.json").read_text(encoding="utf-8"))
    metadata = {
        "generator": "python -m tools.generate_primitive_fixtures",
        "torch": torch.__version__, "numpy": np.__version__, "device": "CPU", "dtype": "float32",
        "source_sha256_lf": manifest["source_sha256"],
        "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "seeds": {"torch": 0, "patch_numpy": 42, "mlp_numpy": 7},
        "arrays": {key: list(value.shape) for key, value in arrays.items()},
        "purpose": "Reference outputs and MLP input gradients; not evidence of MindSpore runtime validation",
    }
    target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(target, target.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
