"""Cloud-only deterministic paired RGB crops from official DIV2K splits.

Paired crop and horizontal/vertical/transpose augmentation follow BasicSR's
training conventions; deterministic per-sample RNG permits data-position resume.
"""

from .runtime import require_cloud_workspace

require_cloud_workspace()

import json
from pathlib import Path

import numpy as np

from .evaluation import file_sha256, read_bgr


def paired_crop(lr, hr, patch, rng, augment=True, hflip=True, rotation=True):
    h, w = lr.shape[:2]
    if min(h, w) < patch or hr.shape[:2] != (h * 4, w * 4):
        raise ValueError("Paired crop shape mismatch")
    y, x = (int(rng.integers(h - patch + 1)), int(rng.integers(w - patch + 1))) if augment else (
        (h - patch) // 2, (w - patch) // 2)
    a, b = lr[y:y + patch, x:x + patch], hr[4*y:4*(y+patch), 4*x:4*(x+patch)]
    flags = [bool(rng.integers(2)) if augment and enabled else False for enabled in (hflip, rotation, rotation)]
    values = []
    for image in (a, b):
        if flags[0]:
            image = image[:, ::-1]
        if flags[1]:
            image = image[::-1]
        if flags[2]:
            image = image.transpose(1, 0, 2)
        values.append(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0)
    return (*values, {"lr_top": y, "lr_left": x, "hflip_vflip_transpose": flags})


class PairedDIV2K:
    def __init__(self, directory, split, patch=64, seed=0, hflip=True, rotation=True):
        self.directory = Path(directory).resolve()
        metadata = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if metadata["dataset"] != "DIV2K" or metadata["source"] != "https://data.vision.ee.ethz.ch/cvl/DIV2K/" or metadata["scale"] != 4:
            raise ValueError("Only explicit official DIV2K provenance is accepted")
        if split not in {"train", "valid"} or patch < 16 or patch % 16:
            raise ValueError("Invalid split/patch (patch must be a window multiple)")
        seen = set()
        for pair in metadata["pairs"]:
            number = int(pair["id"])
            expected_split = "train" if 1 <= number <= 800 else "valid" if 801 <= number <= 900 else None
            if pair["split"] != expected_split or number in seen:
                raise ValueError("Duplicate, overlapping or non-official train/validation IDs")
            seen.add(number)
        self.pairs = [p for p in metadata["pairs"] if p["split"] == split]
        if not self.pairs:
            raise ValueError("Empty dataset split")
        self.images, self.modcrops = [], []
        for pair in self.pairs:
            decoded = {}
            for role in ("lr", "hr"):
                path = (self.directory / pair[role]["file"]).resolve()
                if not path.is_relative_to(self.directory) or file_sha256(path) != pair[role]["sha256"]:
                    raise ValueError("Data integrity/path check failed")
                decoded[role] = read_bgr(path)
            lr, hr = decoded["lr"], decoded["hr"]
            h, w = lr.shape[:2]
            dh, dw = hr.shape[0] - 4*h, hr.shape[1] - 4*w
            if not (0 <= dh < 4 and 0 <= dw < 4) or min(h, w) < patch:
                raise ValueError("HR/LR scale mismatch or patch too large")
            self.modcrops.append({"id": pair["id"], "bottom_hr": dh, "right_hr": dw})
            self.images.append((lr, hr[:4*h, :4*w]))
        self.manifest_sha256 = file_sha256(self.directory / "manifest.json")
        self.split, self.patch, self.seed = split, patch, seed
        self.hflip, self.rotation = hflip, rotation

    def sample(self, index):
        if index < 0:
            raise ValueError("Negative sample index")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index, 1]))
        epoch, offset = divmod(index, len(self.pairs))
        if self.split == "train":
            permutation = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 0])).permutation(len(self.pairs))
            selected = int(permutation[offset])
        else:
            selected = offset
        lr, hr, audit = paired_crop(*self.images[selected], self.patch, rng, self.split == "train", self.hflip, self.rotation)
        audit.update({"id": self.pairs[selected]["id"], "global_sample": index, "epoch": epoch})
        return lr, hr, audit

    def batch(self, step, size):
        values = [self.sample(step * size + j) for j in range(size)]
        return np.stack([v[0] for v in values]), np.stack([v[1] for v in values]), [v[2] for v in values]
