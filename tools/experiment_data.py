"""Cloud-only DIV2K manifest validation and bounded decoding for experiments."""

import json
from functools import lru_cache
from pathlib import Path

from tools.cloud_guard import require_cloud_workspace

require_cloud_workspace()

import numpy as np

from drct_ms.evaluation import file_sha256, read_bgr
from drct_ms.training_data import paired_crop
from tools.fetch_div2k_smoke import BASE


class ExperimentDIV2K:
    def __init__(self, directory, selection):
        directory = Path(directory).resolve()
        manifest = directory / "manifest.json"
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if (metadata["dataset"], metadata["source"], metadata["scale"], metadata["degradation"]) != (
            "DIV2K", BASE, 4, "official bicubic X4"
        ):
            raise ValueError("Expected the official DIV2K bicubic x4 manifest")
        expected = []
        for split, lower, upper in (("train", 1, 800), ("valid", 801, 900)):
            ids = selection[split]
            if not ids or ids != sorted(set(ids)) or any(type(i) is not int or not lower <= i <= upper for i in ids):
                raise ValueError(f"Invalid official DIV2K {split} IDs")
            expected.extend((split, f"{i:04d}") for i in ids)
        self.pairs = metadata["pairs"]
        if [(pair["split"], pair["id"]) for pair in self.pairs] != expected:
            raise ValueError("DIV2K manifest must match the predeclared, disjoint ID lists exactly")
        self.paths = {}
        for pair in self.pairs:
            paths = []
            for role in ("lr", "hr"):
                entry = pair[role]
                path = (directory / entry["file"]).resolve()
                if not path.is_relative_to(directory) or file_sha256(path) != entry["sha256"]:
                    raise ValueError(f"DIV2K file failed integrity check: {pair['id']} {role}")
                paths.append(path)
            self.paths[pair["id"]] = paths
        self.manifest_sha256 = file_sha256(manifest)
        self.train_ids = [pair["id"] for pair in self.pairs if pair["split"] == "train"]
        self.valid_pairs = [pair for pair in self.pairs if pair["split"] == "valid"]
        self.read_pair = lru_cache(maxsize=4)(self._read_pair)

    def _read_pair(self, image_id):
        lr, hr = (read_bgr(path) for path in self.paths[image_id])
        h, w = lr.shape[:2]
        bottom, right = hr.shape[0] - 4 * h, hr.shape[1] - 4 * w
        if not (0 <= bottom < 4 and 0 <= right < 4):
            raise ValueError(f"Invalid DIV2K x4 dimensions: {image_id}")
        return lr, hr[:4 * h, :4 * w], {"bottom_hr": bottom, "right_hr": right}

    def batch(self, step, config):
        size, seed, patch = config["batch_size"], config["seed"], config["lr_patch_size"]
        crops_lr, crops_hr, audit = [], [], []
        for index in range(step * size, (step + 1) * size):
            epoch, offset = divmod(index, len(self.train_ids))
            order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 0])).permutation(len(self.train_ids))
            image_id = self.train_ids[int(order[offset])]
            rng = np.random.default_rng(np.random.SeedSequence([seed, index, 1]))
            lr, hr, trim = self.read_pair(image_id)
            a, b, crop = paired_crop(lr, hr, patch, rng, hflip=config["hflip"], rotation=config["rotation"])
            crops_lr.append(a)
            crops_hr.append(b)
            audit.append({"id": image_id, "global_sample": index, "epoch": epoch, **trim, **crop})
        return np.stack(crops_lr), np.stack(crops_hr), audit
