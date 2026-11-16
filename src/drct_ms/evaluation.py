"""Cloud-only benchmark I/O and BasicSR-compatible evaluation (no model backend).

Metric equations adapted from BasicSR 1.3.4.9; see THIRD_PARTY_NOTICES.md.
Arrays are used only for decoding, padding, quantization and metrics.
"""

from .runtime import require_cloud_workspace

require_cloud_workspace()

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

PROTOCOLS = {
    "y_shave4": {"channel": "BT.601 Y", "crop_border_hr": 4},
    "rgb_shave8": {"channel": "RGB", "crop_border_hr": 8},
}


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_bgr(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Cannot decode an 8-bit BGR image: {path}")
    return image


def load_dataset(directory):
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_source = json.loads((require_cloud_workspace() / "configs/set14_source.json").read_text(encoding="utf-8"))
    if metadata["source"] != expected_source or len(metadata["pairs"]) != 14:
        raise ValueError("Dataset provenance/count differs from pinned Set14")
    if [p["id"] for p in metadata["pairs"]] != [f"img_{i:03d}" for i in range(1, 15)]:
        raise ValueError("Dataset pair IDs are duplicated, missing or out of order")
    for pair in metadata["pairs"]:
        for role in ("hr", "lr"):
            path = (directory / pair[role]["file"]).resolve()
            if pair[role]["png"]["bit_depth"] != 8:
                raise ValueError("Only the pinned 8-bit PNG benchmark is supported")
            if not path.is_relative_to(directory) or file_sha256(path) != pair[role]["sha256"]:
                raise ValueError(f"Dataset file integrity failed: {pair['id']} {role}")
            image = read_bgr(path)
            if list(image.shape[:2]) != [pair[role]["png"]["height"], pair[role]["png"]["width"]]:
                raise ValueError("Decoded dimensions differ from pinned PNG header")
        h, w = pair["lr"]["png"]["height"], pair["lr"]["png"]["width"]
        if pair["hr"]["png"]["height"] != 4 * h or pair["hr"]["png"]["width"] != 4 * w:
            raise ValueError("HR/LR scale mismatch; no silent cropping or resampling")
    return metadata, file_sha256(manifest_path)


def input_from_bgr(image):
    return np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)


def reflect_pad_to_window(image, window=16):
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3 or image.dtype != np.float32:
        raise ValueError("Expected one RGB NCHW float32 image")
    h, w = image.shape[2:]
    ph, pw = (-h) % window, (-w) % window
    if ph >= h or pw >= w:
        raise ValueError("Reflect padding must be smaller than the input dimension")
    padded = np.pad(image, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="reflect")
    return np.ascontiguousarray(padded), (h, w)


def unpad_output(output, original_size, scale=4):
    h, w = original_size
    return np.ascontiguousarray(output[:, :, :h * scale, :w * scale])


def to_bgr_uint8(output):
    if output.ndim != 4 or output.shape[:2] != (1, 3) or not np.isfinite(output).all():
        raise ValueError("Expected finite 1x3xHxW model output")
    output = np.clip(output.astype(np.float32), 0, 1)
    return np.ascontiguousarray((output[0].transpose(1, 2, 0)[:, :, ::-1] * 255.0).round().astype(np.uint8))


def to_y_channel(image):
    value = image.astype(np.float32) / 255.0
    # Match BasicSR's float32 input -> float64 dot -> float32 output range path.
    value = np.dot(value, [24.966, 128.553, 65.481]) + 16.0
    return ((value / 255.0).astype(np.float32) * 255.0)[..., None]


def metric_images(image, target, protocol):
    if image.shape != target.shape or image.dtype != np.uint8 or target.dtype != np.uint8:
        raise ValueError("Metrics require paired same-shape uint8 BGR images")
    spec = PROTOCOLS[protocol]
    border = spec["crop_border_hr"]
    if min(image.shape[:2]) <= 2 * border + 10:
        raise ValueError("Image too small after metric crop/SSIM valid window")
    a = image[border:-border, border:-border].astype(np.float64)
    b = target[border:-border, border:-border].astype(np.float64)
    if spec["channel"] == "BT.601 Y":
        a, b = to_y_channel(a), to_y_channel(b)
    return a, b


def metrics(image, target, protocol="y_shave4"):
    a, b = metric_images(image, target, protocol)
    mse = np.mean((a - b) ** 2)
    psnr = float("inf") if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    values = []
    for channel in range(a.shape[2]):
        x, y = a[..., channel].astype(np.float64), b[..., channel].astype(np.float64)
        mx = cv2.filter2D(x, -1, window)[5:-5, 5:-5]
        my = cv2.filter2D(y, -1, window)[5:-5, 5:-5]
        vx = cv2.filter2D(x * x, -1, window)[5:-5, 5:-5] - mx * mx
        vy = cv2.filter2D(y * y, -1, window)[5:-5, 5:-5] - my * my
        covariance = cv2.filter2D(x * y, -1, window)[5:-5, 5:-5] - mx * my
        values.append(np.mean(((2 * mx * my + c1) * (2 * covariance + c2)) /
                              ((mx * mx + my * my + c1) * (vx + vy + c2))))
    return {"psnr": psnr, "ssim": float(np.mean(values))}
