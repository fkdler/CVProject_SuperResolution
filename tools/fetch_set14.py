"""Download immutable test files and record hashes. Standard library/file I/O only.

Permitted locally as file preparation. No image decoding, resizing, metrics,
framework import, or model execution. Full pairing checks run in the cloud.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def png_header(path):
    with path.open("rb") as stream:
        header = stream.read(29)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a PNG: {path}")
    width, height, bits, color = struct.unpack(">IIBB", header[16:26])
    return {"width": width, "height": height, "bit_depth": bits, "color_type": color}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/Set14")
    args = parser.parse_args()
    source = json.loads((ROOT / "configs/set14_source.json").read_text(encoding="utf-8"))
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["source"] != source:
            raise ValueError("Existing dataset has different provenance")
        for pair in previous["pairs"]:
            for role in ("hr", "lr"):
                if digest(destination / pair[role]["file"]) != pair[role]["sha256"]:
                    raise ValueError("Existing dataset hash mismatch; files were not replaced")
        print("Existing pinned dataset verified")
        return

    def fetch(item):
        number, role = item
        filename = f"img_{number:03d}_SRF_4_{role.upper()}.png"
        url = source["raw_root"] + "/" + filename
        path = destination / filename
        temporary = path.with_suffix(".part")
        request = urllib.request.Request(url, headers={"User-Agent": "DRCT-MindSpore-course-reproduction"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        png_header(temporary)
        temporary.replace(path)
        return number, role, {"file": filename, "url": url, "sha256": digest(path),
                              "bytes": path.stat().st_size, "png": png_header(path)}

    jobs = [(i, role) for i in range(1, source["count"] + 1) for role in ("hr", "lr")]
    pairs = {i: {"id": f"img_{i:03d}"} for i in range(1, source["count"] + 1)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for number, role, entry in pool.map(fetch, jobs):
            pairs[number][role] = entry
    report = {"schema": 1, "source": source, "pairs": list(pairs.values()),
              "preparation": "Original immutable PNG bytes; no decoding, resampling or pixel computation"}
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(pairs), "bytes": sum(p[r]["bytes"] for p in pairs.values() for r in ("hr", "lr")),
                      "dimensions": [(p["id"], p["hr"]["png"], p["lr"]["png"]) for p in pairs.values()]}, indent=2))


if __name__ == "__main__":
    main()
