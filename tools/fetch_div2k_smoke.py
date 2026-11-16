"""Fetch selected original PNG members via HTTP Range; standard-library file I/O only.

No pixel decoding, numeric experiments or framework imports. ZIP CRC and PNG SHA256
are retained; archive ETag/size identify the server snapshot (not a whole-ZIP hash).
DIV2K is for academic research: https://data.vision.ee.ethz.ch/cvl/DIV2K/
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
            self.etag = response.headers.get("ETag")
        self.position = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if self.position < 0:
            raise ValueError("Negative offset")
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        if size > 64 * 1024 * 1024:
            raise ValueError("Refusing a large archive read; only selected PNG members are needed")
        headers = {"Range": f"bytes={self.position}-{self.position + size - 1}",
                   "User-Agent": "DRCT-MindSpore-course"}
        if self.etag:
            headers["If-Range"] = self.etag
        with urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=120) as response:
            if response.status != 206 or response.headers.get("ETag") != self.etag:
                raise ValueError("Range unsupported or archive changed; refusing full download")
            data = response.read(size + 1)
        if len(data) != size:
            raise ValueError("Truncated/overlong range")
        self.position += size
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/DIV2K_smoke")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        for pair in previous["pairs"]:
            for role in ("lr", "hr"):
                value = (args.output / pair[role]["file"]).read_bytes()
                if hashlib.sha256(value).hexdigest() != pair[role]["sha256"]:
                    raise ValueError("Existing data integrity failure")
        print("Existing DIV2K smoke files verified")
        return
    pairs, archives = [], []
    for split, ids in (("train", [1, 2, 3, 4]), ("valid", [801, 802])):
        members = {number: {"id": f"{number:04d}", "split": split} for number in ids}
        for role in ("hr", "lr"):
            archive_name = f"DIV2K_{split}_" + ("HR.zip" if role == "hr" else "LR_bicubic_X4.zip")
            remote = RemoteZip(BASE + archive_name)
            archives.append({"url": remote.url, "etag": remote.etag, "bytes": remote.size})
            with zipfile.ZipFile(remote) as archive:
                for number in ids:
                    name = (f"DIV2K_{split}_HR/{number:04d}.png" if role == "hr" else
                            f"DIV2K_{split}_LR_bicubic/X4/{number:04d}x4.png")
                    info = archive.getinfo(name)
                    value = archive.read(info)  # zipfile verifies CRC32, without decoding pixels.
                    relative = f"{split}/{role}/{number:04d}.png"
                    path = args.output / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists() and path.read_bytes() != value:
                        raise ValueError("Refusing to overwrite different PNG")
                    path.write_bytes(value)
                    members[number][role] = {"file": relative, "sha256": hashlib.sha256(value).hexdigest(),
                                             "zip_member": name, "crc32": info.CRC, "bytes": len(value)}
                    print(f"Fetched {relative}: {len(value)} bytes", flush=True)
            remote.close()
        pairs.extend(members.values())
    metadata = {"dataset": "DIV2K", "source": BASE, "scale": 4, "degradation": "official bicubic X4",
                "academic_research_only": True, "archives": archives, "pairs": pairs}
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(pairs)} pairs; no images decoded")


if __name__ == "__main__":
    main()
