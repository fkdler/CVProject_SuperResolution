"""Fetch selected original PNG members via HTTP Range; standard-library file I/O only.

No pixel decoding, numeric experiments or framework imports. ZIP CRC and PNG SHA256
are retained; archive ETag/size identify the server snapshot (not a whole-ZIP hash).
DIV2K is for academic research: https://data.vision.ee.ethz.ch/cvl/DIV2K/
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.client import IncompleteRead
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"


def request_with_retry(request, consume):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=35) as response:
                return consume(response)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (OSError, EOFError, IncompleteRead):
            if attempt == 2:
                raise
        time.sleep(1 + attempt)


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url = url

        def metadata(response):
            return int(response.headers["Content-Length"]), response.headers.get("ETag")

        self.size, self.etag = request_with_retry(urllib.request.Request(url, method="HEAD"), metadata)
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
        content_range = f"bytes {self.position}-{self.position + size - 1}/{self.size}"

        def read_range(response):
            if (response.status != 206 or response.headers.get("ETag") != self.etag
                    or response.headers.get("Content-Range") != content_range):
                raise ValueError("Range unsupported or archive changed; refusing full download")
            data = response.read(size + 1)
            if len(data) != size:
                raise EOFError("Truncated/overlong range")
            return data

        data = request_with_retry(urllib.request.Request(self.url, headers=headers), read_range)
        self.position += size
        return data


def fetch_role(output, split, ids, role):
    archive_name = f"DIV2K_{split}_" + ("HR.zip" if role == "hr" else "LR_bicubic_X4.zip")
    records = {}
    with RemoteZip(BASE + archive_name) as remote, zipfile.ZipFile(remote) as archive:
        source = {"url": remote.url, "etag": remote.etag, "bytes": remote.size}
        for number in ids:
            name = (f"DIV2K_{split}_HR/{number:04d}.png" if role == "hr" else
                    f"DIV2K_{split}_LR_bicubic/X4/{number:04d}x4.png")
            info = archive.getinfo(name)
            relative = f"{split}/{role}/{number:04d}.png"
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                value = path.read_bytes()
                if len(value) != info.file_size or zlib.crc32(value) != info.CRC:
                    raise ValueError(f"Existing PNG differs from official ZIP member: {relative}")
            else:
                value = archive.read(info)  # zipfile verifies CRC32 without decoding pixels.
                temporary = path.with_suffix(".part")
                temporary.write_bytes(value)
                temporary.replace(path)
            records[number] = {"file": relative, "sha256": hashlib.sha256(value).hexdigest(),
                               "zip_member": name, "crc32": info.CRC, "bytes": len(value)}
            print(f"Fetched {relative}: {len(value)} bytes", flush=True)
    return source, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/DIV2K_smoke")
    parser.add_argument("--selection", type=Path, help="JSON with lists of official DIV2K train/valid IDs")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    selection = (json.loads(args.selection.read_text(encoding="utf-8")) if args.selection
                 else {"train": [1, 2, 3, 4], "valid": [801, 802]})
    if set(selection) != {"train", "valid"}:
        raise ValueError("Selection must contain exactly train and valid")
    for split, minimum, maximum in (("train", 1, 800), ("valid", 801, 900)):
        ids = selection[split]
        if not isinstance(ids, list) or not ids or any(type(n) is not int or not minimum <= n <= maximum for n in ids):
            raise ValueError(f"Invalid {split} IDs: expected integers in [{minimum}, {maximum}]")
        if ids != sorted(set(ids)):
            raise ValueError(f"{split} IDs must be sorted and unique")
    splits = ("train", "valid")
    expected = [(split, f"{n:04d}") for split in splits for n in selection[split]]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if (previous.get("source") != BASE or previous.get("scale") != 4
                or [(pair["split"], pair["id"]) for pair in previous["pairs"]] != expected):
            raise ValueError("Existing manifest has a different source, scale or ID selection")
        for pair in previous["pairs"]:
            for role in ("lr", "hr"):
                path = (args.output / pair[role]["file"]).resolve()
                if not path.is_relative_to(args.output.resolve()):
                    raise ValueError("Manifest path escapes dataset")
                value = path.read_bytes()
                if hashlib.sha256(value).hexdigest() != pair[role]["sha256"]:
                    raise ValueError("Existing data integrity failure")
        print("Existing DIV2K files verified")
        return
    pairs, archives = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {}
        for split in splits:
            for role in ("hr", "lr"):
                ids = selection[split]
                # Train HR is by far the largest archive read. Each shard owns its ZIP cursor.
                shards = min(args.workers, len(ids)) if (split, role) == ("train", "hr") else 1
                jobs[split, role] = [
                    pool.submit(fetch_role, args.output, split, ids[index::shards], role)
                    for index in range(shards)
                ]
        for split in splits:
            members = {number: {"id": f"{number:04d}", "split": split} for number in selection[split]}
            for role in ("hr", "lr"):
                sources = []
                for job in jobs[split, role]:
                    archive, records = job.result()
                    sources.append(archive)
                    for number, record in records.items():
                        members[number][role] = record
                if any(source != sources[0] for source in sources[1:]):
                    raise ValueError(f"Official archive changed across {split} {role} download shards")
                archives.append(sources[0])
            pairs.extend(members.values())
    metadata = {"dataset": "DIV2K", "source": BASE, "scale": 4, "degradation": "official bicubic X4",
                "academic_research_only": True, "archives": archives, "pairs": pairs}
    temporary = manifest.with_suffix(".part")
    # Preserve the legacy smoke manifest's platform newline convention for old resume hashes.
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
                         newline="\n" if args.selection else None)
    temporary.replace(manifest)
    print(f"Saved {len(pairs)} pairs; no images decoded")


if __name__ == "__main__":
    main()
