"""Fetch pinned official checkouts without modifying an existing checkout."""

import json
import subprocess

from tools.reference_loader import ROOT, checked_source


def git(path, *args):
    # Per-command trust only for this known checkout, useful on Windows sandbox accounts.
    return subprocess.run(["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path), *args],
                          check=True, text=True, capture_output=True).stdout.strip()


def main():
    manifest = json.loads((ROOT / "reference/manifest.json").read_text(encoding="utf-8"))
    for repo in manifest["repositories"]:
        path = ROOT / repo["directory"]
        if not path.exists():
            subprocess.run(["git", "clone", "--no-checkout", repo["url"], str(path)], check=True)
            git(path, "checkout", "--detach", repo["commit"])
        if git(path, "rev-parse", "HEAD") != repo["commit"]:
            raise RuntimeError(f"Unexpected version in {path}; existing files were not changed")
        if git(path, "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError(f"Upstream tracked files have local edits: {path}")
        print(repo["name"], repo["commit"])
    for source in manifest["source_sha256"]:
        checked_source(source)


if __name__ == "__main__":
    main()
