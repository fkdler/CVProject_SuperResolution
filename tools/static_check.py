"""Source-only local check. No project import, framework, tests, tensors, or bytecode execution."""

import ast
import json
from pathlib import Path
import sys
import tomllib


def main():
    root = Path(__file__).resolve().parents[1]
    paths = sorted(path for folder in ("src", "tools", "tests") for path in (root / folder).rglob("*.py"))
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if "src" in path.relative_to(root).parts:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) and any(n.name.split(".")[0] == "torch" for n in node.names):
                    raise ValueError(f"PyTorch inference dependency inside MindSpore source: {path}")
                if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "torch":
                    raise ValueError(f"PyTorch inference dependency inside MindSpore source: {path}")
    for path in [*sorted((root / "configs").glob("*.json")), root / "reference/manifest.json"]:
        json.loads(path.read_text(encoding="utf-8"))
    tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    report = {"check": "AST parse + JSON/TOML parse + no torch imports in src", "python": sys.version,
              "python_files": len(paths), "frameworks_imported": False, "numeric_tests_executed": False,
              "passed": True}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
