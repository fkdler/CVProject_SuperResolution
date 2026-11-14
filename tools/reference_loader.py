"""Load official network definitions without installing the BasicSR training stack.

Only AST function/class definitions are executed from the pinned DRCT file.
The registry decorator is removed in memory. BasicSR's two needed helpers are
loaded from its pinned source. Network bodies and upstream files are unchanged.
This is a development adapter, not an official end-to-end evaluation runner.
"""

import ast
import collections.abc
import hashlib
import json
import math
from pathlib import Path
from itertools import repeat
import warnings

ROOT = Path(__file__).resolve().parents[1]


def checked_source(relative_path: str) -> Path:
    manifest = json.loads((ROOT / "reference/manifest.json").read_text(encoding="utf-8"))
    expected = manifest["source_sha256"][relative_path]
    path = ROOT / relative_path
    actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Upstream source has changed: {relative_path}")
    return path


def load_reference():
    import torch

    namespace = {
        "torch": torch, "nn": torch.nn, "math": math, "warnings": warnings,
        "collections": collections, "repeat": repeat,
    }
    util_path = checked_source("reference/BasicSR/basicsr/archs/arch_util.py")
    util_tree = ast.parse(util_path.read_text(encoding="utf-8"))
    helpers = [node for node in util_tree.body if isinstance(node, ast.FunctionDef)
               and node.name in {"_ntuple", "_no_grad_trunc_normal_", "trunc_normal_"}]
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(util_path), "exec"), namespace)
    namespace["to_2tuple"] = namespace["_ntuple"](2)

    arch_path = checked_source("reference/DRCT/drct/archs/DRCT_arch.py")
    tree = ast.parse(arch_path.read_text(encoding="utf-8"))
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    for node in definitions:
        if isinstance(node, ast.ClassDef) and node.name == "DRCT":
            node.decorator_list = []
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(arch_path), "exec"), namespace)
    return namespace
