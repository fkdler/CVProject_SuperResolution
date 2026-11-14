"""Read ModelArts runtime metadata and optionally run the official Ascend check."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from tools.cloud_guard import ROOT, require_cloud_workspace


def main():
    require_cloud_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/cloud_environment.json")
    parser.add_argument("--run-check", action="store_true")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path("/home/ma-user/work")):
        raise ValueError("Environment record must stay on the work volume")
    report = {"python": sys.version, "executable": sys.executable, "machine": platform.machine(),
              "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
              "ascend_home": os.environ.get("ASCEND_HOME_PATH"),
              "ascend_opp": os.environ.get("ASCEND_OPP_PATH")}
    try:
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=20)
        report["npu_smi"] = result.stdout + result.stderr
        report["npu_smi_exit_code"] = result.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        report["npu_smi_error"] = str(error)
    from drct_ms.runtime import configure_ascend
    ms = configure_ascend()
    report["mindspore"] = ms.__version__
    report["device"] = ms.get_context("device_target")
    if args.run_check:
        ms.run_check()
        report["run_check"] = "completed; inspect its stdout for the official success message"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
