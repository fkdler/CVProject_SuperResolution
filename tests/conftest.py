"""Prevent accidental local test collection/imports of inference frameworks."""

from tools.cloud_guard import require_cloud_workspace
import pytest

require_cloud_workspace()


def pytest_addoption(parser):
    parser.addoption("--ms-mode", choices=["pynative", "graph"], default="pynative")


@pytest.fixture(scope="session", autouse=True)
def ascend_runtime(request):
    # Reference-only tests run in a separate cloud environment without MindSpore.
    if request.config.getoption("markexpr") == "reference":
        return
    from drct_ms.runtime import configure_ascend
    configure_ascend(request.config.getoption("--ms-mode"))
