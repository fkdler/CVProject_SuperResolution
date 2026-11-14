"""Model components; user policy permits execution only on ModelArts."""

from drct_ms.runtime import require_cloud_workspace

require_cloud_workspace()
