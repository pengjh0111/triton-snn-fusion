"""Deprecated CLI wrapper for test_temporal_lif_avgpool_linear_kernel.py."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dev_tests.test_temporal_lif_avgpool_linear_kernel import main


if __name__ == "__main__":
    main()
