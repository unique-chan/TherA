"""
Default paths for TherA inference (relative to this repository root).

Download model weights into the `weights/` directory — see README.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

DEFAULT_CHECKPOINT = WEIGHTS_DIR / "checkpoint"
DEFAULT_MERGED_MODEL = WEIGHTS_DIR / "merged_models"
DEFAULT_PRETRAINED_SD = WEIGHTS_DIR / "stable-diffusion"
DEFAULT_REFERENCE_CACHES = WEIGHTS_DIR / "reference_caches"


def setup_project_path() -> Path:
    """Ensure TherA root is on sys.path so local packages import correctly."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT
