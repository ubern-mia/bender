"""
shared/model.py: model-level utilities shared across all versions.

The CNN class itself is defined in each versioned script, since the
architecture changes between versions and is meant to be read directly.
"""

import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_output_path(version: str) -> str:
    """Return the versioned output path under results/."""
    return os.path.join("results", version)
