from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> Path:
    """Save a torch checkpoint via temp file + atomic rename."""
    import torch

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")

    torch.save(payload, tmp_path)
    os.replace(tmp_path, target_path)
    return target_path


def load_torch_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    import torch

    return torch.load(Path(path), map_location=map_location, weights_only=False)
