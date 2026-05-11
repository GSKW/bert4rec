from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WandbRunConfig:
    project: str = "bert4rec-events-embedder"
    entity: str | None = None
    mode: str = "offline"
    run_name: str = "wandb-smoke-test"
    group: str = "smoke"
    tags: tuple[str, ...] = ("smoke", "resume", "checkpoint")
    root_dir: str = "."
    wandb_dir: str = "artifacts/wandb_local"
    state_path: str = "artifacts/manifests/smoke_state.json"


def config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def load_run_state(path: str | Path) -> dict[str, Any] | None:
    state_path = Path(path)
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_run_state(path: str | Path, state: dict[str, Any]) -> Path:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_path, state_path)
    return state_path


def init_wandb_run(
    run_config: WandbRunConfig,
    config: dict[str, Any],
    resume: bool = True,
) -> tuple[Any, dict[str, Any] | None]:
    """Initialize W&B and resume by run_id from state file when present."""
    root_dir = Path(run_config.root_dir)
    wandb_dir = root_dir / run_config.wandb_dir
    cache_dir = wandb_dir / "cache"
    config_dir = wandb_dir / "config"
    data_dir = wandb_dir / "data"
    state_path = root_dir / run_config.state_path
    for path in [wandb_dir, cache_dir, config_dir, data_dir]:
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WANDB_MODE", run_config.mode)
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))
    os.environ.setdefault("WANDB_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("WANDB_DATA_DIR", str(data_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Package 'wandb' is not installed. Install project requirements first: "
            "python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt"
        ) from exc

    previous_state = load_run_state(state_path) if resume else None
    run_id = previous_state.get("run_id") if previous_state else None

    run = wandb.init(
        project=run_config.project,
        entity=run_config.entity,
        name=run_config.run_name,
        group=run_config.group,
        tags=list(run_config.tags),
        dir=str(wandb_dir),
        mode=run_config.mode,
        id=run_id,
        resume="allow" if run_id else None,
        config={
            **config,
            "wandb_run": asdict(run_config),
            "config_hash": config_hash(config),
        },
    )
    return run, previous_state
