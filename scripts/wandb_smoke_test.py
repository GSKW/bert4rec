from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoints import atomic_torch_save, load_torch_checkpoint
from src.wandb_utils import WandbRunConfig, config_hash, init_wandb_run, save_run_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W&B smoke test with checkpoint resume.")
    parser.add_argument("--project", default="bert4rec-events-embedder")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--mode", choices=["offline", "online", "disabled"], default="offline")
    parser.add_argument("--run-name", default="wandb-smoke-test")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    import torch

    args = parse_args()
    set_seed(args.seed)

    smoke_config = {
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
        "task": "wandb_smoke_test",
    }
    run_config = WandbRunConfig(
        project=args.project,
        entity=args.entity,
        mode=args.mode,
        run_name=args.run_name,
        root_dir=str(PROJECT_ROOT),
    )

    run, previous_state = init_wandb_run(run_config, config=smoke_config, resume=args.resume)

    checkpoint_path = PROJECT_ROOT / "artifacts/checkpoints/smoke_checkpoint.pt"
    target = torch.tensor([0.0])
    weight = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([weight], lr=args.lr)

    global_step = 0
    epoch = 0
    resumed = False

    if args.resume and previous_state and checkpoint_path.exists():
        checkpoint = load_torch_checkpoint(checkpoint_path)
        weight.data.copy_(checkpoint["model_state_dict"]["weight"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = int(checkpoint["global_step"])
        epoch = int(checkpoint["epoch"])
        resumed = True

    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = (weight - target).pow(2).mean()
        loss.backward()
        optimizer.step()

        global_step += 1
        epoch = global_step // max(args.steps, 1)
        run.log(
            {
                "train/loss": float(loss.detach().cpu()),
                "global_step": global_step,
                "epoch": epoch,
                "resume_flag": int(resumed),
            },
            step=global_step,
        )

    checkpoint_payload = {
        "model_state_dict": {"weight": weight.detach().cpu().clone()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "scaler_state_dict": None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": -float(loss.detach().cpu()),
        "config": smoke_config,
        "rng_state_python": random.getstate(),
        "rng_state_numpy": np.random.get_state(),
        "rng_state_torch_cpu": torch.get_rng_state(),
        "rng_state_torch_cuda": None,
    }
    atomic_torch_save(checkpoint_payload, checkpoint_path)

    state_path = PROJECT_ROOT / run_config.state_path
    save_run_state(
        state_path,
        {
            "run_id": run.id,
            "epoch": epoch,
            "global_step": global_step,
            "last_checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "best_checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "config_hash": config_hash(smoke_config),
            "wandb_mode": args.mode,
            "resumed": resumed,
        },
    )

    run.finish()

    print(f"W&B smoke test completed. run_id={run.id} global_step={global_step} resumed={resumed}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"State: {state_path}")


if __name__ == "__main__":
    main()
