from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.checkpoints import atomic_torch_save, load_torch_checkpoint
from src.datasets import PrefixCollator, PrefixDataset, SpecialTokenIds
from src.io_utils import load_yaml
from src.losses import multitask_sequence_loss
from src.metrics import MeanMetric, next_event_topk_metrics
from src.model_event_encoder import EventEncoderConfig, EventTransformerEncoder
from src.tokenization import load_manifest
from src.wandb_utils import WandbRunConfig, config_hash, init_wandb_run, save_run_state


@dataclass(frozen=True)
class TrainConfig:
    paths: dict[str, Any]
    wandb: dict[str, Any]
    model: dict[str, Any]
    train: dict[str, Any]


def load_train_config(path: str | Path) -> TrainConfig:
    raw = load_yaml(path)
    return TrainConfig(
        paths=raw["paths"],
        wandb=raw["wandb"],
        model=raw["model"],
        train=raw["train"],
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _none_to_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _grad_norm(parameters: Any) -> float:
    total_norm = torch.norm(
        torch.stack([
            parameter.grad.detach().norm(2)
            for parameter in parameters
            if parameter.grad is not None
        ]),
        2,
    )
    return float(total_norm.cpu())


def _contrastive_kind(rates: dict[str, float]) -> str | None:
    options = [(kind, float(rate)) for kind, rate in rates.items() if float(rate) > 0.0]
    if not options:
        return None
    kinds = [kind for kind, _rate in options]
    probs = torch.tensor([rate for _kind, rate in options], dtype=torch.float32)
    probs = probs / probs.sum()
    idx = int(torch.multinomial(probs, num_samples=1).item())
    return kinds[idx]


def _augment_single_sequence(
    input_ids: torch.Tensor,
    time_gap_ids: torch.Tensor,
    session_flags: torch.Tensor,
    attention_mask: torch.Tensor,
    special: SpecialTokenIds,
    rates: dict[str, float],
) -> None:
    valid_len = int(attention_mask.sum().item())
    event_start = 1
    event_len = valid_len - event_start
    if event_len <= 0:
        return

    op = _contrastive_kind(rates)
    if op is None:
        return

    if op == "mask":
        rate = max(0.0, min(1.0, float(rates["mask"])))
        n_mask = max(1, int(round(event_len * rate)))
        n_mask = min(n_mask, event_len)
        perm = torch.randperm(event_len, device=input_ids.device)[:n_mask] + event_start
        input_ids[perm] = int(special.mask)
        return

    if op == "crop":
        rate = max(0.0, min(0.95, float(rates["crop"])))
        keep_len = max(1, int(round(event_len * (1.0 - rate))))
        keep_len = min(keep_len, event_len)
        if keep_len >= event_len:
            return
        start = int(torch.randint(0, event_len - keep_len + 1, (1,), device=input_ids.device).item())
        src = torch.arange(event_start + start, event_start + start + keep_len, device=input_ids.device)
        old_input = input_ids.clone()
        old_gap = time_gap_ids.clone()
        old_sess = session_flags.clone()
        input_ids.fill_(int(special.pad))
        time_gap_ids.zero_()
        session_flags.zero_()
        attention_mask.fill_(False)
        input_ids[0] = old_input[0]
        time_gap_ids[0] = old_gap[0]
        session_flags[0] = old_sess[0]
        attention_mask[0] = True
        dst_end = 1 + keep_len
        input_ids[1:dst_end] = old_input[src]
        time_gap_ids[1:dst_end] = old_gap[src]
        session_flags[1:dst_end] = old_sess[src]
        attention_mask[1:dst_end] = True
        return

    if op == "reorder":
        if event_len < 2:
            return
        rate = max(0.0, min(1.0, float(rates["reorder"])))
        span = max(2, int(round(event_len * rate)))
        span = min(span, event_len)
        start = int(torch.randint(0, event_len - span + 1, (1,), device=input_ids.device).item())
        sl = slice(event_start + start, event_start + start + span)
        perm = torch.randperm(span, device=input_ids.device)
        input_ids[sl] = input_ids[sl][perm]
        time_gap_ids[sl] = time_gap_ids[sl][perm]
        session_flags[sl] = session_flags[sl][perm]


def _build_augmented_batch(
    batch: dict[str, torch.Tensor],
    special: SpecialTokenIds,
    contrastive_cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    view = {
        "input_ids": batch["input_ids"].clone(),
        "time_gap_ids": batch["time_gap_ids"].clone(),
        "session_flags": batch["session_flags"].clone(),
        "attention_mask": batch["attention_mask"].clone(),
    }
    rates = {
        "mask": float(contrastive_cfg.get("mask_rate", 0.2)),
        "crop": float(contrastive_cfg.get("crop_rate", 0.2)),
        "reorder": float(contrastive_cfg.get("reorder_rate", 0.2)),
    }
    for row_idx in range(view["input_ids"].size(0)):
        _augment_single_sequence(
            view["input_ids"][row_idx],
            view["time_gap_ids"][row_idx],
            view["session_flags"][row_idx],
            view["attention_mask"][row_idx],
            special,
            rates,
        )
    return view


def _contrastive_info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    if z1.size(0) <= 1:
        return z1.new_zeros(())
    z1 = F.normalize(z1, p=2, dim=-1)
    z2 = F.normalize(z2, p=2, dim=-1)
    n = z1.size(0)
    reps = torch.cat([z1, z2], dim=0)
    logits = (reps @ reps.transpose(0, 1)).float()
    logits = logits / max(float(temperature), 1e-6)
    eye = torch.eye(2 * n, device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(eye, -1e9)
    targets = torch.cat([
        torch.arange(n, 2 * n, device=logits.device),
        torch.arange(0, n, device=logits.device),
    ])
    return F.cross_entropy(logits, targets)


def build_dataloaders(
    config: TrainConfig,
    project_root: str | Path,
    special: SpecialTokenIds,
    vocab_size: int,
) -> tuple[DataLoader, DataLoader, int, int]:
    root = Path(project_root)
    train_cfg = config.train

    train_dataset = PrefixDataset(
        root / config.paths["train_prefixes"],
        max_rows=_none_to_int(train_cfg.get("dry_run_train_rows")),
    )
    valid_dataset = PrefixDataset(
        root / config.paths["valid_prefixes"],
        max_rows=_none_to_int(train_cfg.get("dry_run_valid_rows")),
    )

    train_collator = PrefixCollator(
        special,
        max_seq_len=int(config.model["max_seq_len"]),
        mlm_probability=float(train_cfg["mlm_probability"]),
        mask_token_probability=float(train_cfg["mask_token_probability"]),
        random_token_probability=float(train_cfg["random_token_probability"]),
        vocab_size=vocab_size,
        train=True,
    )
    valid_collator = PrefixCollator(
        special,
        max_seq_len=int(config.model["max_seq_len"]),
        mlm_probability=float(train_cfg["mlm_probability"]),
        mask_token_probability=float(train_cfg["mask_token_probability"]),
        random_token_probability=float(train_cfg["random_token_probability"]),
        vocab_size=vocab_size,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=train_collator,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(train_cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=valid_collator,
        drop_last=False,
    )
    return train_loader, valid_loader, len(train_dataset), len(valid_dataset)


@torch.no_grad()
def evaluate(
    model: EventTransformerEncoder,
    valid_loader: DataLoader,
    device: torch.device,
    loss_weights: dict[str, float],
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    loss_total = MeanMetric()
    loss_mlm = MeanMetric()
    loss_next = MeanMetric()
    hit_at_10 = MeanMetric()
    mrr_at_10 = MeanMetric()

    for batch_idx, batch in enumerate(valid_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = _move_batch_to_device(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            time_gap_ids=batch["time_gap_ids"],
            session_flags=batch["session_flags"],
            attention_mask=batch["attention_mask"],
        )
        mlm_mask = batch["mlm_labels"] != -100
        mlm_logits = model.mlm_head(outputs["last_hidden_state"][mlm_mask])
        mlm_labels = batch["mlm_labels"][mlm_mask]
        losses = multitask_sequence_loss(
            mlm_logits,
            mlm_labels,
            outputs["next_logits"],
            batch["next_labels"],
            mlm_weight=float(loss_weights["mlm"]),
            next_weight=float(loss_weights["next"]),
        )
        metrics = next_event_topk_metrics(outputs["next_logits"], batch["next_labels"], k=10)
        weight = batch["input_ids"].size(0)
        loss_total.update(float(losses["loss_total"].detach().cpu()), weight)
        loss_mlm.update(float(losses["loss_mlm"].detach().cpu()), weight)
        loss_next.update(float(losses["loss_next"].detach().cpu()), weight)
        hit_at_10.update(metrics["hit_at_10"], weight)
        mrr_at_10.update(metrics["mrr_at_10"], weight)

    model.train()
    return {
        "valid/loss_total": loss_total.compute(),
        "valid/loss_mlm": loss_mlm.compute(),
        "valid/loss_next": loss_next.compute(),
        "valid/hit_at_10": hit_at_10.compute(),
        "valid/mrr_at_10": mrr_at_10.compute(),
    }


def _checkpoint_payload(
    model: EventTransformerEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_metric: float,
    config: TrainConfig,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "config": asdict(config),
        "rng_state_python": random.getstate(),
        "rng_state_numpy": np.random.get_state(),
        "rng_state_torch_cpu": torch.get_rng_state(),
        "rng_state_torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def train_main_embedder(
    config: TrainConfig,
    project_root: str | Path = ".",
    resume: bool = True,
    wandb_mode_override: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    train_cfg = config.train
    set_seed(int(train_cfg["seed"]))

    vocab_payload = load_manifest(root / config.paths["event_vocab"])
    token_to_id = vocab_payload["token_to_id"]
    special = SpecialTokenIds(
        pad=int(token_to_id["[PAD]"]),
        unk=int(token_to_id["[UNK]"]),
        mask=int(token_to_id["[MASK]"]),
        cls=int(token_to_id["[CLS]"]),
    )
    vocab_size = len(token_to_id)

    train_loader, valid_loader, train_rows, valid_rows = build_dataloaders(config, root, special, vocab_size)
    steps_per_epoch = math.ceil(len(train_loader) / int(train_cfg["gradient_accumulation_steps"]))
    total_steps = int(train_cfg["epochs"]) * steps_per_epoch
    max_steps = _none_to_int(train_cfg.get("max_steps"))
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    warmup_steps = int(total_steps * float(train_cfg["warmup_ratio"]))

    device = resolve_device(str(train_cfg["device"]))
    model_config = EventEncoderConfig(
        vocab_size=vocab_size,
        pad_token_id=special.pad,
        **config.model,
    )
    model = EventTransformerEncoder(model_config).to(device)
    if bool(train_cfg.get("compile_model", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = _make_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=float(train_cfg["min_lr_ratio"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg["mixed_precision"]) and device.type == "cuda")

    checkpoint_dir = root / config.paths["checkpoint_dir"]
    checkpoint_last = checkpoint_dir / "checkpoint_last.pt"
    checkpoint_best = checkpoint_dir / "checkpoint_best.pt"
    state_path = root / config.paths["run_state_path"]

    start_epoch = 0
    global_step = 0
    best_metric = -float("inf")
    if resume and checkpoint_last.exists():
        checkpoint = load_torch_checkpoint(checkpoint_last, map_location=str(device))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_metric = float(checkpoint["best_metric"])

    wandb_cfg = config.wandb
    run_config = WandbRunConfig(
        project=wandb_cfg["project"],
        entity=wandb_cfg.get("entity"),
        mode=wandb_mode_override or wandb_cfg["mode"],
        run_name=wandb_cfg["run_name"],
        group=wandb_cfg["group"],
        tags=tuple(wandb_cfg["tags"]),
        root_dir=str(root),
        state_path=config.paths["run_state_path"],
    )
    run, _previous_state = init_wandb_run(run_config, config={**asdict(config), "vocab_size": vocab_size}, resume=resume)
    run.summary["train_rows"] = train_rows
    run.summary["valid_rows"] = valid_rows
    run.summary["device"] = str(device)

    model.train()
    grad_accum = int(train_cfg["gradient_accumulation_steps"])
    loss_weights = train_cfg["loss_weights"]
    contrastive_cfg = dict(train_cfg.get("contrastive", {}))
    contrastive_enabled = bool(contrastive_cfg.get("enabled", False))
    stop_training = False

    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader):
            batch = _move_batch_to_device(batch, device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=bool(train_cfg["mixed_precision"]) and device.type == "cuda",
            ):
                outputs = model(
                    input_ids=batch["input_ids"],
                    time_gap_ids=batch["time_gap_ids"],
                    session_flags=batch["session_flags"],
                    attention_mask=batch["attention_mask"],
                )
                mlm_mask = batch["mlm_labels"] != -100
                mlm_logits = model.mlm_head(outputs["last_hidden_state"][mlm_mask])
                mlm_labels = batch["mlm_labels"][mlm_mask]
                losses = multitask_sequence_loss(
                    mlm_logits,
                    mlm_labels,
                    outputs["next_logits"],
                    batch["next_labels"],
                    mlm_weight=float(loss_weights["mlm"]),
                    next_weight=float(loss_weights["next"]),
                )
                contrastive_loss = outputs["next_logits"].new_zeros(())
                if contrastive_enabled:
                    aug_view_1 = _build_augmented_batch(batch, special, contrastive_cfg)
                    aug_view_2 = _build_augmented_batch(batch, special, contrastive_cfg)
                    out_1 = model(
                        input_ids=aug_view_1["input_ids"],
                        time_gap_ids=aug_view_1["time_gap_ids"],
                        session_flags=aug_view_1["session_flags"],
                        attention_mask=aug_view_1["attention_mask"],
                    )
                    out_2 = model(
                        input_ids=aug_view_2["input_ids"],
                        time_gap_ids=aug_view_2["time_gap_ids"],
                        session_flags=aug_view_2["session_flags"],
                        attention_mask=aug_view_2["attention_mask"],
                    )
                    contrastive_loss = _contrastive_info_nce_loss(
                        out_1["sequence_embedding"],
                        out_2["sequence_embedding"],
                        temperature=float(contrastive_cfg.get("temperature", 0.2)),
                    )
                total_loss = losses["loss_total"] + float(contrastive_cfg.get("weight", 0.0)) * contrastive_loss
                loss = total_loss / grad_accum

            scaler.scale(loss).backward()
            should_step = (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader)
            if not should_step:
                continue

            scaler.unscale_(optimizer)
            grad_norm = _grad_norm(model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            lr = scheduler.get_last_lr()[0]
            tokens_seen = int(batch["attention_mask"].sum().detach().cpu())

            if global_step % int(train_cfg["log_every_steps"]) == 0 or global_step == 1:
                run.log(
                    {
                        "train/loss_total": float(losses["loss_total"].detach().cpu()),
                        "train/loss_mlm": float(losses["loss_mlm"].detach().cpu()),
                        "train/loss_next": float(losses["loss_next"].detach().cpu()),
                        "train/loss_contrastive": float(contrastive_loss.detach().cpu()) if contrastive_enabled else 0.0,
                        "train/loss_total_with_contrastive": float(total_loss.detach().cpu()),
                        "train/lr": lr,
                        "train/grad_norm": grad_norm,
                        "train/tokens_seen": tokens_seen,
                        "train/global_step": global_step,
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            if global_step % int(train_cfg["validate_every_steps"]) == 0 or global_step == 1:
                valid_metrics = evaluate(model, valid_loader, device, loss_weights)
                run.log(valid_metrics, step=global_step)
                score = valid_metrics["valid/mrr_at_10"]
                if score > best_metric:
                    best_metric = score
                    atomic_torch_save(
                        _checkpoint_payload(model, optimizer, scheduler, scaler, epoch, global_step, best_metric, config),
                        checkpoint_best,
                    )
                    run.summary["best_valid_mrr_at_10"] = best_metric
                    run.summary["best_step"] = global_step

            if global_step % int(train_cfg["checkpoint_every_steps"]) == 0:
                atomic_torch_save(
                    _checkpoint_payload(model, optimizer, scheduler, scaler, epoch, global_step, best_metric, config),
                    checkpoint_last,
                )
                save_run_state(
                    state_path,
                    {
                        "run_id": run.id,
                        "epoch": epoch,
                        "global_step": global_step,
                        "last_checkpoint": str(checkpoint_last.relative_to(root)),
                        "best_checkpoint": str(checkpoint_best.relative_to(root)),
                        "best_metric": best_metric,
                        "config_hash": config_hash(asdict(config)),
                    },
                )

            if max_steps is not None and global_step >= max_steps:
                stop_training = True
                break

        atomic_torch_save(
            _checkpoint_payload(model, optimizer, scheduler, scaler, epoch + 1, global_step, best_metric, config),
            checkpoint_last,
        )
        save_run_state(
            state_path,
            {
                "run_id": run.id,
                "epoch": epoch + 1,
                "global_step": global_step,
                "last_checkpoint": str(checkpoint_last.relative_to(root)),
                "best_checkpoint": str(checkpoint_best.relative_to(root)),
                "best_metric": best_metric,
                "config_hash": config_hash(asdict(config)),
            },
        )
        if stop_training:
            break

    final_metrics = evaluate(model, valid_loader, device, loss_weights)
    run.log(final_metrics, step=global_step)
    run.finish()

    return {
        "global_step": global_step,
        "best_metric": best_metric,
        "checkpoint_last": str(checkpoint_last),
        "checkpoint_best": str(checkpoint_best),
        "final_metrics": final_metrics,
        "train_rows": train_rows,
        "valid_rows": valid_rows,
    }
