from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.checkpoints import load_torch_checkpoint
from src.datasets import PrefixCollator, SpecialTokenIds
from src.io_utils import atomic_write_json, load_yaml
from src.model_event_encoder import EventEncoderConfig, EventTransformerEncoder
from src.tokenization import load_manifest
from src.training import TrainConfig, load_train_config, resolve_device


@dataclass(frozen=True)
class ExportEmbeddingsConfig:
    checkpoint_path: str
    event_vocab_path: str
    train_config_path: str
    manifest_path: str
    input_prefixes: dict[str, str]
    outputs: dict[str, str]
    batch_size: int
    num_workers: int
    device: str
    mixed_precision: bool
    dry_run_rows: int | None


class EmbeddingPrefixDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx]
        return {
            "event_token_ids": row["event_token_ids"],
            "time_gap_ids": row["time_gap_ids"],
            "session_flags": row["session_flags"],
            "next_event_token_id": row["next_event_token_id"] if pd.notna(row["next_event_token_id"]) else 0,
            "prefix_len": row["prefix_len"],
        }


def load_export_embeddings_config(path: str | Path) -> ExportEmbeddingsConfig:
    raw = load_yaml(path)
    export = raw["export"]
    return ExportEmbeddingsConfig(
        checkpoint_path=raw["checkpoint_path"],
        event_vocab_path=raw["event_vocab_path"],
        train_config_path=raw["train_config_path"],
        manifest_path=raw["manifest_path"],
        input_prefixes=raw["input_prefixes"],
        outputs=raw["outputs"],
        batch_size=int(export["batch_size"]),
        num_workers=int(export["num_workers"]),
        device=export["device"],
        mixed_precision=bool(export["mixed_precision"]),
        dry_run_rows=None if export.get("dry_run_rows") is None else int(export["dry_run_rows"]),
    )


def _special_tokens(vocab_path: Path) -> tuple[dict[str, int], SpecialTokenIds]:
    token_to_id = load_manifest(vocab_path)["token_to_id"]
    special = SpecialTokenIds(
        pad=int(token_to_id["[PAD]"]),
        unk=int(token_to_id["[UNK]"]),
        mask=int(token_to_id["[MASK]"]),
        cls=int(token_to_id["[CLS]"]),
    )
    return token_to_id, special


def _load_model(
    checkpoint_path: Path,
    train_config: TrainConfig,
    vocab_size: int,
    special: SpecialTokenIds,
    device: torch.device,
) -> EventTransformerEncoder:
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=str(device))
    model_config = EventEncoderConfig(
        vocab_size=vocab_size,
        pad_token_id=special.pad,
        **train_config.model,
    )
    model = EventTransformerEncoder(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _metadata_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "user_id",
        "split",
        "prefix_len",
        "num_events_total",
        "prefix_start_ts",
        "prefix_end_ts",
        "next_event_token_id",
        "label_available_retention_7d",
        "label_retention_7d",
        "label_available_retention_14d",
        "label_retention_14d",
    ]
    return frame[columns].copy()


@torch.no_grad()
def export_main_embeddings(
    config: ExportEmbeddingsConfig,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root)
    device = resolve_device(config.device)
    train_config = load_train_config(root / config.train_config_path)
    token_to_id, special = _special_tokens(root / config.event_vocab_path)
    model = _load_model(root / config.checkpoint_path, train_config, len(token_to_id), special, device)

    collator = PrefixCollator(
        special,
        max_seq_len=int(train_config.model["max_seq_len"]),
        vocab_size=len(token_to_id),
        train=False,
    )

    cls_rows: list[pd.DataFrame] = []
    mean_rows: list[pd.DataFrame] = []
    readout_rows: list[pd.DataFrame] = []
    split_counts: dict[str, int] = {}

    for split, rel_path in config.input_prefixes.items():
        frame = pd.read_parquet(root / rel_path)
        if config.dry_run_rows is not None:
            frame = frame.head(config.dry_run_rows)
        split_counts[split] = len(frame)

        dataset = EmbeddingPrefixDataset(frame)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collator,
        )

        cls_embeddings: list[np.ndarray] = []
        mean_embeddings: list[np.ndarray] = []
        readout_embeddings: list[np.ndarray] = []
        for batch in tqdm(loader, desc=f"Export {split} embeddings"):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=config.mixed_precision and device.type == "cuda"):
                outputs = model(
                    input_ids=batch["input_ids"],
                    time_gap_ids=batch["time_gap_ids"],
                    session_flags=batch["session_flags"],
                    attention_mask=batch["attention_mask"],
                )
            hidden = outputs["last_hidden_state"]
            cls = outputs["cls_embedding"]
            readout = outputs["sequence_embedding"]
            token_mask = batch["attention_mask"].clone()
            token_mask[:, 0] = False
            lengths = token_mask.sum(dim=1).clamp_min(1).unsqueeze(1)
            mean = (hidden * token_mask.unsqueeze(-1)).sum(dim=1) / lengths

            cls_embeddings.append(cls.float().cpu().numpy())
            mean_embeddings.append(mean.float().cpu().numpy())
            readout_embeddings.append(readout.float().cpu().numpy())

        meta = _metadata_columns(frame)
        cls_meta = meta.copy()
        mean_meta = meta.copy()
        readout_meta = meta.copy()
        cls_meta["embedding"] = list(np.concatenate(cls_embeddings, axis=0).astype(np.float32))
        mean_meta["embedding"] = list(np.concatenate(mean_embeddings, axis=0).astype(np.float32))
        readout_meta["embedding"] = list(np.concatenate(readout_embeddings, axis=0).astype(np.float32))
        cls_meta["pooling"] = "cls"
        mean_meta["pooling"] = "mean"
        readout_meta["pooling"] = str(train_config.model.get("next_pooling", "cls"))
        cls_meta["checkpoint_path"] = config.checkpoint_path
        mean_meta["checkpoint_path"] = config.checkpoint_path
        readout_meta["checkpoint_path"] = config.checkpoint_path
        cls_rows.append(cls_meta)
        mean_rows.append(mean_meta)
        readout_rows.append(readout_meta)

    outputs_to_write = [("cls", cls_rows), ("mean", mean_rows)]
    if "readout" in config.outputs:
        outputs_to_write.append(("readout", readout_rows))
    for name, rows in outputs_to_write:
        output_path = root / config.outputs[name]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(rows, ignore_index=True).to_parquet(output_path, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": config.checkpoint_path,
        "event_vocab_path": config.event_vocab_path,
        "train_config_path": config.train_config_path,
        "device": str(device),
        "dry_run_rows": config.dry_run_rows,
        "split_counts": split_counts,
        "outputs": config.outputs,
    }
    atomic_write_json(manifest, root / config.manifest_path)
    return manifest
