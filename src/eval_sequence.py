from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.baselines import build_markov_model
from src.checkpoints import load_torch_checkpoint
from src.datasets import PrefixCollator, PrefixDataset, SpecialTokenIds
from src.io_utils import atomic_write_json, load_yaml
from src.model_event_encoder import EventEncoderConfig, EventTransformerEncoder
from src.tokenization import load_manifest
from src.training import load_train_config, resolve_device


@dataclass(frozen=True)
class SequenceEvalConfig:
    checkpoint_path: str
    event_vocab_path: str
    train_config_path: str
    output_path: str
    manifest_path: str
    input_prefixes: dict[str, str]
    k: int
    batch_size: int
    num_workers: int
    device: str
    mixed_precision: bool
    dry_run_rows: int | None


def load_sequence_eval_config(path: str | Path) -> SequenceEvalConfig:
    raw = load_yaml(path)
    cfg = raw["eval"]
    return SequenceEvalConfig(
        checkpoint_path=raw["checkpoint_path"],
        event_vocab_path=raw["event_vocab_path"],
        train_config_path=raw["train_config_path"],
        output_path=raw["output_path"],
        manifest_path=raw["manifest_path"],
        input_prefixes=raw["input_prefixes"],
        k=int(cfg["k"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        device=cfg["device"],
        mixed_precision=bool(cfg["mixed_precision"]),
        dry_run_rows=None if cfg.get("dry_run_rows") is None else int(cfg["dry_run_rows"]),
    )


def _load_frames(config: SequenceEvalConfig, root: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for split, rel_path in config.input_prefixes.items():
        frame = pd.read_parquet(root / rel_path)
        if config.dry_run_rows is not None:
            frame = frame.head(config.dry_run_rows)
        frame = frame[frame["next_event_token_id"].notna()].reset_index(drop=True)
        frames[split] = frame
    return frames


def _rank_metrics_from_predictions(frame: pd.DataFrame, predictions: list[list[int]], k: int, model_name: str) -> list[dict[str, Any]]:
    rows = []
    pred_series = pd.Series(predictions)
    tmp = frame[["split", "prefix_len", "next_event_token_id"]].copy()
    tmp["predictions"] = pred_series
    for (split, prefix_len), group in tmp.groupby(["split", "prefix_len"]):
        hits = []
        rr = []
        for target, pred in zip(group["next_event_token_id"], group["predictions"], strict=False):
            pred_k = list(pred)[:k]
            target = int(target)
            if target in pred_k:
                hits.append(1.0)
                rr.append(1.0 / (pred_k.index(target) + 1))
            else:
                hits.append(0.0)
                rr.append(0.0)
        rows.append({
            "model_name": model_name,
            "split": split,
            "prefix_len": int(prefix_len),
            "mrr_at_10": sum(rr) / len(rr) if rr else 0.0,
            "hit_at_10": sum(hits) / len(hits) if hits else 0.0,
            "num_rows": len(group),
        })
    return rows


def _most_popular_predictions(train_frame: pd.DataFrame, frames: dict[str, pd.DataFrame], k: int) -> list[dict[str, Any]]:
    counter = Counter(int(value) for value in train_frame["next_event_token_id"].dropna())
    top = [token for token, _count in counter.most_common(k)]
    rows = []
    for frame in frames.values():
        rows.extend(_rank_metrics_from_predictions(frame, [top] * len(frame), k, "MostPopular"))
    return rows


def _markov_predictions(train_frame: pd.DataFrame, frames: dict[str, pd.DataFrame], k: int) -> list[dict[str, Any]]:
    markov_top, _stats, popular_top = build_markov_model(train_frame, top_k=k)
    rows = []
    for frame in frames.values():
        predictions = []
        for events in frame["event_token_ids"]:
            events = [int(value) for value in events]
            if not events:
                predictions.append(popular_top)
                continue
            pred = [token for token, _count in markov_top.get(events[-1], [])]
            predictions.append((pred + popular_top)[:k])
        rows.extend(_rank_metrics_from_predictions(frame, predictions, k, "Markov1"))
    return rows


def _load_model(root: Path, config: SequenceEvalConfig, device: torch.device) -> tuple[EventTransformerEncoder, SpecialTokenIds, int]:
    train_config = load_train_config(root / config.train_config_path)
    token_to_id = load_manifest(root / config.event_vocab_path)["token_to_id"]
    special = SpecialTokenIds(int(token_to_id["[PAD]"]), int(token_to_id["[UNK]"]), int(token_to_id["[MASK]"]), int(token_to_id["[CLS]"]))
    checkpoint = load_torch_checkpoint(root / config.checkpoint_path, map_location=str(device))
    model_config = EventEncoderConfig(vocab_size=len(token_to_id), pad_token_id=special.pad, **train_config.model)
    model = EventTransformerEncoder(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, special, len(token_to_id)


def _update_prefix_topk_stats(
    stats: defaultdict[int, dict[str, float]],
    prefix_lens: list[int],
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> None:
    valid_mask = labels >= 0
    topk = torch.topk(logits, k=min(k, logits.size(-1)), dim=-1).indices
    matches = topk.eq(labels.unsqueeze(-1)) & valid_mask.unsqueeze(-1)
    hits = matches.any(dim=-1).float()

    ranks = torch.arange(1, topk.size(1) + 1, device=topk.device, dtype=torch.float32)
    reciprocal_ranks = torch.where(matches, 1.0 / ranks, torch.zeros_like(matches, dtype=torch.float32))
    mrr = reciprocal_ranks.max(dim=-1).values

    for prefix_len, is_valid, hit, reciprocal_rank in zip(
        prefix_lens,
        valid_mask.detach().cpu().tolist(),
        hits.detach().cpu().tolist(),
        mrr.detach().cpu().tolist(),
        strict=False,
    ):
        if not is_valid:
            continue
        prefix_stats = stats[int(prefix_len)]
        prefix_stats["hit"] += float(hit)
        prefix_stats["mrr"] += float(reciprocal_rank)
        prefix_stats["count"] += 1.0


@torch.no_grad()
def _transformer_metrics(config: SequenceEvalConfig, frames: dict[str, pd.DataFrame], root: Path) -> list[dict[str, Any]]:
    device = resolve_device(config.device)
    model, special, vocab_size = _load_model(root, config, device)
    train_config = load_train_config(root / config.train_config_path)
    rows = []
    for split, frame in frames.items():
        tmp_path = root / f"artifacts/work/sequence_eval_{split}.parquet"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(tmp_path, index=False)
        dataset = PrefixDataset(tmp_path)
        collator = PrefixCollator(special, int(train_config.model["max_seq_len"]), vocab_size=vocab_size, train=False)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=collator, pin_memory=device.type == "cuda")
        stats_by_prefix: defaultdict[int, dict[str, float]] = defaultdict(lambda: {"hit": 0.0, "mrr": 0.0, "count": 0.0})
        prefix_lens = frame["prefix_len"].astype(int).tolist()
        offset = 0
        for batch in tqdm(loader, desc=f"Transformer sequence eval {split}"):
            batch_size = batch["input_ids"].size(0)
            batch_prefix = prefix_lens[offset: offset + batch_size]
            offset += batch_size
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=config.mixed_precision and device.type == "cuda"):
                outputs = model(batch["input_ids"], batch["time_gap_ids"], batch["session_flags"], batch["attention_mask"])
            _update_prefix_topk_stats(stats_by_prefix, batch_prefix, outputs["next_logits"], batch["next_labels"], config.k)
        for prefix_len, stats in stats_by_prefix.items():
            count = int(stats["count"])
            rows.append({
                "model_name": "Transformer",
                "split": split,
                "prefix_len": int(prefix_len),
                "mrr_at_10": stats["mrr"] / count if count else 0.0,
                "hit_at_10": stats["hit"] / count if count else 0.0,
                "num_rows": count,
            })
    return rows


def evaluate_sequence_quality(config: SequenceEvalConfig, project_root: str | Path = ".") -> dict[str, Any]:
    root = Path(project_root)
    frames = _load_frames(config, root)
    rows = []
    rows.extend(_most_popular_predictions(frames["train"], frames, config.k))
    rows.extend(_markov_predictions(frames["train"], frames, config.k))
    rows.extend(_transformer_metrics(config, frames, root))
    output = pd.DataFrame(rows).sort_values(["model_name", "split", "prefix_len"])
    (root / config.output_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(root / config.output_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": config.checkpoint_path,
        "dry_run_rows": config.dry_run_rows,
        "output_path": config.output_path,
        "rows": len(output),
    }
    atomic_write_json(manifest, root / config.manifest_path)
    return manifest
