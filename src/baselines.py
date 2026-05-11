from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from tqdm.auto import tqdm

from src.io_utils import atomic_write_json, load_yaml


@dataclass(frozen=True)
class BaselineConfig:
    input_prefixes: dict[str, str]
    outputs: dict[str, str]
    svd_dim: int
    hash_features: int
    top_markov_k: int
    last_k: int
    random_state: int
    dry_run_rows: int | None


def load_baseline_config(path: str | Path) -> BaselineConfig:
    raw = load_yaml(path)
    cfg = raw["baseline"]
    return BaselineConfig(
        input_prefixes=raw["input_prefixes"],
        outputs=raw["outputs"],
        svd_dim=int(cfg["svd_dim"]),
        hash_features=int(cfg["hash_features"]),
        top_markov_k=int(cfg["top_markov_k"]),
        last_k=int(cfg["last_k"]),
        random_state=int(cfg["random_state"]),
        dry_run_rows=None if cfg.get("dry_run_rows") is None else int(cfg["dry_run_rows"]),
    )


def load_prefix_frames(config: BaselineConfig, project_root: str | Path = ".") -> pd.DataFrame:
    root = Path(project_root)
    frames = []
    for split, rel_path in config.input_prefixes.items():
        frame = pd.read_parquet(root / rel_path)
        if config.dry_run_rows is not None:
            frame = frame.head(config.dry_run_rows)
        frame["split"] = split
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_markov_model(train_frame: pd.DataFrame, top_k: int) -> tuple[dict[int, list[tuple[int, int]]], pd.DataFrame, list[int]]:
    transition_counts: dict[int, Counter[int]] = defaultdict(Counter)
    popular_counter: Counter[int] = Counter()
    for events, next_id in zip(train_frame["event_token_ids"], train_frame["next_event_token_id"], strict=False):
        events = [int(value) for value in events]
        if pd.isna(next_id) or len(events) == 0:
            continue
        last_id = int(events[-1])
        target = int(next_id)
        transition_counts[last_id][target] += 1
        popular_counter[target] += 1

    markov_top = {
        last_id: counter.most_common(top_k)
        for last_id, counter in transition_counts.items()
    }
    rows = []
    for last_id, counter in transition_counts.items():
        total = sum(counter.values())
        for rank, (next_id, count) in enumerate(counter.most_common(top_k), start=1):
            rows.append({
                "last_event_token_id": last_id,
                "next_event_token_id": next_id,
                "rank": rank,
                "count": int(count),
                "prob": count / total if total else 0.0,
                "total_count": int(total),
            })
    popular_top = [token for token, _count in popular_counter.most_common(top_k)]
    return markov_top, pd.DataFrame(rows), popular_top


def _entropy(ids: list[int]) -> float:
    if not ids:
        return 0.0
    counts = Counter(ids)
    total = len(ids)
    return float(-sum((count / total) * math.log(count / total) for count in counts.values()))


def _markov_features(events: list[int], next_id: Any, markov_top: dict[int, list[tuple[int, int]]]) -> dict[str, float]:
    if not events:
        return {
            "markov_last_total_count": 0.0,
            "markov_top1_prob": 0.0,
            "markov_top10_entropy": 0.0,
            "markov_actual_prob": np.nan,
            "markov_actual_rank": np.nan,
        }
    transitions = markov_top.get(int(events[-1]), [])
    total = sum(count for _token, count in transitions)
    probs = [count / total for _token, count in transitions] if total else []
    actual_prob = np.nan
    actual_rank = np.nan
    if not pd.isna(next_id):
        for rank, (token, count) in enumerate(transitions, start=1):
            if token == int(next_id):
                actual_prob = count / total if total else 0.0
                actual_rank = rank
                break
    return {
        "markov_last_total_count": float(total),
        "markov_top1_prob": float(probs[0]) if probs else 0.0,
        "markov_top10_entropy": float(-sum(p * math.log(p) for p in probs if p > 0.0)),
        "markov_actual_prob": actual_prob,
        "markov_actual_rank": actual_rank,
    }


def build_dense_features(frame: pd.DataFrame, markov_top: dict[int, list[tuple[int, int]]], last_k: int) -> pd.DataFrame:
    rows = []
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="Dense baseline features"):
        events = [int(value) for value in row.event_token_ids]
        gaps = [int(value) for value in row.time_gap_ids]
        sessions = [int(value) for value in row.session_flags]
        unique_events = len(set(events))
        repeats = len(events) - unique_events
        feature = {
            "user_id": row.user_id,
            "split": row.split,
            "prefix_len": int(row.prefix_len),
            "num_events_total": int(row.num_events_total),
            "prefix_start_ts": int(row.prefix_start_ts),
            "prefix_end_ts": int(row.prefix_end_ts),
            "next_event_token_id": None if pd.isna(row.next_event_token_id) else int(row.next_event_token_id),
            "label_available_retention_7d": bool(row.label_available_retention_7d),
            "label_retention_7d": row.label_retention_7d,
            "label_available_retention_14d": bool(row.label_available_retention_14d),
            "label_retention_14d": row.label_retention_14d,
            "unique_event_count": unique_events,
            "event_entropy": _entropy(events),
            "repeat_count": repeats,
            "repeat_rate": repeats / len(events) if events else 0.0,
            "session_boundary_count": int(sum(sessions)),
            "gap_mean": float(np.mean(gaps)) if gaps else 0.0,
            "gap_std": float(np.std(gaps)) if gaps else 0.0,
            "gap_max": int(max(gaps)) if gaps else 0,
        }
        for idx in range(last_k):
            feature[f"last_event_{idx + 1}"] = events[-idx - 1] if len(events) > idx else -1
        feature.update(_markov_features(events, row.next_event_token_id, markov_top))
        rows.append(feature)
    return pd.DataFrame(rows)


def _hashed_feature_dict(row: Any, last_k: int) -> dict[str, float]:
    events = [int(value) for value in row.event_token_ids]
    features: dict[str, float] = {}
    counts = Counter(events)
    for event_id, count in counts.items():
        features[f"u={event_id}"] = float(count)
    for left, right in zip(events[:-1], events[1:], strict=False):
        features[f"b={left}_{right}"] = features.get(f"b={left}_{right}", 0.0) + 1.0
    for idx in range(min(last_k, len(events))):
        features[f"last{idx + 1}={events[-idx - 1]}"] = 1.0
    return features


def build_sparse_features(frame: pd.DataFrame, hash_features: int, last_k: int) -> sparse.csr_matrix:
    hasher = FeatureHasher(n_features=hash_features, input_type="dict", alternate_sign=False)
    return hasher.transform(_hashed_feature_dict(row, last_k) for row in frame.itertuples(index=False))


def build_baseline_features(config: BaselineConfig, project_root: str | Path = ".") -> dict[str, Any]:
    root = Path(project_root)
    frame = load_prefix_frames(config, root)
    train_frame = frame[frame["split"] == "train"].copy()
    markov_top, markov_stats, _popular_top = build_markov_model(train_frame, top_k=config.top_markov_k)

    dense = build_dense_features(frame, markov_top, config.last_k)
    sparse_features = build_sparse_features(frame, config.hash_features, config.last_k)
    numeric_cols = [
        "prefix_len",
        "num_events_total",
        "unique_event_count",
        "event_entropy",
        "repeat_count",
        "repeat_rate",
        "session_boundary_count",
        "gap_mean",
        "gap_std",
        "gap_max",
        "markov_last_total_count",
        "markov_top1_prob",
        "markov_top10_entropy",
    ] + [f"last_event_{idx + 1}" for idx in range(config.last_k)]
    dense_numeric = dense[numeric_cols].fillna(0.0).astype("float32").to_numpy()
    full_sparse = sparse.hstack([sparse_features, sparse.csr_matrix(dense_numeric)], format="csr")

    svd_dim = min(config.svd_dim, max(1, full_sparse.shape[1] - 1), max(1, full_sparse.shape[0] - 1))
    svd = TruncatedSVD(n_components=svd_dim, random_state=config.random_state)
    embeddings = svd.fit_transform(full_sparse).astype("float32")

    emb_frame = dense[[
        "user_id",
        "split",
        "prefix_len",
        "label_available_retention_7d",
        "label_retention_7d",
        "label_available_retention_14d",
        "label_retention_14d",
    ]].copy()
    emb_frame["embedding"] = list(embeddings)
    emb_frame["embedding_type"] = "baseline_svd"

    for rel_path in [config.outputs["features"], config.outputs["embeddings"], config.outputs["markov_stats"]]:
        (root / rel_path).parent.mkdir(parents=True, exist_ok=True)
    dense.to_parquet(root / config.outputs["features"], index=False)
    emb_frame.to_parquet(root / config.outputs["embeddings"], index=False)
    markov_stats.to_parquet(root / config.outputs["markov_stats"], index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(frame),
        "dry_run_rows": config.dry_run_rows,
        "svd_dim": int(svd_dim),
        "hash_features": config.hash_features,
        "svd_explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "outputs": config.outputs,
        "numeric_feature_columns": numeric_cols,
        "markov_rows": len(markov_stats),
    }
    atomic_write_json(manifest, root / config.outputs["manifest"])
    return manifest
