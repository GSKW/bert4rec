from __future__ import annotations

import hashlib
import json
import os
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.io_utils import atomic_write_json, load_yaml
from src.tokenization import load_manifest

SECONDS_PER_DAY = 86_400
TIME_GAP_BUCKETS = [
    "gap=0",
    "gap=1s",
    "gap=2_10s",
    "gap=11_60s",
    "gap=1_5m",
    "gap=5_60m",
    "gap=1_6h",
    "gap=6_24h",
    "gap=1_7d",
    "gap=gt_7d",
]
TIME_GAP_TOKEN_TO_ID = {token: idx for idx, token in enumerate(TIME_GAP_BUCKETS)}


@dataclass(frozen=True)
class SequenceConfig:
    input_manifest_path: str
    vocab_path: str
    manifest_path: str
    work_dir: str
    sorted_events_path: str
    splits_dir: str
    processed_dir: str
    prefix_lengths: tuple[int, ...]
    retention_days: tuple[int, ...]
    split_ratios: dict[str, float]
    split_seed: int
    duckdb_memory_limit: str
    duckdb_threads: int
    duckdb_temp_directory: str
    compression: str
    row_group_size: int


def load_sequence_config(path: str | Path) -> SequenceConfig:
    raw = load_yaml(path)
    split_cfg = raw["split"]
    duckdb_cfg = raw["duckdb"]
    parquet_cfg = raw["parquet"]
    return SequenceConfig(
        input_manifest_path=raw["input_manifest_path"],
        vocab_path=raw["vocab_path"],
        manifest_path=raw["manifest_path"],
        work_dir=raw["work_dir"],
        sorted_events_path=raw["sorted_events_path"],
        splits_dir=raw["splits_dir"],
        processed_dir=raw["processed_dir"],
        prefix_lengths=tuple(int(value) for value in raw["prefix_lengths"]),
        retention_days=tuple(int(value) for value in raw["retention_days"]),
        split_ratios={
            "train": float(split_cfg["train"]),
            "valid": float(split_cfg["valid"]),
            "test": float(split_cfg["test"]),
        },
        split_seed=int(split_cfg["seed"]),
        duckdb_memory_limit=duckdb_cfg.get("memory_limit", "20GB"),
        duckdb_threads=int(duckdb_cfg.get("threads", 4)),
        duckdb_temp_directory=duckdb_cfg["temp_directory"],
        compression=parquet_cfg.get("compression", "zstd"),
        row_group_size=int(parquet_cfg.get("row_group_size", 100_000)),
    )


def load_token_to_id(path: str | Path) -> dict[str, int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(token): int(idx) for token, idx in payload["token_to_id"].items()}


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_list(paths: list[Path]) -> str:
    return "[" + ", ".join(_sql_string(str(path)) for path in paths) + "]"


def selected_normalized_paths(
    manifest: dict[str, Any],
    project_root: str | Path = ".",
    max_shards: int | None = None,
) -> list[Path]:
    root = Path(project_root)
    shards = manifest["shards"][:max_shards]
    return [root / shard["output_path"] for shard in shards]


def create_sorted_event_token_parquet(
    config: SequenceConfig,
    project_root: str | Path = ".",
    max_shards: int | None = None,
    overwrite: bool = False,
) -> Path:
    import duckdb
    import pandas as pd

    root = Path(project_root)
    sorted_path = root / config.sorted_events_path
    if max_shards is not None:
        sorted_path = sorted_path.with_name(f"{sorted_path.stem}.dryrun_{max_shards}{sorted_path.suffix}")
    if sorted_path.exists() and not overwrite:
        return sorted_path

    manifest = load_manifest(root / config.input_manifest_path)
    input_paths = selected_normalized_paths(manifest, root, max_shards=max_shards)
    sorted_path.parent.mkdir(parents=True, exist_ok=True)
    Path(root / config.duckdb_temp_directory).mkdir(parents=True, exist_ok=True)

    token_to_id = load_token_to_id(root / config.vocab_path)
    vocab_frame = pd.DataFrame(
        {"event_signature": list(token_to_id.keys()), "event_token_id": list(token_to_id.values())}
    )

    tmp_path = sorted_path.with_suffix(sorted_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    con = duckdb.connect(database=":memory:")
    con.execute(f"SET memory_limit = {_sql_string(config.duckdb_memory_limit)}")
    con.execute(f"SET threads = {config.duckdb_threads}")
    con.execute(f"SET temp_directory = {_sql_string(str(root / config.duckdb_temp_directory))}")
    con.register("event_vocab", vocab_frame)

    con.execute(
        f"""
        COPY (
            SELECT
                CAST(events.user_id AS VARCHAR) AS user_id,
                CAST(events.event_timestamp AS BIGINT) AS event_timestamp,
                CAST(events.session_id AS VARCHAR) AS session_id,
                COALESCE(vocab.event_token_id, 1) AS event_token_id
            FROM read_parquet({_sql_list(input_paths)}) AS events
            LEFT JOIN event_vocab AS vocab
                ON events.event_signature = vocab.event_signature
            ORDER BY
                user_id,
                event_timestamp,
                session_id,
                event_token_id
        )
        TO {_sql_string(str(tmp_path))}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {config.row_group_size})
        """
    )
    con.close()
    os.replace(tmp_path, sorted_path)
    return sorted_path


def split_for_user(user_id: str, config: SequenceConfig) -> str:
    digest = hashlib.blake2b(
        f"{config.split_seed}:{user_id}".encode("utf-8"),
        digest_size=8,
    ).digest()
    value = int.from_bytes(digest, "big") / 2**64
    if value < config.split_ratios["train"]:
        return "train"
    if value < config.split_ratios["train"] + config.split_ratios["valid"]:
        return "valid"
    return "test"


def time_gap_id(delta_seconds: int) -> int:
    if delta_seconds <= 0:
        return TIME_GAP_TOKEN_TO_ID["gap=0"]
    if delta_seconds == 1:
        return TIME_GAP_TOKEN_TO_ID["gap=1s"]
    if delta_seconds <= 10:
        return TIME_GAP_TOKEN_TO_ID["gap=2_10s"]
    if delta_seconds <= 60:
        return TIME_GAP_TOKEN_TO_ID["gap=11_60s"]
    if delta_seconds <= 5 * 60:
        return TIME_GAP_TOKEN_TO_ID["gap=1_5m"]
    if delta_seconds <= 60 * 60:
        return TIME_GAP_TOKEN_TO_ID["gap=5_60m"]
    if delta_seconds <= 6 * 60 * 60:
        return TIME_GAP_TOKEN_TO_ID["gap=1_6h"]
    if delta_seconds <= SECONDS_PER_DAY:
        return TIME_GAP_TOKEN_TO_ID["gap=6_24h"]
    if delta_seconds <= 7 * SECONDS_PER_DAY:
        return TIME_GAP_TOKEN_TO_ID["gap=1_7d"]
    return TIME_GAP_TOKEN_TO_ID["gap=gt_7d"]


def _has_event_in_window(timestamps: list[int], start_ts: int, end_ts: int) -> bool:
    idx = bisect_left(timestamps, start_ts)
    return idx < len(timestamps) and timestamps[idx] < end_ts


def _make_prefix_records_for_user(
    user_id: str,
    event_token_ids: list[int],
    timestamps: list[int],
    session_ids: list[str],
    config: SequenceConfig,
    dataset_max_ts: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = split_for_user(user_id, config)
    user_record = {
        "user_id": user_id,
        "split": split,
        "num_events": len(event_token_ids),
        "first_event_ts": timestamps[0],
        "last_event_ts": timestamps[-1],
    }

    records: list[dict[str, Any]] = []
    for prefix_len in config.prefix_lengths:
        if len(event_token_ids) < prefix_len:
            continue

        prefix_timestamps = timestamps[:prefix_len]
        gaps = [0]
        session_flags = [1]
        for idx in range(1, prefix_len):
            gaps.append(time_gap_id(prefix_timestamps[idx] - prefix_timestamps[idx - 1]))
            session_flags.append(1 if session_ids[idx] != session_ids[idx - 1] else 0)

        prefix_end_ts = prefix_timestamps[-1]
        record = {
            "user_id": user_id,
            "split": split,
            "prefix_len": prefix_len,
            "num_events_total": len(event_token_ids),
            "prefix_start_ts": prefix_timestamps[0],
            "prefix_end_ts": prefix_end_ts,
            "event_token_ids": event_token_ids[:prefix_len],
            "time_gap_ids": gaps,
            "session_flags": session_flags,
            "next_event_token_id": event_token_ids[prefix_len] if len(event_token_ids) > prefix_len else None,
        }

        for days in config.retention_days:
            start_ts = prefix_end_ts + days * SECONDS_PER_DAY
            end_ts = start_ts + SECONDS_PER_DAY
            available = dataset_max_ts >= end_ts
            record[f"label_available_retention_{days}d"] = available
            record[f"label_retention_{days}d"] = (
                int(_has_event_in_window(timestamps, start_ts, end_ts)) if available else None
            )
        records.append(record)

    return records, user_record


def _write_records(records: list[dict[str, Any]], path: Path, compression: str, row_group_size: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp_path, compression=compression, row_group_size=row_group_size)
    os.replace(tmp_path, path)


def build_user_sequences_and_labels(
    config: SequenceConfig,
    project_root: str | Path = ".",
    max_shards: int | None = None,
    overwrite_sorted: bool = False,
) -> dict[str, Any]:
    import pyarrow.parquet as pq
    from tqdm.auto import tqdm

    root = Path(project_root)
    sorted_path = create_sorted_event_token_parquet(
        config,
        project_root=root,
        max_shards=max_shards,
        overwrite=overwrite_sorted,
    )

    parquet_file = pq.ParquetFile(sorted_path)
    dataset_max_ts = 0
    for row_group_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_idx, columns=["event_timestamp"])
        chunk_max = table["event_timestamp"].combine_chunks().to_numpy().max()
        dataset_max_ts = max(dataset_max_ts, int(chunk_max))

    prefix_records_by_split = {"train": [], "valid": [], "test": []}
    user_records_by_split = {"train": [], "valid": [], "test": []}

    current_user: str | None = None
    current_event_ids: list[int] = []
    current_timestamps: list[int] = []
    current_session_ids: list[str] = []

    def flush_current_user() -> None:
        nonlocal current_user, current_event_ids, current_timestamps, current_session_ids
        if current_user is None or not current_event_ids:
            return
        prefix_records, user_record = _make_prefix_records_for_user(
            current_user,
            current_event_ids,
            current_timestamps,
            current_session_ids,
            config,
            dataset_max_ts,
        )
        user_records_by_split[user_record["split"]].append(user_record)
        prefix_records_by_split[user_record["split"]].extend(prefix_records)
        current_user = None
        current_event_ids = []
        current_timestamps = []
        current_session_ids = []

    for row_group_idx in tqdm(range(parquet_file.num_row_groups), desc="Build user prefixes"):
        table = parquet_file.read_row_group(
            row_group_idx,
            columns=["user_id", "event_timestamp", "session_id", "event_token_id"],
        )
        user_ids = table["user_id"].combine_chunks().to_pylist()
        timestamps = table["event_timestamp"].combine_chunks().to_pylist()
        session_ids = table["session_id"].combine_chunks().to_pylist()
        event_ids = table["event_token_id"].combine_chunks().to_pylist()

        for user_id, timestamp, session_id, event_id in zip(user_ids, timestamps, session_ids, event_ids, strict=True):
            user_id = str(user_id)
            if current_user is not None and user_id != current_user:
                flush_current_user()
            if current_user is None:
                current_user = user_id
            current_event_ids.append(int(event_id))
            current_timestamps.append(int(timestamp))
            current_session_ids.append(str(session_id))
    flush_current_user()

    splits_dir = root / config.splits_dir
    processed_dir = root / config.processed_dir
    summary_by_split: dict[str, dict[str, int]] = {}
    for split in ("train", "valid", "test"):
        users_path = splits_dir / f"users_{split}.parquet"
        prefixes_path = processed_dir / f"{split}_prefixes.parquet"
        _write_records(user_records_by_split[split], users_path, config.compression, config.row_group_size)
        _write_records(prefix_records_by_split[split], prefixes_path, config.compression, config.row_group_size)
        summary_by_split[split] = {
            "users": len(user_records_by_split[split]),
            "prefixes": len(prefix_records_by_split[split]),
        }

    time_gap_vocab_path = root / "artifacts/vocab/time_gap_vocab.json"
    atomic_write_json({"token_to_id": TIME_GAP_TOKEN_TO_ID}, time_gap_vocab_path)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_path": config.input_manifest_path,
        "vocab_path": config.vocab_path,
        "sorted_events_path": str(sorted_path.relative_to(root)),
        "max_shards": max_shards,
        "dataset_max_ts": dataset_max_ts,
        "prefix_lengths": list(config.prefix_lengths),
        "retention_days": list(config.retention_days),
        "label_definition": (
            "retention_Xd = any event in [prefix_end_ts + X days, "
            "prefix_end_ts + X days + 1 day); unavailable if dataset ends before window end"
        ),
        "time_gap_vocab_path": str(time_gap_vocab_path.relative_to(root)),
        "splits": summary_by_split,
    }
    atomic_write_json(manifest, root / config.manifest_path)
    return manifest
