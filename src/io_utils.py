from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class IngestConfig:
    raw_csv_path: str = "data/extracted/events.csv"
    interim_dir: str = "data/interim"
    manifest_path: str = "artifacts/manifests/ingest_manifest.json"
    encoding: str = "utf-8-sig"
    chunksize: int = 1_000_000
    compression: str = "zstd"
    row_group_size: int = 100_000
    user_col: str = "appmetrica_device_id"
    event_name_col: str = "event_name"
    event_json_col: str = "event_json"
    event_datetime_col: str = "event_datetime"
    event_timestamp_col: str = "event_timestamp"
    session_id_col: str = "session_id"


def atomic_write_json(payload: dict[str, Any], path: str | Path) -> Path:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_path, target_path)
    return target_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_ingest_config(path: str | Path) -> IngestConfig:
    raw = load_yaml(path)
    csv_cfg = raw.get("csv", {})
    parquet_cfg = raw.get("parquet", {})
    columns = raw.get("columns", {})

    return IngestConfig(
        raw_csv_path=raw["raw_csv_path"],
        interim_dir=raw["interim_dir"],
        manifest_path=raw["manifest_path"],
        encoding=csv_cfg.get("encoding", "utf-8-sig"),
        chunksize=int(csv_cfg.get("chunksize", 1_000_000)),
        compression=parquet_cfg.get("compression", "zstd"),
        row_group_size=int(parquet_cfg.get("row_group_size", 100_000)),
        user_col=columns.get("user_id", "appmetrica_device_id"),
        event_name_col=columns.get("event_name", "event_name"),
        event_json_col=columns.get("event_json", "event_json"),
        event_datetime_col=columns.get("event_datetime", "event_datetime"),
        event_timestamp_col=columns.get("event_timestamp", "event_timestamp"),
        session_id_col=columns.get("session_id", "session_id"),
    )


def _normalize_header(columns: Iterable[str]) -> list[str]:
    return [column.lstrip("\ufeff") for column in columns]


def _prepare_events_chunk(chunk: Any, config: IngestConfig) -> tuple[Any, dict[str, int]]:
    import pandas as pd

    chunk = chunk.copy()
    chunk.columns = _normalize_header(chunk.columns)

    required_columns = [
        config.user_col,
        config.event_name_col,
        config.event_json_col,
        config.event_datetime_col,
        config.event_timestamp_col,
        config.session_id_col,
    ]
    missing_columns = [column for column in required_columns if column not in chunk.columns]
    if missing_columns:
        raise ValueError(f"Missing required CSV columns: {missing_columns}")

    chunk = chunk[required_columns]
    before_rows = len(chunk)
    chunk = chunk.dropna(subset=[config.user_col, config.event_name_col, config.event_timestamp_col])
    dropped_rows = before_rows - len(chunk)

    chunk[config.user_col] = chunk[config.user_col].astype("string")
    chunk[config.event_name_col] = chunk[config.event_name_col].astype("string").str.strip()
    chunk[config.event_json_col] = chunk[config.event_json_col].fillna("{}").astype("string")
    chunk[config.session_id_col] = chunk[config.session_id_col].astype("string")
    chunk[config.event_timestamp_col] = pd.to_numeric(
        chunk[config.event_timestamp_col],
        errors="coerce",
        downcast="integer",
    )
    chunk[config.event_datetime_col] = pd.to_datetime(
        chunk[config.event_datetime_col],
        errors="coerce",
        utc=False,
    )

    before_ts_rows = len(chunk)
    chunk = chunk.dropna(subset=[config.event_timestamp_col, config.event_datetime_col])
    dropped_rows += before_ts_rows - len(chunk)

    chunk = chunk.rename(
        columns={
            config.user_col: "user_id",
            config.event_name_col: "event_name",
            config.event_json_col: "event_json",
            config.event_datetime_col: "event_datetime",
            config.event_timestamp_col: "event_timestamp",
            config.session_id_col: "session_id",
        }
    )

    return chunk, {"rows_in": before_rows, "rows_out": len(chunk), "rows_dropped": dropped_rows}


def ingest_csv_to_parquet(
    config: IngestConfig,
    project_root: str | Path = ".",
    max_chunks: int | None = None,
) -> dict[str, Any]:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm.auto import tqdm

    root = Path(project_root)
    csv_path = root / config.raw_csv_path
    interim_dir = root / config.interim_dir
    manifest_path = root / config.manifest_path

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    interim_dir.mkdir(parents=True, exist_ok=True)

    shard_records: list[dict[str, Any]] = []
    total_rows_in = 0
    total_rows_out = 0
    total_rows_dropped = 0

    reader = pd.read_csv(
        csv_path,
        chunksize=config.chunksize,
        encoding=config.encoding,
        dtype="string",
        low_memory=False,
    )

    for shard_idx, raw_chunk in enumerate(tqdm(reader, desc="CSV chunks")):
        if max_chunks is not None and shard_idx >= max_chunks:
            break

        chunk, stats = _prepare_events_chunk(raw_chunk, config)
        shard_path = interim_dir / f"events_shard_{shard_idx:05d}.parquet"
        tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_table(
            table,
            tmp_path,
            compression=config.compression,
            row_group_size=config.row_group_size,
        )
        os.replace(tmp_path, shard_path)

        total_rows_in += stats["rows_in"]
        total_rows_out += stats["rows_out"]
        total_rows_dropped += stats["rows_dropped"]
        shard_records.append(
            {
                "shard_idx": shard_idx,
                "path": str(shard_path.relative_to(root)),
                **stats,
            }
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_csv_path": config.raw_csv_path,
        "interim_dir": config.interim_dir,
        "chunksize": config.chunksize,
        "compression": config.compression,
        "row_group_size": config.row_group_size,
        "num_shards": len(shard_records),
        "rows_in": total_rows_in,
        "rows_out": total_rows_out,
        "rows_dropped": total_rows_dropped,
        "shards": shard_records,
    }
    atomic_write_json(manifest, manifest_path)
    return manifest
