from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.io_utils import atomic_write_json, load_yaml
from src.json_parser import JsonParseConfig, parse_event_json


@dataclass(frozen=True)
class NormalizeConfig:
    input_manifest_path: str = "artifacts/manifests/ingest_manifest.json"
    output_dir: str = "data/interim"
    output_pattern: str = "events_normalized_{shard_idx:05d}.parquet"
    manifest_path: str = "artifacts/manifests/normalize_manifest.json"
    compression: str = "zstd"
    row_group_size: int = 100_000
    json: JsonParseConfig = JsonParseConfig()


def load_normalize_config(path: str | Path) -> NormalizeConfig:
    raw = load_yaml(path)
    parquet_cfg = raw.get("parquet", {})
    json_cfg = raw.get("json", {})
    return NormalizeConfig(
        input_manifest_path=raw["input_manifest_path"],
        output_dir=raw["output_dir"],
        output_pattern=raw["output_pattern"],
        manifest_path=raw["manifest_path"],
        compression=parquet_cfg.get("compression", "zstd"),
        row_group_size=int(parquet_cfg.get("row_group_size", 100_000)),
        json=JsonParseConfig(
            max_depth=int(json_cfg.get("max_depth", 8)),
            max_leaf_tokens=int(json_cfg.get("max_leaf_tokens", 48)),
            max_value_chars=int(json_cfg.get("max_value_chars", 80)),
            numeric_bucket_base=int(json_cfg.get("numeric_bucket_base", 2)),
        ),
    )


def load_ingest_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parquet_num_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def normalize_events_frame(events: Any, config: NormalizeConfig) -> tuple[Any, dict[str, int]]:
    import pandas as pd

    events = events.copy()
    events["event_name_norm"] = events["event_name"].astype("string").str.strip()
    events["event_token"] = "event=" + events["event_name_norm"].astype("string")

    unique_json = events["event_json"].astype("string").drop_duplicates()
    parsed_by_raw = {
        raw_json: parse_event_json(raw_json, config.json)
        for raw_json in unique_json
    }

    parsed_series = events["event_json"].astype("string").map(parsed_by_raw)
    parsed_frame = pd.DataFrame(parsed_series.tolist(), index=events.index)
    normalized = pd.concat([events, parsed_frame], axis=1)

    top_key_signature = normalized["json_top_keys"].map(lambda keys: "+".join(keys) if keys else "__empty__")
    normalized["event_signature"] = normalized["event_token"] + "|json=" + top_key_signature

    parse_errors = int((normalized["json_parse_error"] != "").sum())
    return normalized, {"rows": len(normalized), "parse_errors": parse_errors}


def normalize_parquet_shards(
    config: NormalizeConfig,
    project_root: str | Path = ".",
    max_shards: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm.auto import tqdm

    root = Path(project_root)
    ingest_manifest = load_ingest_manifest(root / config.input_manifest_path)
    output_dir = root / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_records: list[dict[str, Any]] = []
    total_rows = 0
    total_parse_errors = 0
    selected_shards = ingest_manifest["shards"][:max_shards]
    previous_manifest_path = root / config.manifest_path
    previous_manifest = load_ingest_manifest(previous_manifest_path) if previous_manifest_path.exists() else {}
    previous_stats_by_shard = {
        int(record["shard_idx"]): record
        for record in previous_manifest.get("shards", [])
    }

    for shard_record in tqdm(selected_shards, desc="Normalize shards"):
        shard_idx = int(shard_record["shard_idx"])
        input_path = root / shard_record["path"]
        output_path = output_dir / config.output_pattern.format(shard_idx=shard_idx)

        if output_path.exists() and not overwrite:
            rows = _parquet_num_rows(output_path)
            previous_stats = previous_stats_by_shard.get(shard_idx, {})
            parse_errors = previous_stats.get("parse_errors", 0)
            shard_records.append(
                {
                    "shard_idx": shard_idx,
                    "input_path": str(input_path.relative_to(root)),
                    "output_path": str(output_path.relative_to(root)),
                    "rows": rows,
                    "parse_errors": parse_errors,
                    "status": "skipped_existing",
                }
            )
            total_rows += rows
            total_parse_errors += int(parse_errors or 0)
            continue

        events = pd.read_parquet(input_path)
        normalized, stats = normalize_events_frame(events, config)

        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        table = pa.Table.from_pandas(normalized, preserve_index=False)
        pq.write_table(
            table,
            tmp_path,
            compression=config.compression,
            row_group_size=config.row_group_size,
        )
        os.replace(tmp_path, output_path)

        total_rows += stats["rows"]
        total_parse_errors += stats["parse_errors"]
        shard_records.append(
            {
                "shard_idx": shard_idx,
                "input_path": str(input_path.relative_to(root)),
                "output_path": str(output_path.relative_to(root)),
                **stats,
                "status": "written",
            }
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_path": config.input_manifest_path,
        "output_dir": config.output_dir,
        "num_shards": len(shard_records),
        "rows": total_rows,
        "parse_errors": total_parse_errors,
        "shards": shard_records,
    }
    atomic_write_json(manifest, root / config.manifest_path)
    return manifest
