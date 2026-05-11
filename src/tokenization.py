from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.io_utils import atomic_write_json, load_yaml


@dataclass(frozen=True)
class VocabLimit:
    min_freq: int = 1
    max_size: int | None = None


@dataclass(frozen=True)
class VocabConfig:
    input_manifest_path: str = "artifacts/manifests/normalize_manifest.json"
    vocab_dir: str = "artifacts/vocab"
    manifest_path: str = "artifacts/manifests/vocab_manifest.json"
    special_tokens: tuple[str, ...] = ("[PAD]", "[UNK]", "[MASK]", "[CLS]")
    sources: dict[str, str] | None = None
    limits: dict[str, VocabLimit] | None = None


def load_vocab_config(path: str | Path) -> VocabConfig:
    raw = load_yaml(path)
    special_raw = raw.get("special_tokens", {})
    special_tokens = (
        special_raw.get("pad", "[PAD]"),
        special_raw.get("unk", "[UNK]"),
        special_raw.get("mask", "[MASK]"),
        special_raw.get("cls", "[CLS]"),
    )
    limits = {
        name: VocabLimit(
            min_freq=int(limit.get("min_freq", 1)),
            max_size=limit.get("max_size"),
        )
        for name, limit in raw.get("limits", {}).items()
    }
    limits = {
        name: VocabLimit(limit.min_freq, int(limit.max_size) if limit.max_size is not None else None)
        for name, limit in limits.items()
    }
    return VocabConfig(
        input_manifest_path=raw["input_manifest_path"],
        vocab_dir=raw["vocab_dir"],
        manifest_path=raw["manifest_path"],
        special_tokens=special_tokens,
        sources=raw["sources"],
        limits=limits,
    )


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _update_counter_from_value_counts(counter: Counter[str], value_counts: Any) -> None:
    for item in value_counts.to_pylist():
        token = item.get("values")
        if token is not None:
            counter[str(token)] += int(item["counts"])


def _is_list_type(arrow_type: Any) -> bool:
    import pyarrow.types as pat

    return pat.is_list(arrow_type) or pat.is_large_list(arrow_type)


def _update_counter_from_arrow(counter: Counter[str], array: Any) -> None:
    import pyarrow.compute as pc

    if _is_list_type(array.type):
        array = pc.list_flatten(array)
    value_counts = pc.value_counts(array)
    _update_counter_from_value_counts(counter, value_counts)


def _build_vocab(counter: Counter[str], special_tokens: tuple[str, ...], limit: VocabLimit) -> dict[str, int]:
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    candidates = (
        (token, freq)
        for token, freq in counter.items()
        if freq >= limit.min_freq and token not in vocab
    )
    sorted_tokens = sorted(candidates, key=lambda item: (-item[1], item[0]))

    max_regular_tokens = None
    if limit.max_size is not None:
        max_regular_tokens = max(limit.max_size - len(vocab), 0)
        sorted_tokens = sorted_tokens[:max_regular_tokens]

    for token, _freq in sorted_tokens:
        vocab[token] = len(vocab)
    return vocab


def _coverage(counter: Counter[str], vocab: dict[str, int], special_tokens: tuple[str, ...]) -> dict[str, Any]:
    total_occurrences = int(sum(counter.values()))
    total_unique = len(counter)
    kept_tokens = set(vocab) - set(special_tokens)
    kept_occurrences = int(sum(counter[token] for token in kept_tokens))
    return {
        "total_occurrences": total_occurrences,
        "total_unique": total_unique,
        "kept_unique": len(kept_tokens),
        "kept_occurrences": kept_occurrences,
        "oov_unique": total_unique - len(kept_tokens),
        "oov_occurrences": total_occurrences - kept_occurrences,
        "coverage_occurrences": kept_occurrences / total_occurrences if total_occurrences else 0.0,
    }


def save_counter(counter: Counter[str], path: str | Path) -> Path:
    rows = [
        {"token": token, "count": int(count)}
        for token, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]
    return atomic_write_json({"tokens": rows}, path)


def build_vocabs_from_normalized_shards(
    config: VocabConfig,
    project_root: str | Path = ".",
    max_shards: int | None = None,
) -> dict[str, Any]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from tqdm.auto import tqdm

    if config.sources is None or config.limits is None:
        raise ValueError("VocabConfig.sources and VocabConfig.limits must be set")

    root = Path(project_root)
    normalized_manifest = load_manifest(root / config.input_manifest_path)
    vocab_dir = root / config.vocab_dir
    vocab_dir.mkdir(parents=True, exist_ok=True)

    counters = {name: Counter() for name in config.sources}
    selected_shards = normalized_manifest["shards"][:max_shards]
    rows_seen = 0
    parse_errors = 0

    required_columns = sorted(set(config.sources.values()) | {"json_parse_error"})

    for shard in tqdm(selected_shards, desc="Build vocab"):
        shard_path = root / shard["output_path"]
        parquet_file = pq.ParquetFile(shard_path)

        for row_group_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_idx, columns=required_columns)
            rows_seen += table.num_rows

            parse_error_mask = pc.not_equal(table["json_parse_error"], "")
            parse_errors += int(pc.sum(pc.cast(parse_error_mask, "int64")).as_py() or 0)

            for vocab_name, column in config.sources.items():
                _update_counter_from_arrow(counters[vocab_name], table[column].combine_chunks())

    vocab_records: dict[str, Any] = {}
    for vocab_name, counter in counters.items():
        limit = config.limits.get(vocab_name, VocabLimit())
        vocab = _build_vocab(counter, config.special_tokens, limit)
        vocab_path = vocab_dir / f"{vocab_name}_vocab.json"
        freq_path = vocab_dir / f"{vocab_name}_frequencies.json"

        atomic_write_json(
            {
                "special_tokens": list(config.special_tokens),
                "min_freq": limit.min_freq,
                "max_size": limit.max_size,
                "token_to_id": vocab,
            },
            vocab_path,
        )
        save_counter(counter, freq_path)

        vocab_records[vocab_name] = {
            "source_column": config.sources[vocab_name],
            "vocab_path": str(vocab_path.relative_to(root)),
            "frequency_path": str(freq_path.relative_to(root)),
            "size": len(vocab),
            "min_freq": limit.min_freq,
            "max_size": limit.max_size,
            **_coverage(counter, vocab, config.special_tokens),
            "top_tokens": [
                {"token": token, "count": int(count)}
                for token, count in counter.most_common(20)
            ],
        }

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_path": config.input_manifest_path,
        "num_shards": len(selected_shards),
        "rows_seen": rows_seen,
        "parse_errors_seen": parse_errors,
        "special_tokens": list(config.special_tokens),
        "vocabs": vocab_records,
    }
    atomic_write_json(manifest, root / config.manifest_path)
    return manifest
