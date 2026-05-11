from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")
_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)$")


@dataclass(frozen=True)
class JsonParseConfig:
    max_depth: int = 8
    max_leaf_tokens: int = 48
    max_value_chars: int = 80
    numeric_bucket_base: int = 2


def normalize_text(value: Any, max_chars: int = 80) -> str:
    text = _WHITESPACE_RE.sub(" ", str(value).strip())
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def normalize_path_part(value: Any, max_chars: int = 80) -> str:
    return normalize_text(value, max_chars=max_chars).replace(".", "_")


def parse_numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        if math.isfinite(float(value)):
            return float(value)
        return None

    text = str(value).strip().replace(",", ".")
    if not _NUMERIC_RE.match(text):
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def numeric_bucket(value: float, base: int = 2) -> str:
    if value == 0:
        return "zero"

    sign = "neg" if value < 0 else "pos"
    abs_value = abs(value)
    if abs_value < 1:
        return f"{sign}_lt_1"

    exponent = int(math.floor(math.log(abs_value, base)))
    low = base**exponent
    high = base ** (exponent + 1)
    return f"{sign}_{low:g}_{high:g}"


def canonicalize_json(value: Any, max_value_chars: int = 80) -> Any:
    if isinstance(value, dict):
        return {
            normalize_text(key, max_chars=max_value_chars): canonicalize_json(child, max_value_chars)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [canonicalize_json(child, max_value_chars) for child in value]
    if isinstance(value, str):
        return normalize_text(value, max_chars=max_value_chars)
    return value


def flatten_json(
    value: Any,
    prefix: str = "",
    max_depth: int = 8,
    max_value_chars: int = 80,
) -> list[tuple[str, Any]]:
    if max_depth < 0:
        return [(prefix or "__root__", "__max_depth__")]

    if isinstance(value, dict):
        items: list[tuple[str, Any]] = []
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            key_part = normalize_path_part(key, max_chars=max_value_chars)
            child_prefix = f"{prefix}.{key_part}" if prefix else key_part
            items.extend(flatten_json(child, child_prefix, max_depth - 1, max_value_chars))
        return items

    if isinstance(value, list):
        items = []
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            items.extend(flatten_json(child, child_prefix, max_depth - 1, max_value_chars))
        return items

    return [(prefix or "__root__", value)]


def parse_event_json(raw_json: Any, config: JsonParseConfig | None = None) -> dict[str, Any]:
    cfg = config or JsonParseConfig()
    raw_text = "{}" if raw_json is None else str(raw_json).strip()
    if not raw_text:
        raw_text = "{}"

    try:
        parsed = json.loads(raw_text)
        parse_error = ""
    except json.JSONDecodeError as exc:
        parsed = {}
        parse_error = f"{exc.__class__.__name__}: {exc.msg}"

    if not isinstance(parsed, dict):
        parsed = {"__root__": parsed}

    canonical_obj = canonicalize_json(parsed, max_value_chars=cfg.max_value_chars)
    canonical_json = json.dumps(canonical_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    top_keys = [normalize_path_part(key, cfg.max_value_chars) for key in sorted(parsed.keys(), key=str)]
    leaf_items = flatten_json(
        parsed,
        max_depth=cfg.max_depth,
        max_value_chars=cfg.max_value_chars,
    )[: cfg.max_leaf_tokens]

    leaf_paths: list[str] = []
    attribute_tokens: list[str] = []
    numeric_bucket_tokens: list[str] = []

    for path, value in leaf_items:
        path = normalize_path_part(path, cfg.max_value_chars)
        leaf_paths.append(path)
        attribute_tokens.append(f"path={path}")

        parsed_number = parse_numeric(value)
        if parsed_number is None:
            normalized_value = normalize_text(value, max_chars=cfg.max_value_chars)
            attribute_tokens.append(f"cat={path}:{normalized_value}")
        else:
            bucket = numeric_bucket(parsed_number, base=cfg.numeric_bucket_base)
            numeric_bucket_tokens.append(f"num={path}:{bucket}")

    return {
        "event_json_canonical": canonical_json,
        "json_top_keys": top_keys,
        "json_leaf_paths": leaf_paths,
        "attribute_tokens": attribute_tokens,
        "numeric_bucket_tokens": numeric_bucket_tokens,
        "json_parse_error": parse_error,
    }
