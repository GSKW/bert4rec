from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import BaselineConfig, build_baseline_features
from src.eval_downstream import DownstreamEvalConfig, evaluate_downstream
from src.export_embeddings import ExportEmbeddingsConfig, export_main_embeddings
from src.io_utils import atomic_write_json
from src.sequences import (
    SECONDS_PER_DAY,
    SequenceConfig,
    create_sorted_event_token_parquet,
    load_sequence_config,
    load_split_map,
    time_gap_id,
)
from src.training import TrainConfig, load_train_config, train_main_embedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the BERT/Transformer encoder in a SimCLR-like "
            "full-history last-N regime while keeping the existing BERT tokenization."
        )
    )
    parser.add_argument("--project-root", default=".", help="Repository root.")
    parser.add_argument("--sequence-config", default="configs/sequences.yaml")
    parser.add_argument("--train-template", default="configs/train_contrastive_step1.yaml")
    parser.add_argument("--run-name", default="full_history_bert_last512")
    parser.add_argument("--max-history-len", type=int, default=512)
    parser.add_argument("--min-events", type=int, default=10)
    parser.add_argument("--retention-days", type=int, nargs="+", default=[7, 14, 30])
    parser.add_argument(
        "--downstream-targets",
        nargs="+",
        default=["retention_7d", "retention_14d", "retention_30d"],
    )
    parser.add_argument("--dry-run-users", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-data", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-downstream", action="store_true")
    return parser.parse_args()


def _duration_label(first_ts: int, last_ts: int, days: int) -> int:
    return int((last_ts - first_ts) >= days * SECONDS_PER_DAY)


def _window_gaps(timestamps: list[int]) -> list[int]:
    if not timestamps:
        return []
    gaps = [0]
    for idx in range(1, len(timestamps)):
        gaps.append(time_gap_id(int(timestamps[idx]) - int(timestamps[idx - 1])))
    return gaps


def _session_flags(session_ids: list[str]) -> list[int]:
    if not session_ids:
        return []
    flags = [1]
    for idx in range(1, len(session_ids)):
        flags.append(1 if str(session_ids[idx]) != str(session_ids[idx - 1]) else 0)
    return flags


def _record_for_window(
    user_id: str,
    split: str,
    event_ids: list[int],
    timestamps: list[int],
    session_ids: list[str],
    next_event_id: int,
    max_history_len: int,
    retention_days: list[int],
    total_events: int,
    label_first_ts: int | None = None,
    label_last_ts: int | None = None,
) -> dict[str, Any]:
    window_first_ts = int(timestamps[0])
    window_last_ts = int(timestamps[-1])
    full_first_ts = window_first_ts if label_first_ts is None else int(label_first_ts)
    full_last_ts = window_last_ts if label_last_ts is None else int(label_last_ts)
    record = {
        "user_id": user_id,
        "split": split,
        "prefix_len": max_history_len,
        "num_events_total": total_events,
        "prefix_start_ts": window_first_ts,
        "prefix_end_ts": window_last_ts,
        "label_first_event_ts": full_first_ts,
        "label_last_event_ts": full_last_ts,
        "event_token_ids": [int(value) for value in event_ids],
        "time_gap_ids": _window_gaps(timestamps),
        "session_flags": _session_flags(session_ids),
        "next_event_token_id": int(next_event_id),
    }
    for days in retention_days:
        target = f"retention_{days}d"
        record[f"label_available_{target}"] = True
        record[f"label_{target}"] = _duration_label(full_first_ts, full_last_ts, days)
    return record


def build_full_history_frames(
    sequence_config: SequenceConfig,
    project_root: Path,
    run_name: str,
    max_history_len: int,
    min_events: int,
    retention_days: list[int],
    dry_run_users: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm.auto import tqdm

    output_dir = project_root / "data" / "processed" / run_name
    train_paths = {split: output_dir / f"{split}_train_windows.parquet" for split in ("train", "valid", "test")}
    eval_paths = {split: output_dir / f"{split}_full_history.parquet" for split in ("train", "valid", "test")}
    manifest_path = project_root / "artifacts" / "manifests" / f"{run_name}_dataset_manifest.json"

    if manifest_path.exists() and not overwrite:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            **payload,
            "train_prefixes": {split: str(path.relative_to(project_root)) for split, path in train_paths.items()},
            "eval_prefixes": {split: str(path.relative_to(project_root)) for split, path in eval_paths.items()},
        }

    sorted_path = create_sorted_event_token_parquet(sequence_config, project_root=project_root, overwrite=False)
    split_map = None
    if sequence_config.split_map_path is not None:
        split_map = load_split_map(
            project_root / sequence_config.split_map_path,
            user_col=sequence_config.split_map_user_col,
            split_col=sequence_config.split_map_split_col,
        )

    records_train = {"train": [], "valid": [], "test": []}
    records_eval = {"train": [], "valid": [], "test": []}
    stats = {
        "users_seen": 0,
        "users_kept": 0,
        "users_too_short": 0,
        "users_missing_split": 0,
    }

    parquet_file = pq.ParquetFile(sorted_path)
    current_user: str | None = None
    current_event_ids: list[int] = []
    current_timestamps: list[int] = []
    current_session_ids: list[str] = []

    def flush_user() -> None:
        nonlocal current_user, current_event_ids, current_timestamps, current_session_ids
        if current_user is None:
            return
        stats["users_seen"] += 1
        if len(current_event_ids) < min_events:
            stats["users_too_short"] += 1
        else:
            split = split_map.get(current_user) if split_map is not None else None
            if split is None:
                stats["users_missing_split"] += 1
            else:
                eval_start = max(0, len(current_event_ids) - max_history_len)
                eval_ids = current_event_ids[eval_start:]
                eval_ts = current_timestamps[eval_start:]
                eval_sessions = current_session_ids[eval_start:]

                train_end = max(1, len(current_event_ids) - 1)
                train_start = max(0, train_end - max_history_len)
                train_ids = current_event_ids[train_start:train_end]
                train_ts = current_timestamps[train_start:train_end]
                train_sessions = current_session_ids[train_start:train_end]
                next_event_id = current_event_ids[-1]

                if train_ids:
                    full_first_ts = int(current_timestamps[0])
                    full_last_ts = int(current_timestamps[-1])
                    records_train[split].append(
                        _record_for_window(
                            current_user,
                            split,
                            train_ids,
                            train_ts,
                            train_sessions,
                            next_event_id,
                            max_history_len,
                            retention_days,
                            len(current_event_ids),
                            full_first_ts,
                            full_last_ts,
                        )
                    )
                    records_eval[split].append(
                        _record_for_window(
                            current_user,
                            split,
                            eval_ids,
                            eval_ts,
                            eval_sessions,
                            next_event_id,
                            max_history_len,
                            retention_days,
                            len(current_event_ids),
                            full_first_ts,
                            full_last_ts,
                        )
                    )
                    stats["users_kept"] += 1

        current_user = None
        current_event_ids = []
        current_timestamps = []
        current_session_ids = []

    for row_group_idx in tqdm(range(parquet_file.num_row_groups), desc="Build full-history windows"):
        table = parquet_file.read_row_group(
            row_group_idx,
            columns=["user_id", "event_timestamp", "session_id", "event_token_id"],
        )
        for user_id, timestamp, session_id, event_id in zip(
            table["user_id"].combine_chunks().to_pylist(),
            table["event_timestamp"].combine_chunks().to_pylist(),
            table["session_id"].combine_chunks().to_pylist(),
            table["event_token_id"].combine_chunks().to_pylist(),
            strict=True,
        ):
            user_id = str(user_id)
            if current_user is not None and user_id != current_user:
                flush_user()
                if dry_run_users is not None and stats["users_kept"] >= dry_run_users:
                    break
            if current_user is None:
                current_user = user_id
            current_event_ids.append(int(event_id))
            current_timestamps.append(int(timestamp))
            current_session_ids.append(str(session_id))
        if dry_run_users is not None and stats["users_kept"] >= dry_run_users:
            break
    if dry_run_users is None or stats["users_kept"] < dry_run_users:
        flush_user()

    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "valid", "test"):
        for paths, records in ((train_paths, records_train), (eval_paths, records_eval)):
            table = pa.Table.from_pylist(records[split])
            pq.write_table(table, paths[split], compression="zstd", row_group_size=100_000)
        split_counts[split] = {
            "train_windows": len(records_train[split]),
            "eval_windows": len(records_eval[split]),
        }

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "sorted_events_path": str(sorted_path.relative_to(project_root)),
        "split_source": sequence_config.split_map_path,
        "max_history_len": max_history_len,
        "min_events": min_events,
        "retention_days": retention_days,
        "dry_run_users": dry_run_users,
        "label_definition": "retention_Xd = last_event_ts - first_event_ts >= X days (SimCLR-style duration label)",
        "train_window_definition": "last max_history_len events before the final event; final event is next-event target",
        "eval_window_definition": "last max_history_len events of full available user history",
        "stats": stats,
        "splits": split_counts,
        "train_prefixes": {split: str(path.relative_to(project_root)) for split, path in train_paths.items()},
        "eval_prefixes": {split: str(path.relative_to(project_root)) for split, path in eval_paths.items()},
    }
    atomic_write_json(manifest, manifest_path)
    return manifest


def make_train_config(args: argparse.Namespace, dataset_manifest: dict[str, Any], root: Path) -> TrainConfig:
    base = load_train_config(root / args.train_template)
    paths = copy.deepcopy(base.paths)
    train = copy.deepcopy(base.train)
    model = copy.deepcopy(base.model)
    wandb = copy.deepcopy(base.wandb)

    paths["train_prefixes"] = dataset_manifest["train_prefixes"]["train"]
    paths["valid_prefixes"] = dataset_manifest["train_prefixes"]["valid"]
    paths["test_prefixes"] = dataset_manifest["train_prefixes"]["test"]
    paths["checkpoint_dir"] = f"artifacts/checkpoints/{args.run_name}"
    paths["run_state_path"] = f"artifacts/manifests/{args.run_name}_train_run_state.json"

    model["max_seq_len"] = args.max_history_len + 1
    wandb["run_name"] = args.run_name
    wandb["group"] = "full-history-bert"
    wandb["tags"] = sorted(set([*wandb.get("tags", []), "full-history", "last-512", "simclr-like"]))
    wandb["mode"] = args.wandb_mode

    if args.max_steps is not None:
        train["max_steps"] = args.max_steps
    if args.epochs is not None:
        train["epochs"] = args.epochs
    if args.batch_size is not None:
        train["batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        train["eval_batch_size"] = args.eval_batch_size
    train["num_workers"] = args.num_workers

    return TrainConfig(paths=paths, wandb=wandb, model=model, train=train)


def write_runtime_train_config(train_config: TrainConfig, root: Path, run_name: str) -> str:
    import yaml

    rel_path = Path("artifacts") / "manifests" / f"{run_name}_train_config.yaml"
    full_path = root / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(
        yaml.safe_dump(asdict(train_config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return str(rel_path).replace("\\", "/")


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.project_root).resolve()
    sequence_config = load_sequence_config(root / args.sequence_config)
    dataset_manifest = build_full_history_frames(
        sequence_config=sequence_config,
        project_root=root,
        run_name=args.run_name,
        max_history_len=args.max_history_len,
        min_events=args.min_events,
        retention_days=args.retention_days,
        dry_run_users=args.dry_run_users,
        overwrite=args.overwrite_data,
    )

    train_config = make_train_config(args, dataset_manifest, root)
    runtime_train_config_path = write_runtime_train_config(train_config, root, args.run_name)
    train_result = None
    if not args.skip_train:
        train_result = train_main_embedder(
            train_config,
            project_root=root,
            resume=args.resume,
            wandb_mode_override=args.wandb_mode,
        )

    checkpoint_path = train_config.paths["checkpoint_dir"] + "/checkpoint_best.pt"
    if train_result is not None and train_result.get("checkpoint_best"):
        checkpoint_path = str(Path(train_result["checkpoint_best"]).relative_to(root))

    export_manifest = None
    if not args.skip_export:
        export_manifest = export_main_embeddings(
            ExportEmbeddingsConfig(
                checkpoint_path=checkpoint_path,
                event_vocab_path=train_config.paths["event_vocab"],
                train_config_path=runtime_train_config_path,
                manifest_path=f"artifacts/manifests/{args.run_name}_embeddings_manifest.json",
                input_prefixes=dataset_manifest["eval_prefixes"],
                outputs={
                    "cls": f"data/exports/{args.run_name}_cls.parquet",
                    "mean": f"data/exports/{args.run_name}_mean.parquet",
                    "readout": f"data/exports/{args.run_name}_readout.parquet",
                },
                batch_size=512,
                num_workers=0,
                device="auto",
                mixed_precision=True,
                dry_run_rows=None,
            ),
            project_root=root,
        )

    baseline_manifest = None
    if not args.skip_baseline:
        baseline_manifest = build_baseline_features(
            BaselineConfig(
                input_prefixes=dataset_manifest["eval_prefixes"],
                outputs={
                    "features": f"data/exports/{args.run_name}_baseline_features.parquet",
                    "embeddings": f"data/exports/{args.run_name}_baseline_embeddings_128d.parquet",
                    "markov_stats": f"artifacts/baselines/{args.run_name}_markov_transition_stats.parquet",
                    "manifest": f"artifacts/manifests/{args.run_name}_baseline_manifest.json",
                },
                svd_dim=128,
                hash_features=262_144,
                top_markov_k=10,
                last_k=5,
                random_state=42,
                dry_run_rows=None,
            ),
            project_root=root,
        )

    downstream_manifest = None
    if not args.skip_downstream:
        downstream_manifest = evaluate_downstream(
            DownstreamEvalConfig(
                inputs={
                    "main_cls": f"data/exports/{args.run_name}_cls.parquet",
                    "main_mean": f"data/exports/{args.run_name}_mean.parquet",
                    "main_readout": f"data/exports/{args.run_name}_readout.parquet",
                    "baseline_features": f"data/exports/{args.run_name}_baseline_features.parquet",
                    "baseline_embeddings": f"data/exports/{args.run_name}_baseline_embeddings_128d.parquet",
                },
                outputs={
                    "metrics": f"artifacts/reports/{args.run_name}_downstream_metrics.csv",
                    "manifest": f"artifacts/manifests/{args.run_name}_downstream_eval_manifest.json",
                },
                targets=tuple(args.downstream_targets),
                random_state=42,
                dry_run_rows=None,
                include_legacy_proxy_feature_models=False,
                model_subset=(
                    "baseline_features_hgb_clean",
                    "baseline_svd_logreg",
                    "combined_cls_baseline_svd_logreg",
                    "main_cls_logreg",
                    "main_mean_logreg",
                    "main_readout_logreg",
                ),
                embedding_preprocess_mode="l2_pca_standardize",
                embedding_pca_components=64,
                include_mlp_probe_models=False,
                mlp_hidden_dim=128,
                mlp_max_iter=120,
            ),
            project_root=root,
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "train_config": asdict(train_config),
        "dataset_manifest": dataset_manifest,
        "train_result": train_result,
        "export_manifest": export_manifest,
        "baseline_manifest": baseline_manifest,
        "downstream_manifest": downstream_manifest,
    }
    atomic_write_json(summary, root / "artifacts" / "manifests" / f"{args.run_name}_pipeline_summary.json")
    return summary


if __name__ == "__main__":
    run_pipeline(parse_args())
