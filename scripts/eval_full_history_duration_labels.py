from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval_downstream import DownstreamEvalConfig, evaluate_downstream

SECONDS_PER_DAY = 86_400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate full-history BERT artifacts with SimCLR-style duration labels "
            "computed from the full user lifetime, not from the last-512 window."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-name", default="bert_full_history_last512")
    parser.add_argument("--days", type=int, nargs="+", default=[7, 14, 30])
    parser.add_argument("--output-suffix", default="full_duration")
    return parser.parse_args()


def load_label_lookup(root: Path, days: list[int]) -> pd.DataFrame:
    frames = []
    for split in ("train", "valid", "test"):
        path = root / "data" / "splits" / f"users_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=["user_id", "split", "first_event_ts", "last_event_ts"])
        frame["split"] = split
        duration = frame["last_event_ts"].astype("int64") - frame["first_event_ts"].astype("int64")
        frame["label_first_event_ts"] = frame["first_event_ts"].astype("int64")
        frame["label_last_event_ts"] = frame["last_event_ts"].astype("int64")
        for day in days:
            target = f"retention_{day}d"
            frame[f"label_available_{target}"] = True
            frame[f"label_{target}"] = (duration >= day * SECONDS_PER_DAY).astype("int64")
        frames.append(frame.drop(columns=["first_event_ts", "last_event_ts"]))
    return pd.concat(frames, ignore_index=True)


def rewrite_labels(input_path: Path, output_path: Path, labels: pd.DataFrame, days: list[int]) -> pd.DataFrame:
    frame = pd.read_parquet(input_path)
    drop_cols = ["label_first_event_ts", "label_last_event_ts"]
    for day in days:
        target = f"retention_{day}d"
        drop_cols.extend([f"label_available_{target}", f"label_{target}"])
    frame = frame.drop(columns=drop_cols, errors="ignore")
    frame = frame.merge(labels, on=["user_id", "split"], how="left", validate="many_to_one")
    missing = int(frame["label_first_event_ts"].isna().sum())
    if missing:
        raise ValueError(f"{input_path} has {missing} rows without full-duration labels")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return frame


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    run = args.run_name
    days = sorted(set(args.days))
    suffix = args.output_suffix

    input_paths = {
        "main_cls": root / "data" / "exports" / f"{run}_cls.parquet",
        "main_mean": root / "data" / "exports" / f"{run}_mean.parquet",
        "main_readout": root / "data" / "exports" / f"{run}_readout.parquet",
        "baseline_features": root / "data" / "exports" / f"{run}_baseline_features.parquet",
        "baseline_embeddings": root / "data" / "exports" / f"{run}_baseline_embeddings_128d.parquet",
    }
    output_paths = {
        name: root / "data" / "exports" / f"{run}_{suffix}_{name}.parquet"
        for name in input_paths
    }

    labels = load_label_lookup(root, days)
    for name, input_path in input_paths.items():
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        rewrite_labels(input_path, output_paths[name], labels, days)

    targets = tuple(f"retention_{day}d" for day in days)
    evaluate_downstream(
        DownstreamEvalConfig(
            inputs={name: str(path.relative_to(root)) for name, path in output_paths.items()},
            outputs={
                "metrics": f"artifacts/reports/{run}_{suffix}_downstream_metrics.csv",
                "manifest": f"artifacts/manifests/{run}_{suffix}_downstream_eval_manifest.json",
            },
            targets=targets,
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


if __name__ == "__main__":
    main()
