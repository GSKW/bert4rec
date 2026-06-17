from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

from src.io_utils import atomic_write_json, load_yaml


@dataclass(frozen=True)
class DownstreamEvalConfig:
    inputs: dict[str, str]
    outputs: dict[str, str]
    targets: tuple[str, ...]
    random_state: int
    dry_run_rows: int | None
    include_legacy_proxy_feature_models: bool
    model_subset: tuple[str, ...] | None
    embedding_preprocess_mode: str
    embedding_pca_components: int | None
    include_mlp_probe_models: bool
    mlp_hidden_dim: int
    mlp_max_iter: int


def load_downstream_eval_config(path: str | Path) -> DownstreamEvalConfig:
    raw = load_yaml(path)
    cfg = raw["eval"]
    return DownstreamEvalConfig(
        inputs=raw["inputs"],
        outputs=raw["outputs"],
        targets=tuple(cfg["targets"]),
        random_state=int(cfg["random_state"]),
        dry_run_rows=None if cfg.get("dry_run_rows") is None else int(cfg["dry_run_rows"]),
        include_legacy_proxy_feature_models=bool(cfg.get("include_legacy_proxy_feature_models", True)),
        model_subset=None if cfg.get("model_subset") is None else tuple(cfg["model_subset"]),
        embedding_preprocess_mode=str(cfg.get("embedding_preprocess_mode", "none")),
        embedding_pca_components=None if cfg.get("embedding_pca_components") is None else int(cfg["embedding_pca_components"]),
        include_mlp_probe_models=bool(cfg.get("include_mlp_probe_models", False)),
        mlp_hidden_dim=int(cfg.get("mlp_hidden_dim", 128)),
        mlp_max_iter=int(cfg.get("mlp_max_iter", 120)),
    )


def _embedding_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.vstack(frame["embedding"].to_numpy()).astype("float32")


def _feature_columns(frame: pd.DataFrame, include_proxy_next_event_features: bool = False) -> list[str]:
    excluded = {
        "user_id", "split", "prefix_start_ts", "prefix_end_ts",
    }
    if not include_proxy_next_event_features:
        excluded.update({"next_event_token_id", "markov_actual_prob", "markov_actual_rank"})
    cols = []
    for col in frame.columns:
        if col in excluded:
            continue
        if col.startswith("label_") or col.startswith("label_available_"):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def _limit_dry_run_frame(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows >= len(frame) or "split" not in frame.columns:
        return frame.head(max_rows).reset_index(drop=True)

    splits = list(frame["split"].drop_duplicates())
    rows_per_split = max(1, ceil(max_rows / len(splits)))
    limited = frame.groupby("split", sort=False, group_keys=False).head(rows_per_split)
    return limited.head(max_rows).reset_index(drop=True)


def _metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "logloss": float(log_loss(y_true, np.clip(prob, 1e-6, 1 - 1e-6), labels=[0, 1])),
    }


def _transform_embeddings(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    mode: str,
    pca_components: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    mode = mode.lower()
    if mode in {"none", "raw"}:
        return x_train, x_eval
    if mode in {"standardize", "zscore"}:
        scaler = StandardScaler()
        return scaler.fit_transform(x_train), scaler.transform(x_eval)
    if mode in {"l2_then_standardize", "l2_standardize"}:
        x_train_n = normalize(x_train)
        x_eval_n = normalize(x_eval)
        scaler = StandardScaler()
        return scaler.fit_transform(x_train_n), scaler.transform(x_eval_n)
    if mode in {"l2_pca_standardize", "l2_then_pca_then_standardize"}:
        x_train_n = normalize(x_train)
        x_eval_n = normalize(x_eval)
        max_components = min(x_train_n.shape[0] - 1, x_train_n.shape[1])
        if max_components < 1:
            return x_train_n, x_eval_n
        requested = pca_components if pca_components is not None else min(64, max_components)
        pca = PCA(n_components=max(1, min(requested, max_components)), random_state=42)
        x_train_p = pca.fit_transform(x_train_n)
        x_eval_p = pca.transform(x_eval_n)
        scaler = StandardScaler()
        return scaler.fit_transform(x_train_p), scaler.transform(x_eval_p)
    raise ValueError(f"Unsupported embedding_preprocess_mode: {mode}")


def _eval_model(
    frame: pd.DataFrame,
    x: np.ndarray,
    target: str,
    model_name: str,
    estimator: Any,
    embedding_preprocess_mode: str = "none",
    embedding_pca_components: int | None = None,
    use_embedding_preprocess: bool = False,
) -> list[dict[str, Any]]:
    label_col = f"label_{target}"
    avail_col = f"label_available_{target}"
    rows = []
    for prefix_len in sorted(frame["prefix_len"].unique()):
        mask_prefix = frame["prefix_len"].eq(prefix_len)
        train_mask = mask_prefix & frame["split"].eq("train") & frame[avail_col].eq(True) & frame[label_col].notna()
        if train_mask.sum() < 10:
            continue
        y_train = frame.loc[train_mask, label_col].astype(int).to_numpy()
        if len(np.unique(y_train)) < 2:
            continue
        x_train = x[train_mask.to_numpy()]
        clf = None
        if not use_embedding_preprocess:
            clf = estimator()
            clf.fit(x_train, y_train)
        for split in ["valid", "test"]:
            eval_mask = mask_prefix & frame["split"].eq(split) & frame[avail_col].eq(True) & frame[label_col].notna()
            if eval_mask.sum() == 0:
                continue
            y = frame.loc[eval_mask, label_col].astype(int).to_numpy()
            x_eval = x[eval_mask.to_numpy()]
            if use_embedding_preprocess:
                x_train_fit, x_eval_fit = _transform_embeddings(x_train, x_eval, embedding_preprocess_mode, embedding_pca_components)
                clf = estimator()
                clf.fit(x_train_fit, y_train)
                x_pred = x_eval_fit
            else:
                x_pred = x_eval
            if hasattr(clf, "predict_proba"):
                prob = clf.predict_proba(x_pred)[:, 1]
            else:
                prob = clf.decision_function(x_pred)
            metrics = _metrics(y, prob)
            rows.append({
                "model_name": model_name,
                "target": target,
                "prefix_len": int(prefix_len),
                "split": split,
                "num_rows": int(eval_mask.sum()),
                "positive_rate": float(y.mean()),
                **metrics,
            })
    return rows


def evaluate_downstream(config: DownstreamEvalConfig, project_root: str | Path = ".") -> dict[str, Any]:
    root = Path(project_root)
    cls = pd.read_parquet(root / config.inputs["main_cls"])
    mean = pd.read_parquet(root / config.inputs["main_mean"])
    base_features = pd.read_parquet(root / config.inputs["baseline_features"])
    base_embeddings = pd.read_parquet(root / config.inputs["baseline_embeddings"])
    readout_path = config.inputs.get("main_readout")
    readout = None
    if readout_path:
        full_readout_path = root / readout_path
        if full_readout_path.exists():
            readout = pd.read_parquet(full_readout_path)
    if config.dry_run_rows is not None:
        cls = _limit_dry_run_frame(cls, config.dry_run_rows)
        mean = _limit_dry_run_frame(mean, config.dry_run_rows)
        base_features = _limit_dry_run_frame(base_features, config.dry_run_rows)
        base_embeddings = _limit_dry_run_frame(base_embeddings, config.dry_run_rows)
        if readout is not None:
            readout = _limit_dry_run_frame(readout, config.dry_run_rows)

    feature_cols_clean = _feature_columns(base_features, include_proxy_next_event_features=False)
    feature_cols_legacy = _feature_columns(base_features, include_proxy_next_event_features=True)
    datasets = {
        "main_cls_logreg": (cls, _embedding_matrix(cls), lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.random_state), True),
        "main_mean_logreg": (mean, _embedding_matrix(mean), lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.random_state), True),
        "baseline_svd_logreg": (base_embeddings, _embedding_matrix(base_embeddings), lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.random_state), True),
        "baseline_features_logreg_clean": (base_features, base_features[feature_cols_clean].fillna(0.0).astype("float32").to_numpy(), lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.random_state)), False),
        "baseline_features_hgb_clean": (base_features, base_features[feature_cols_clean].fillna(0.0).astype("float32").to_numpy(), lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=config.random_state), False),
    }
    if config.include_mlp_probe_models:
        datasets["main_cls_mlp"] = (cls, _embedding_matrix(cls), lambda: MLPClassifier(hidden_layer_sizes=(config.mlp_hidden_dim,), max_iter=config.mlp_max_iter, early_stopping=True, random_state=config.random_state), True)
        datasets["main_mean_mlp"] = (mean, _embedding_matrix(mean), lambda: MLPClassifier(hidden_layer_sizes=(config.mlp_hidden_dim,), max_iter=config.mlp_max_iter, early_stopping=True, random_state=config.random_state), True)
    if config.include_legacy_proxy_feature_models:
        datasets["baseline_features_logreg_legacy"] = (base_features, base_features[feature_cols_legacy].fillna(0.0).astype("float32").to_numpy(), lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.random_state)), False)
        datasets["baseline_features_hgb_legacy"] = (base_features, base_features[feature_cols_legacy].fillna(0.0).astype("float32").to_numpy(), lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=config.random_state), False)
    if readout is not None:
        datasets["main_readout_logreg"] = (readout, _embedding_matrix(readout), lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.random_state), True)
        if config.include_mlp_probe_models:
            datasets["main_readout_mlp"] = (readout, _embedding_matrix(readout), lambda: MLPClassifier(hidden_layer_sizes=(config.mlp_hidden_dim,), max_iter=config.mlp_max_iter, early_stopping=True, random_state=config.random_state), True)
    combined = cls[["user_id", "split", "prefix_len"]].merge(base_embeddings[["user_id", "split", "prefix_len", "embedding"]], on=["user_id", "split", "prefix_len"], suffixes=("_main", "_base"))
    cls_lookup = cls.set_index(["user_id", "split", "prefix_len"])
    combined_meta = cls_lookup.loc[pd.MultiIndex.from_frame(combined[["user_id", "split", "prefix_len"]])].reset_index()
    combined_x = np.hstack([_embedding_matrix(combined_meta), np.vstack(combined["embedding"].to_numpy()).astype("float32")])
    datasets["combined_cls_baseline_svd_logreg"] = (combined_meta, combined_x, lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.random_state), True)
    if config.model_subset is not None:
        requested = set(config.model_subset)
        datasets = {name: payload for name, payload in datasets.items() if name in requested}
        missing = sorted(requested.difference(datasets))
        if missing:
            raise ValueError(f"Requested models are unavailable in this run: {missing}")

    rows = []
    for target in config.targets:
        for model_name, (frame, x, estimator, use_embedding_preprocess) in datasets.items():
            rows.extend(_eval_model(
                frame.reset_index(drop=True),
                x,
                target,
                model_name,
                estimator,
                embedding_preprocess_mode=config.embedding_preprocess_mode,
                embedding_pca_components=config.embedding_pca_components,
                use_embedding_preprocess=use_embedding_preprocess,
            ))

    if not rows:
        raise ValueError(
            "No downstream evaluation rows were produced. Check that the selected data contains train, valid/test, "
            "available labels, and at least two train classes for each target."
        )
    output = pd.DataFrame(rows).sort_values(["target", "prefix_len", "model_name", "split"])
    (root / config.outputs["metrics"]).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(root / config.outputs["metrics"], index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(output),
        "targets": list(config.targets),
        "dry_run_rows": config.dry_run_rows,
        "feature_columns_clean": feature_cols_clean,
        "feature_columns_legacy": feature_cols_legacy,
        "include_legacy_proxy_feature_models": config.include_legacy_proxy_feature_models,
        "embedding_preprocess_mode": config.embedding_preprocess_mode,
        "embedding_pca_components": config.embedding_pca_components,
        "include_mlp_probe_models": config.include_mlp_probe_models,
        "mlp_hidden_dim": config.mlp_hidden_dim,
        "mlp_max_iter": config.mlp_max_iter,
        "outputs": config.outputs,
    }
    atomic_write_json(manifest, root / config.outputs["manifest"])
    return manifest
