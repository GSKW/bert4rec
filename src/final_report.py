from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.io_utils import atomic_write_json, load_yaml


@dataclass(frozen=True)
class FinalReportConfig:
    sequence_metrics_path: str
    downstream_metrics_path: str
    output_xlsx: str
    output_dir: str
    manifest_path: str


def load_final_report_config(path: str | Path) -> FinalReportConfig:
    raw = load_yaml(path)
    return FinalReportConfig(**raw)


def build_final_report(config: FinalReportConfig, project_root: str | Path = ".") -> dict[str, Any]:
    root = Path(project_root)
    mpl_config_dir = root / ".cache/matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib.pyplot as plt

    output_dir = root / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seq = pd.read_csv(root / config.sequence_metrics_path)
    down = pd.read_csv(root / config.downstream_metrics_path)

    best_down_oracle = (
        down.sort_values(["target", "prefix_len", "split", "roc_auc"], ascending=[True, True, True, False])
        .groupby(["target", "prefix_len", "split"], as_index=False)
        .head(1)
    )
    best_down_valid = (
        down[down["split"] == "valid"]
        .sort_values(["target", "prefix_len", "roc_auc"], ascending=[True, True, False])
        .groupby(["target", "prefix_len"], as_index=False)
        .head(1)
    )
    valid_winner_keys = best_down_valid[["target", "prefix_len", "model_name"]].drop_duplicates()
    best_down_test_of_valid = down[down["split"] == "test"].merge(
        valid_winner_keys,
        on=["target", "prefix_len", "model_name"],
        how="inner",
    )
    best_seq = (
        seq.sort_values(["split", "prefix_len", "mrr_at_10"], ascending=[True, True, False])
        .groupby(["split", "prefix_len"], as_index=False)
        .head(1)
    )

    xlsx_path = root / config.output_xlsx
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(xlsx_path) as writer:
        seq.to_excel(writer, sheet_name="sequence_metrics", index=False)
        down.to_excel(writer, sheet_name="downstream_metrics", index=False)
        best_seq.to_excel(writer, sheet_name="best_sequence", index=False)
        best_down_test_of_valid.to_excel(writer, sheet_name="best_downstream", index=False)
        best_down_valid.to_excel(writer, sheet_name="best_downstream_valid", index=False)
        best_down_test_of_valid.to_excel(writer, sheet_name="best_downstream_test_of_valid", index=False)
        best_down_oracle.to_excel(writer, sheet_name="best_downstream_oracle", index=False)

    if not down.empty:
        for target in sorted(down["target"].unique()):
            subset = down[(down["target"] == target) & (down["split"] == "test")]
            if subset.empty:
                continue
            pivot = subset.pivot_table(index="prefix_len", columns="model_name", values="roc_auc", aggfunc="mean")
            ax = pivot.plot(kind="bar", figsize=(12, 5), title=f"Test ROC-AUC: {target}")
            ax.set_ylabel("ROC-AUC")
            plt.tight_layout()
            plt.savefig(output_dir / f"downstream_{target}_test_roc_auc.png")
            plt.close()

    if not seq.empty:
        subset = seq[seq["split"] == "test"]
        pivot = subset.pivot_table(index="prefix_len", columns="model_name", values="mrr_at_10", aggfunc="mean")
        ax = pivot.plot(kind="bar", figsize=(10, 5), title="Test MRR@10")
        ax.set_ylabel("MRR@10")
        plt.tight_layout()
        plt.savefig(output_dir / "sequence_test_mrr_at_10.png")
        plt.close()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sequence_metrics_path": config.sequence_metrics_path,
        "downstream_metrics_path": config.downstream_metrics_path,
        "output_xlsx": config.output_xlsx,
        "output_dir": config.output_dir,
        "sequence_rows": len(seq),
        "downstream_rows": len(down),
        "best_downstream_valid_rows": len(best_down_valid),
        "best_downstream_test_of_valid_rows": len(best_down_test_of_valid),
    }
    atomic_write_json(manifest, root / config.manifest_path)
    return manifest
