from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts" / "reports"
OUT = ROOT / "report_assets" / "figures"


TARGET_ORDER = ["retention_7d", "retention_14d", "retention_30d"]
TARGET_LABELS = {
    "retention_7d": "7d",
    "retention_14d": "14d",
    "retention_30d": "30d",
}
MODEL_LABELS = {
    "main_mean_logreg": "BERT mean",
    "main_cls_logreg": "BERT CLS",
    "main_readout_logreg": "BERT readout",
    "baseline_svd_logreg": "SVD baseline",
    "combined_cls_baseline_svd_logreg": "BERT CLS + SVD",
    "baseline_features_hgb_clean": "Tabular baseline",
}
ALIGNED_COMPARISON = [
    {"Horizon": "7d", "Model": "SimCLR", "ROC-AUC": 0.9249, "PR-AUC": 0.6263, "Test users": 10260, "Positive rate": 0.111},
    {"Horizon": "7d", "Model": "BERT mean", "ROC-AUC": 0.9296, "PR-AUC": 0.6228, "Test users": 10260, "Positive rate": 0.111},
    {"Horizon": "7d", "Model": "RNN/GRU aligned", "ROC-AUC": 0.9413, "PR-AUC": 0.6777, "Test users": 10260, "Positive rate": 0.111},
    {"Horizon": "14d", "Model": "SimCLR", "ROC-AUC": 0.9189, "PR-AUC": 0.4974, "Test users": 10260, "Positive rate": 0.060},
    {"Horizon": "14d", "Model": "BERT mean", "ROC-AUC": 0.9256, "PR-AUC": 0.4467, "Test users": 10260, "Positive rate": 0.060},
    {"Horizon": "14d", "Model": "RNN/GRU aligned", "ROC-AUC": 0.9319, "PR-AUC": 0.4960, "Test users": 10260, "Positive rate": 0.060},
    {"Horizon": "30d", "Model": "SimCLR", "ROC-AUC": 0.8621, "PR-AUC": 0.1725, "Test users": 10260, "Positive rate": 0.0034},
    {"Horizon": "30d", "Model": "BERT mean", "ROC-AUC": 0.9665, "PR-AUC": 0.1500, "Test users": 10260, "Positive rate": 0.0034},
    {"Horizon": "30d", "Model": "RNN/GRU aligned", "ROC-AUC": 0.9658, "PR-AUC": 0.1910, "Test users": 10260, "Positive rate": 0.0034},
]
ALIGNED_MODEL_COLORS = {
    "SimCLR": "#8B5CF6",
    "BERT mean": "#2F6FED",
    "RNN/GRU aligned": "#10B981",
}
COLORS = {
    "roc": "#2F6FED",
    "pr": "#F28E2B",
    "mean": "#2F6FED",
    "cls": "#6C5CE7",
    "readout": "#00A8A8",
    "before": "#9AA4B2",
    "after": "#2F6FED",
    "baseline": "#6B7280",
    "combined": "#10B981",
    "tabular": "#F59E0B",
}


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#111827",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
    })


def target_sort_key(target: str) -> int:
    return TARGET_ORDER.index(target)


def load_metrics() -> pd.DataFrame:
    path = REPORTS / "bert_full_history_last512_full_duration_downstream_metrics.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["split"].eq("test")].copy()
    frame["target_label"] = frame["target"].map(TARGET_LABELS)
    frame["model_label"] = frame["model_name"].map(MODEL_LABELS).fillna(frame["model_name"])
    return frame


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGET_ORDER:
        subset = metrics[
            metrics["target"].eq(target)
            & metrics["model_name"].isin(["main_mean_logreg", "main_cls_logreg", "main_readout_logreg"])
        ].sort_values("pr_auc", ascending=False)
        best = subset.iloc[0]
        rows.append({
            "Target": TARGET_LABELS[target],
            "BERT variant": MODEL_LABELS[best["model_name"]],
            "Rows": int(best["num_rows"]),
            "Positive rate": best["positive_rate"],
            "ROC-AUC": best["roc_auc"],
            "PR-AUC": best["pr_auc"],
        })
    return pd.DataFrame(rows)


def write_summary_tables(metrics: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main = main_table(metrics)
    main.to_csv(OUT / "bert_main_metrics.csv", index=False)
    aligned = pd.DataFrame(ALIGNED_COMPARISON)
    aligned.to_csv(OUT / "aligned_model_comparison.csv", index=False)
    all_bert = metrics[metrics["model_name"].isin(["main_mean_logreg", "main_cls_logreg", "main_readout_logreg"])].copy()
    all_bert = all_bert[["target", "model_label", "num_rows", "positive_rate", "roc_auc", "pr_auc", "f1", "logloss"]]
    all_bert.to_csv(OUT / "bert_all_pooling_metrics.csv", index=False)

    lines = [
        "# Report Figures Index",
        "",
        "Generated from `artifacts/reports/bert_full_history_last512_full_duration_downstream_metrics.csv`.",
        "",
        "## Main BERT Metrics",
        "",
        "| Target | BERT variant | Rows | Positive rate | ROC-AUC | PR-AUC |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in main.itertuples(index=False):
        lines.append(
            f"| {row.Target} | **{row._1}** | {row.Rows:,}".replace(",", " ")
            + f" | {row._3:.4f} | **{row._4:.4f}** | **{row._5:.4f}** |"
        )
    lines.extend([
        "",
        "## Figures",
        "",
        "- `01_bert_main_metrics_table.png/pdf`: compact table with the best BERT variant for 7d, 14d, 30d.",
        "- `02_bert_mean_roc_pr.png/pdf`: ROC-AUC and PR-AUC for BERT mean pooling.",
        "- `03_bert_pooling_comparison.png/pdf`: CLS, mean pooling, and readout comparison.",
        "- `04_positive_rate_fix.png/pdf`: old window-based positive rate vs corrected full-user positive rate.",
        "- `05_bert_vs_context_models.png/pdf`: BERT mean compared with SVD, combined, and tabular baselines.",
        "- `06_aligned_models_roc_pr.png/pdf`: SimCLR, BERT mean, and RNN/GRU aligned comparison.",
        "- `07_aligned_models_table.png/pdf`: compact table for SimCLR, BERT mean, and RNN/GRU aligned.",
    ])
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def plot_main_table(metrics: pd.DataFrame) -> None:
    table = main_table(metrics)
    display = table.copy()
    display["Positive rate"] = display["Positive rate"].map(lambda value: f"{value:.4f}")
    display["ROC-AUC"] = display["ROC-AUC"].map(lambda value: f"{value:.4f}")
    display["PR-AUC"] = display["PR-AUC"].map(lambda value: f"{value:.4f}")
    display["Rows"] = display["Rows"].map(lambda value: f"{value:,}".replace(",", " "))

    fig, ax = plt.subplots(figsize=(11.2, 2.65))
    ax.axis("off")
    ax.set_title("BERT full-history last-512: main test metrics", loc="left", pad=16, fontsize=16)
    tbl = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.12, 0.22, 0.12, 0.18, 0.16, 0.16],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 1.55)
    for (row, _col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#111827")
        else:
            cell.set_facecolor("#F9FAFB" if row % 2 else "white")
            if _col in {4, 5}:
                cell.set_text_props(weight="bold")
    fig.text(
        0.01,
        0.02,
        "Labels corrected to full user lifetime: last_event_ts - first_event_ts.",
        fontsize=9,
        color="#4B5563",
    )
    save(fig, "01_bert_main_metrics_table")


def plot_bert_mean(metrics: pd.DataFrame) -> None:
    df = metrics[metrics["model_name"].eq("main_mean_logreg")].copy()
    df = df.sort_values("target", key=lambda s: s.map(target_sort_key))
    x = np.arange(len(df))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, df["roc_auc"], width, label="ROC-AUC", color=COLORS["roc"])
    ax.bar(x + width / 2, df["pr_auc"], width, label="PR-AUC", color=COLORS["pr"])
    ax.set_title("BERT mean pooling: retention prediction on test")
    ax.set_ylabel("Score")
    ax.set_xticks(x, df["target_label"])
    ax.set_ylim(0, 1.04)
    ax.grid(axis="y")
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    for xpos, value in zip(x - width / 2, df["roc_auc"], strict=True):
        ax.text(xpos, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    for xpos, value in zip(x + width / 2, df["pr_auc"], strict=True):
        ax.text(xpos, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    save(fig, "02_bert_mean_roc_pr")


def plot_pooling_comparison(metrics: pd.DataFrame) -> None:
    models = ["main_cls_logreg", "main_mean_logreg", "main_readout_logreg"]
    labels = ["CLS", "Mean", "Readout"]
    colors = [COLORS["cls"], COLORS["mean"], COLORS["readout"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    for ax, metric, title in zip(axes, ["roc_auc", "pr_auc"], ["ROC-AUC", "PR-AUC"], strict=True):
        x = np.arange(len(TARGET_ORDER))
        width = 0.25
        for offset, model, label, color in zip([-width, 0, width], models, labels, colors, strict=True):
            values = []
            for target in TARGET_ORDER:
                row = metrics[metrics["target"].eq(target) & metrics["model_name"].eq(model)].iloc[0]
                values.append(row[metric])
            ax.bar(x + offset, values, width, label=label, color=color)
        ax.set_title(title)
        ax.set_xticks(x, [TARGET_LABELS[t] for t in TARGET_ORDER])
        ax.set_ylim(0, 1.04)
        ax.grid(axis="y")
    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("BERT embedding extraction variants on test", fontsize=15, weight="bold", y=1.02)
    save(fig, "03_bert_pooling_comparison")


def plot_positive_rate_fix(metrics: pd.DataFrame) -> None:
    old_rates = {
        "retention_7d": 0.05038986354775828,
        "retention_14d": 0.0195906432748538,
        "retention_30d": 0.00009746588693957115,
    }
    corrected = metrics[metrics["model_name"].eq("main_mean_logreg")].set_index("target")
    targets = TARGET_ORDER
    x = np.arange(len(targets))
    width = 0.34
    before = [old_rates[t] for t in targets]
    after = [float(corrected.loc[t, "positive_rate"]) for t in targets]

    fig, ax = plt.subplots(figsize=(8.7, 4.8))
    ax.bar(x - width / 2, before, width, color=COLORS["before"], label="Before: last-512 window")
    ax.bar(x + width / 2, after, width, color=COLORS["after"], label="After: full user lifetime")
    ax.set_title("Positive rate correction on test")
    ax.set_ylabel("Positive rate")
    ax.set_xticks(x, [TARGET_LABELS[t] for t in targets])
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    for xpos, value in zip(x - width / 2, before, strict=True):
        ax.text(xpos, value + 0.003, f"{value:.4f}", ha="center", fontsize=9)
    for xpos, value in zip(x + width / 2, after, strict=True):
        ax.text(xpos, value + 0.003, f"{value:.4f}", ha="center", fontsize=9, weight="bold")
    fig.text(
        0.01,
        0.01,
        "Correct definition: last_event_ts - first_event_ts >= N days.",
        fontsize=9,
        color="#4B5563",
    )
    save(fig, "04_positive_rate_fix")


def plot_context_models(metrics: pd.DataFrame) -> None:
    models = [
        "main_mean_logreg",
        "baseline_svd_logreg",
        "combined_cls_baseline_svd_logreg",
        "baseline_features_hgb_clean",
    ]
    palette = [COLORS["mean"], COLORS["baseline"], COLORS["combined"], COLORS["tabular"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), sharey=True)
    for ax, metric, title in zip(axes, ["roc_auc", "pr_auc"], ["ROC-AUC", "PR-AUC"], strict=True):
        x = np.arange(len(TARGET_ORDER))
        width = 0.19
        offsets = np.linspace(-1.5 * width, 1.5 * width, len(models))
        for offset, model, color in zip(offsets, models, palette, strict=True):
            values = []
            for target in TARGET_ORDER:
                row = metrics[metrics["target"].eq(target) & metrics["model_name"].eq(model)].iloc[0]
                values.append(row[metric])
            ax.bar(x + offset, values, width, label=MODEL_LABELS[model], color=color)
        ax.set_title(title)
        ax.set_xticks(x, [TARGET_LABELS[t] for t in TARGET_ORDER])
        ax.set_ylim(0, 1.04)
        ax.grid(axis="y")
    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=9)
    fig.suptitle("BERT in context: embeddings and baseline models on test", fontsize=15, weight="bold", y=1.02)
    save(fig, "05_bert_vs_context_models")


def plot_aligned_models() -> None:
    df = pd.DataFrame(ALIGNED_COMPARISON)
    horizons = ["7d", "14d", "30d"]
    models = ["SimCLR", "BERT mean", "RNN/GRU aligned"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, metric in zip(axes, ["ROC-AUC", "PR-AUC"], strict=True):
        x = np.arange(len(horizons))
        width = 0.24
        for offset, model in zip([-width, 0, width], models, strict=True):
            values = [
                float(df[df["Horizon"].eq(horizon) & df["Model"].eq(model)][metric].iloc[0])
                for horizon in horizons
            ]
            ax.bar(
                x + offset,
                values,
                width,
                label=model,
                color=ALIGNED_MODEL_COLORS[model],
            )
        ax.set_title(metric)
        ax.set_xticks(x, horizons)
        ax.set_ylim(0, 1.04)
        ax.grid(axis="y")
    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=9)
    fig.suptitle("Aligned full-history comparison on the same test users", fontsize=15, weight="bold", y=1.02)
    fig.text(
        0.01,
        0.01,
        "All rows use 10 260 test users and corrected full-user duration labels.",
        fontsize=9,
        color="#4B5563",
    )
    save(fig, "06_aligned_models_roc_pr")


def plot_aligned_table() -> None:
    df = pd.DataFrame(ALIGNED_COMPARISON)
    display = df.copy()
    display["ROC-AUC"] = display["ROC-AUC"].map(lambda value: f"{value:.4f}")
    display["PR-AUC"] = display["PR-AUC"].map(lambda value: f"{value:.4f}")
    display["Test users"] = display["Test users"].map(lambda value: f"{value:,}".replace(",", " "))
    display["Positive rate"] = display["Positive rate"].map(lambda value: f"{value * 100:.2f}%")

    fig, ax = plt.subplots(figsize=(11.7, 4.3))
    ax.axis("off")
    ax.set_title("Aligned model comparison on test", loc="left", pad=16, fontsize=16)
    tbl = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.12, 0.24, 0.14, 0.14, 0.17, 0.17],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.3)
    tbl.scale(1, 1.45)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#111827")
        else:
            model = display.iloc[row - 1]["Model"]
            cell.set_facecolor("#F9FAFB" if row % 2 else "white")
            if model == "BERT mean":
                cell.set_facecolor("#EFF6FF")
            if col in {2, 3}:
                cell.set_text_props(weight="bold")
    fig.text(
        0.01,
        0.02,
        "BERT numbers are from the corrected full-duration downstream evaluation.",
        fontsize=9,
        color="#4B5563",
    )
    save(fig, "07_aligned_models_table")


def main() -> None:
    setup_style()
    metrics = load_metrics()
    write_summary_tables(metrics)
    plot_main_table(metrics)
    plot_bert_mean(metrics)
    plot_pooling_comparison(metrics)
    plot_positive_rate_fix(metrics)
    plot_context_models(metrics)
    plot_aligned_models()
    plot_aligned_table()
    print(f"Saved figures to {OUT}")


if __name__ == "__main__":
    main()
