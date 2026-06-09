from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from evalkit.models import GroupSummary, ReportModel

logger = logging.getLogger("graphrag")


def _import_mpl() -> Any:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _bar_with_errorbars(
    plt: Any,
    labels: list[str],
    means: list[float],
    ci_lowers: list[float],
    ci_uppers: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4))
    x = range(len(labels))
    yerr_lower = [m - lo for m, lo in zip(means, ci_lowers)]
    yerr_upper = [hi - m for m, hi in zip(means, ci_uppers)]
    ax.bar(x, means, yerr=[yerr_lower, yerr_upper], capsize=4, color="steelblue", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=120)
    plt.close(fig)


def _line_trend(
    plt: Any,
    trend: list[dict[str, Any]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    x_labels = [entry["run_dir"][-20:] for entry in trend]
    values = [entry["value"] for entry in trend]
    fig, ax = plt.subplots(figsize=(max(6, len(x_labels) * 0.8), 4))
    ax.plot(range(len(values)), values, marker="o", color="steelblue")
    ax.set_xticks(list(range(len(x_labels))))
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=120)
    plt.close(fig)


def _heatmap(
    plt: Any,
    groups: list[GroupSummary],
    metric_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    import numpy as np  # type: ignore

    global_groups = [g for g in groups if g.keys.get("segment") == "global"]
    if not global_groups:
        return

    row_labels = [
        f"{g.keys.get('strategy', '')} ({g.keys.get('model_id', '')})"
        for g in global_groups
    ]
    data = []
    for g in global_groups:
        row = [g.metrics.get(m, {}).get("mean", float("nan")) for m in metric_names]
        data.append(row)

    data_array = np.array(data)
    fig, ax = plt.subplots(
        figsize=(max(6, len(metric_names) * 1.2), max(3, len(row_labels) * 0.7))
    )
    im = ax.imshow(data_array, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(list(range(len(metric_names))))
    ax.set_xticklabels(
        [m.replace("_at_k", "@k").replace("_", "\n") for m in metric_names],
        fontsize=8,
    )
    ax.set_yticks(list(range(len(row_labels))))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax)

    for i in range(len(row_labels)):
        for j in range(len(metric_names)):
            val = data_array[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=120)
    plt.close(fig)


RETRIEVAL_PLOT_METRICS = [
    "entity_coverage", "precision_at_k", "recall_at_k", "ndcg_at_k", "mrr", "map"
]


def generate_plots(report: ReportModel, plots_dir: Path) -> list[Path]:
    """Generate all plots for a ReportModel and save to plots_dir.

    Returns list of generated file paths. Silently skips if matplotlib unavailable.
    """
    plt = _import_mpl()
    if plt is None:
        logger.warning("matplotlib not installed; skipping plots")
        return []

    generated: list[Path] = []

    global_groups = [g for g in report.groups if g.keys.get("segment") == "global"]

    # ── Bar chart: retrieval metrics per strategy ─────────────────────────
    for metric in ["precision_at_k", "recall_at_k", "mrr", "map", "entity_coverage"]:
        data_points = [
            (
                f"{g.keys.get('strategy', '')}",
                g.metrics.get(metric, {}).get("mean", 0.0),
                g.metrics.get(metric, {}).get("ci_lower", 0.0),
                g.metrics.get(metric, {}).get("ci_upper", 0.0),
            )
            for g in global_groups
            if g.metrics.get(metric, {}).get("n", 0) > 0
        ]
        if not data_points:
            continue
        labels, means, ci_l, ci_u = zip(*data_points)
        out_path = plots_dir / f"retrieval_{metric}.png"
        _bar_with_errorbars(
            plt,
            list(labels), list(means), list(ci_l), list(ci_u),
            title=f"{metric.replace('_', ' ').title()} by Strategy",
            ylabel=metric,
            output_path=out_path,
        )
        generated.append(out_path)

    # ── Heatmap: strategy × metric ────────────────────────────────────────
    available_metrics = [
        m for m in RETRIEVAL_PLOT_METRICS
        if any(g.metrics.get(m, {}).get("n", 0) > 0 for g in global_groups)
    ]
    if available_metrics and len(global_groups) > 1:
        out_path = plots_dir / "heatmap_strategies_vs_metrics.png"
        _heatmap(plt, report.groups, available_metrics, "Strategy × Metric", out_path)
        generated.append(out_path)

    # ── Latency trend (project scope) ─────────────────────────────────────
    latency_trend = report.meta.get("latency_trend", [])
    if latency_trend and len(latency_trend) > 1:
        out_path = plots_dir / "trend_latency.png"
        _line_trend(plt, latency_trend, "Latency Trend (avg)", "avg_latency_ms (ms)", out_path)
        generated.append(out_path)

    # ── Text metrics bar ──────────────────────────────────────────────────
    for metric in ["token_f1", "rouge_l", "bleu"]:
        data_points = [
            (
                g.keys.get("strategy", ""),
                g.metrics.get(metric, {}).get("mean", 0.0),
                g.metrics.get(metric, {}).get("ci_lower", 0.0),
                g.metrics.get(metric, {}).get("ci_upper", 0.0),
            )
            for g in global_groups
            if g.metrics.get(metric, {}).get("n", 0) > 0
        ]
        if not data_points:
            continue
        labels, means, ci_l, ci_u = zip(*data_points)
        out_path = plots_dir / f"text_{metric}.png"
        _bar_with_errorbars(
            plt,
            list(labels), list(means), list(ci_l), list(ci_u),
            title=f"{metric.upper()} by Strategy",
            ylabel=metric,
            output_path=out_path,
        )
        generated.append(out_path)

    logger.info("Generated %d plots in %s", len(generated), plots_dir)
    return generated
