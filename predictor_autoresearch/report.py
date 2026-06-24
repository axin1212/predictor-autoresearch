from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import html

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from predictor_autoresearch.model_runner import HoldoutRunResult


@dataclass(frozen=True)
class CandidateReport:
    candidate_id: str
    score: float
    status: str
    holdouts: list[HoldoutRunResult] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class ReportState:
    candidates: list[CandidateReport]
    metadata: dict[str, str] = field(default_factory=dict)


def write_report(path: Path, state: ReportState, top_n: int = 20) -> Path:
    ranked = sorted(state.candidates, key=lambda c: c.score, reverse=True)
    plot_candidates = [candidate for candidate in ranked if candidate.holdouts][:top_n]
    holdout_count = max((len(candidate.holdouts) for candidate in plot_candidates), default=1)
    rows = max(1, len(plot_candidates))
    fig = make_subplots(
        rows=rows,
        cols=holdout_count,
        subplot_titles=_subplot_titles(plot_candidates, holdout_count),
        vertical_spacing=min(0.10, 0.65 / max(rows - 1, 1)),
        horizontal_spacing=0.08,
    )

    for row, candidate in enumerate(plot_candidates, start=1):
        for col, holdout in enumerate(candidate.holdouts, start=1):
            actual = np.asarray(holdout.actual, dtype=float)
            predicted = np.asarray(holdout.predictions, dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=actual,
                    y=predicted,
                    mode="markers",
                    name=f"{candidate.candidate_id}:{holdout.holdout_name}",
                    hovertemplate="actual=%{x}<br>predicted=%{y}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            finite = np.concatenate([actual[np.isfinite(actual)], predicted[np.isfinite(predicted)]])
            if len(finite):
                low = float(np.min(finite))
                high = float(np.max(finite))
                fig.add_trace(
                    go.Scatter(
                        x=[low, high],
                        y=[low, high],
                        mode="lines",
                        name="45-degree reference",
                        line={"dash": "dash", "color": "#666"},
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )
            fig.update_xaxes(title_text="Actual", row=row, col=col)
            fig.update_yaxes(title_text="Predicted", row=row, col=col)

    fig.update_annotations(font_size=11)
    fig.update_layout(
        height=max(420, rows * 390),
        margin={"t": 90, "b": 70, "l": 70, "r": 35},
        showlegend=False,
        title_text="Predictor AutoResearch",
    )
    horizon_body = _horizon_summary_chart(ranked)
    body = fig.to_html(full_html=False, include_plotlyjs=False if horizon_body else "cdn")
    index = _candidate_index(ranked)
    metadata = _metadata_block(state.metadata)
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Predictor AutoResearch</title>"
        f"<style>{_report_css()}</style></head><body><h1>Predictor AutoResearch</h1>"
        "<p>Future prediction search ranked by RMSE improvement versus a train-mean constant baseline. "
        "R² is shown as an auxiliary fit metric.</p>"
        f"{metadata}{horizon_body}{index}{body}</body></html>",
        encoding="utf-8",
    )
    return path


def _candidate_index(candidates: list[CandidateReport]) -> str:
    rows = []
    for rank, candidate in enumerate(candidates, start=1):
        holdout_values = "".join(_holdout_detail(h) for h in candidate.holdouts)
        detail = holdout_values or html.escape(candidate.error or "")
        selected_features = _selected_feature_detail(candidate)
        horizon = _candidate_horizon_label(candidate)
        rows.append(
            "<tr>"
            f"<td>#{rank}</td>"
            f"<td>{html.escape(candidate.candidate_id)}</td>"
            f"<td>{html.escape(horizon)}</td>"
            f"<td>{_format_float(candidate.score, digits=2)}</td>"
            f"<td>{html.escape(candidate.status)}</td>"
            f"<td>{detail}</td>"
            f"<td>{selected_features}</td>"
            "</tr>"
        )
    return (
        "<table class='candidate-index'><thead><tr><th>Rank</th><th>Candidate</th><th>Horizon</th>"
        "<th>Mean RMSE Improvement (%)</th><th>Status</th><th>Metrics / Error</th><th>Selected Features</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _holdout_detail(holdout: HoldoutRunResult) -> str:
    name = html.escape(holdout.holdout_name)
    if holdout.error:
        return f"<div class='holdout-detail'><strong>{name}</strong>: error: {html.escape(holdout.error)}</div>"
    actual = np.asarray(holdout.actual, dtype=float)
    predicted = np.asarray(holdout.predictions, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not mask.any():
        return (
            f"<div class='holdout-detail'><strong>{name}</strong>: "
            f"h=t+{holdout.horizon_step}, n={len(holdout.actual)}, "
            "RMSE improvement=nan%, R²=nan, RMSE=nan, naive RMSE=nan, MAE=nan, y_std=nan</div>"
        )
    error = actual[mask] - predicted[mask]
    mae = float(np.mean(np.abs(error)))
    y_std = float(np.std(actual[mask], ddof=1)) if mask.sum() > 1 else float("nan")
    rmse_to_std = holdout.rmse / y_std if np.isfinite(y_std) and y_std > 0 else float("nan")
    return (
        f"<div class='holdout-detail'><strong>{name}</strong>: "
        f"h=t+{holdout.horizon_step}, n={len(holdout.actual)}, "
        f"RMSE improvement={_format_float(holdout.rmse_improvement_pct, digits=2)}%, "
        f"R²={_format_float(holdout.r2, digits=3)}, "
        f"RMSE={_format_float(holdout.rmse, digits=4)}, "
        f"naive RMSE={_format_float(holdout.naive_rmse, digits=4)}, "
        f"MAE={_format_float(mae, digits=4)}, y_std={_format_float(y_std, digits=4)}, "
        f"RMSE/std={_format_float(rmse_to_std, digits=2)}</div>"
    )


def _selected_feature_detail(candidate: CandidateReport) -> str:
    groups = []
    total_features = 0
    for holdout in candidate.holdouts:
        if not holdout.selected_features:
            continue
        total_features += len(holdout.selected_features)
        features = "".join(f"<li>{html.escape(feature)}</li>" for feature in holdout.selected_features)
        groups.append(
            "<div class='feature-group'>"
            f"<div class='feature-holdout'>{html.escape(holdout.holdout_name)} "
            f"({len(holdout.selected_features)})</div>"
            f"<ul>{features}</ul>"
            "</div>"
        )
    if not groups:
        return ""
    entry_label = "feature entry" if total_features == 1 else "feature entries"
    holdout_label = "holdout" if len(groups) == 1 else "holdouts"
    summary = f"{total_features} {entry_label} across {len(groups)} {holdout_label}"
    return f"<details class='feature-details'><summary>{html.escape(summary)}</summary>{''.join(groups)}</details>"


def _subplot_titles(candidates: list[CandidateReport], holdout_count: int) -> list[str]:
    titles = []
    for candidate in candidates:
        for index in range(holdout_count):
            if index >= len(candidate.holdouts):
                titles.append("")
                continue
            holdout = candidate.holdouts[index]
            titles.append(
                f"{_short_label(candidate.candidate_id, 14)} / t+{holdout.horizon_step} / "
                f"{_short_label(holdout.holdout_name, 12)} / Imp={_format_float(holdout.rmse_improvement_pct, digits=1)}%"
            )
    return titles


def _horizon_summary_chart(candidates: list[CandidateReport]) -> str:
    best_by_horizon: dict[int, tuple[float, str]] = {}
    for candidate in candidates:
        horizon = _candidate_horizon(candidate)
        if horizon is None:
            continue
        score = _mean_improvement(candidate)
        if not np.isfinite(score):
            continue
        if horizon not in best_by_horizon or score > best_by_horizon[horizon][0]:
            best_by_horizon[horizon] = (score, candidate.candidate_id)
    if not best_by_horizon:
        return ""
    horizons = sorted(best_by_horizon)
    scores = [best_by_horizon[horizon][0] for horizon in horizons]
    names = [best_by_horizon[horizon][1] for horizon in horizons]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=horizons,
            y=scores,
            mode="lines+markers",
            text=names,
            hovertemplate="h=t+%{x}<br>best improvement=%{y:.2f}%<br>%{text}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#7b8794")
    fig.update_layout(
        title_text="Best RMSE Improvement by Prediction Horizon",
        xaxis_title="Prediction horizon (steps)",
        yaxis_title="Best RMSE improvement (%)",
        height=360,
        margin={"t": 70, "b": 65, "l": 70, "r": 30},
        showlegend=False,
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _metadata_block(metadata: dict[str, str]) -> str:
    if not metadata:
        return ""
    items = "".join(
        f"<div class='metadata-item'><span>{html.escape(key)}</span><strong>{html.escape(value)}</strong></div>"
        for key, value in metadata.items()
    )
    return f"<section class='metadata-grid'>{items}</section>"


def _candidate_horizon_label(candidate: CandidateReport) -> str:
    horizon = _candidate_horizon(candidate)
    return f"t+{horizon}" if horizon is not None else ""


def _candidate_horizon(candidate: CandidateReport) -> int | None:
    horizons = {holdout.horizon_step for holdout in candidate.holdouts}
    if len(horizons) == 1:
        return next(iter(horizons))
    return None


def _mean_improvement(candidate: CandidateReport) -> float:
    values = [
        holdout.rmse_improvement_pct
        for holdout in candidate.holdouts
        if np.isfinite(holdout.rmse_improvement_pct)
    ]
    if not values:
        return float("nan")
    return float(np.mean(values))


def _format_float(value: float, digits: int) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _short_label(value: str, max_len: int = 18) -> str:
    if len(value) <= max_len:
        return value
    return f"{value[: max_len - 1]}…"


def _report_css() -> str:
    return """
body {
  color: #1f2933;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 24px;
}
.candidate-index {
  border-collapse: collapse;
  font-size: 13px;
  table-layout: fixed;
  width: 100%;
}
.candidate-index th,
.candidate-index td {
  border: 1px solid #d9e2ec;
  padding: 8px;
  text-align: left;
  vertical-align: top;
  word-break: break-word;
}
.candidate-index th {
  background: #f5f7fa;
  font-weight: 600;
}
.candidate-index th:nth-child(1),
.candidate-index td:nth-child(1) {
  width: 56px;
}
.candidate-index th:nth-child(2),
.candidate-index td:nth-child(2) {
  width: 150px;
}
.candidate-index th:nth-child(3),
.candidate-index td:nth-child(3) {
  width: 72px;
}
.candidate-index th:nth-child(4),
.candidate-index td:nth-child(4) {
  width: 120px;
}
.candidate-index th:nth-child(5),
.candidate-index td:nth-child(5) {
  width: 86px;
}
.metadata-grid {
  display: grid;
  gap: 8px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  margin: 18px 0 22px;
}
.metadata-item {
  border: 1px solid #d9e2ec;
  padding: 10px 12px;
}
.metadata-item span {
  color: #52606d;
  display: block;
  font-size: 12px;
  margin-bottom: 4px;
}
.metadata-item strong {
  font-size: 14px;
}
.holdout-detail + .holdout-detail {
  margin-top: 5px;
}
.feature-details summary {
  cursor: pointer;
  white-space: nowrap;
}
.feature-group {
  margin-top: 8px;
}
.feature-holdout {
  font-weight: 600;
  margin-bottom: 4px;
}
.feature-details ul {
  margin: 0 0 0 18px;
  padding: 0;
}
.feature-details li {
  line-height: 1.35;
  margin: 2px 0;
}
"""
