from __future__ import annotations

import numpy as np

from predictor_autoresearch.model_runner import HoldoutRunResult
from predictor_autoresearch.report import CandidateReport, ReportState, _subplot_titles, write_report


def _holdout(
    candidate_id: str = "c1",
    holdout_name: str = "h1",
    r2: float = 0.98,
    rmse: float = 0.1,
    naive_rmse: float = 0.2,
    improvement: float = 50.0,
    horizon: int = 3,
    actual=None,
    predictions=None,
    selected_features=None,
    status: str = "ok",
    error: str | None = None,
) -> HoldoutRunResult:
    return HoldoutRunResult(
        candidate_id=candidate_id,
        holdout_name=holdout_name,
        status=status,
        actual=np.array([1.0, 2.0]) if actual is None else np.asarray(actual),
        predictions=np.array([1.1, 1.9]) if predictions is None else np.asarray(predictions),
        r2=r2,
        rmse=rmse,
        naive_rmse=naive_rmse,
        rmse_improvement_pct=improvement,
        horizon_step=horizon,
        selected_features=["f0"] if selected_features is None else selected_features,
        error=error,
    )


def test_write_report_contains_core_elements(tmp_path):
    state = ReportState(
        metadata={"Target tag": "target", "Window size": "90 steps (~90 min)"},
        candidates=[
            CandidateReport(
                candidate_id="c1",
                score=0.42,
                status="complete",
                holdouts=[_holdout()],
            ),
            CandidateReport(candidate_id="bad", score=-999.0, status="failed", holdouts=[], error="boom"),
        ]
    )
    path = tmp_path / "report.html"

    write_report(path, state)

    html = path.read_text()
    assert "plotly" in html.lower()
    assert "Target tag" in html
    assert "target" in html
    assert "Best RMSE Improvement by Prediction Horizon" in html
    assert "Best R\\u00b2 by Prediction Horizon" in html
    assert "Mean RMSE Improvement (%)" in html
    assert "Mean R²" in html
    assert "0.980" in html
    assert "RMSE improvement=50.00%" in html
    assert "R²" in html
    assert "RMSE=0.1000" in html
    assert "naive RMSE=0.2000" in html
    assert "MAE=0.1000" in html
    assert "y_std=" in html
    assert "RMSE/std=" in html
    assert "45-degree" in html
    assert "#1" in html
    assert "Selected Features" in html
    assert "<details class='feature-details'>" in html
    assert "1 feature entry across 1 holdout" in html
    assert "<li>f0</li>" in html
    assert "f0" in html
    assert "boom" in html
    assert "c1 / h1 n=2 R²=0.980" not in html


def test_write_report_surfaces_holdout_errors(tmp_path):
    state = ReportState(
        candidates=[
            CandidateReport(
                candidate_id="c1",
                score=float("-inf"),
                status="complete",
                holdouts=[
                    _holdout(
                        status="error",
                        actual=[],
                        predictions=[],
                        r2=float("nan"),
                        rmse=float("nan"),
                        naive_rmse=float("nan"),
                        improvement=float("nan"),
                        selected_features=[],
                        error="TPT child process failed",
                    )
                ],
            )
        ]
    )
    path = tmp_path / "report.html"

    write_report(path, state)

    html = path.read_text()
    assert "<strong>h1</strong>: error: TPT child process failed" in html


def test_write_report_keeps_selected_features_collapsed_and_grouped(tmp_path):
    features = [f"feature_{index}" for index in range(35)]
    state = ReportState(
        candidates=[
            CandidateReport(
                candidate_id="very_long_candidate_name_that_needs_shortening",
                score=0.1,
                status="complete",
                holdouts=[
                    _holdout(
                        holdout_name="very_long_holdout_name_that_needs_shortening",
                        actual=np.array([1.0, 2.0, 3.0]),
                        predictions=np.array([1.1, 1.9, 3.1]),
                        r2=0.5,
                        rmse=0.1,
                        selected_features=features,
                    )
                ],
            )
        ]
    )
    path = tmp_path / "report.html"

    write_report(path, state)

    html = path.read_text()
    assert "35 feature entries across 1 holdout" in html
    assert "<div class='feature-holdout'>very_long_holdout_name_that_needs_shortening (35)</div>" in html
    assert "<li>feature_34</li>" in html
    assert "very_long_candidate_name_that_needs_shortening / very_long_holdout_name_that_needs_shortening" not in html


def test_subplot_titles_are_short_and_aligned_to_grid():
    candidates = [
        CandidateReport(
            candidate_id="very_long_candidate_name_that_needs_shortening",
                score=0.1,
                status="complete",
                holdouts=[
                    _holdout(
                        holdout_name="very_long_holdout_name_that_needs_shortening",
                        actual=np.array([1.0]),
                        predictions=np.array([1.0]),
                        r2=0.5,
                        rmse=0.0,
                        naive_rmse=1.0,
                        improvement=100.0,
                        horizon=3,
                        selected_features=[],
                    )
                ],
        ),
        CandidateReport(
            candidate_id="c2",
            score=0.0,
                status="partial",
                holdouts=[
                    _holdout(
                        candidate_id="c2",
                        holdout_name="h1",
                        actual=np.array([1.0]),
                        predictions=np.array([1.0]),
                        r2=0.1,
                        rmse=0.0,
                        naive_rmse=1.0,
                        improvement=10.0,
                        horizon=5,
                        selected_features=[],
                    ),
                    _holdout(
                        candidate_id="c2",
                        holdout_name="h2",
                        actual=np.array([1.0]),
                        predictions=np.array([1.0]),
                        r2=0.2,
                        rmse=0.0,
                        naive_rmse=1.0,
                        improvement=20.0,
                        horizon=5,
                        selected_features=[],
                    ),
                ],
        ),
    ]

    titles = _subplot_titles(candidates, holdout_count=2)

    assert titles == [
        "very_long_can… / t+3 / very_long_h… / Imp=100.0%",
        "",
        "c2 / t+5 / h1 / Imp=10.0%",
        "c2 / t+5 / h2 / Imp=20.0%",
    ]
