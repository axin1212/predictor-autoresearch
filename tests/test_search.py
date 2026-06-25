from __future__ import annotations

import numpy as np

from predictor_autoresearch.holdout import HoldoutInterval, HoldoutPlan
from predictor_autoresearch.model_runner import CandidateConfig, HoldoutRunResult
from predictor_autoresearch.search import SearchConfig, _initial_candidates, run_search


def _holdouts() -> HoldoutPlan:
    return HoldoutPlan(
        intervals=[
            HoldoutInterval("h1", np.datetime64("2026-01-01"), np.datetime64("2026-01-02"), [1]),
            HoldoutInterval("h2", np.datetime64("2026-01-03"), np.datetime64("2026-01-04"), [2]),
            HoldoutInterval("h3", np.datetime64("2026-01-05"), np.datetime64("2026-01-06"), [3]),
        ],
        confidence="high",
    )


def _result(config: CandidateConfig, holdout: HoldoutInterval, improvement: float = 50.0) -> HoldoutRunResult:
    return HoldoutRunResult(
        candidate_id=config.candidate_id,
        holdout_name=holdout.name,
        status="ok",
        actual=np.array([1.0, 2.0]),
        predictions=np.array([1.0, 2.0]),
        r2=0.5,
        rmse=0.5,
        naive_rmse=1.0,
        rmse_improvement_pct=improvement,
        horizon_step=config.horizon_step,
        selected_features=["f0"],
    )


def test_run_search_baseline_all_holdouts_and_quick_screen(tmp_path):
    calls: list[tuple[str, str]] = []

    def fake_runner(config: CandidateConfig, holdout: HoldoutInterval) -> HoldoutRunResult:
        calls.append((config.candidate_id, holdout.name))
        improvement = {
            ("baseline_h+0", "h1"): 40.0,
            ("baseline_h+0", "h2"): -20.0,
            ("baseline_h+0", "h3"): 30.0,
        }.get((config.candidate_id, holdout.name), 0.5)
        return _result(config, holdout, improvement)

    state = run_search(
        _holdouts(),
        SearchConfig(time_budget_seconds=1, report_path=tmp_path / "report.html"),
        fake_runner,
    )

    assert ("baseline_h+0", "h1") in calls
    assert ("baseline_h+0", "h2") in calls
    assert ("baseline_h+0", "h3") in calls
    assert ("baseline_h+1", "h1") in calls
    assert calls[0][0] == "baseline_h+0"
    quick_screen_calls = [call for call in calls if not call[0].startswith("baseline")]
    assert quick_screen_calls
    assert quick_screen_calls[0][1] == "h2"
    assert quick_screen_calls[0][0] == "identity_recent_h+0"
    assert state.candidates[0].score >= state.candidates[-1].score
    assert (tmp_path / "report.html").exists()


def test_initial_candidates_keep_fixed_context_sample_count():
    candidates = _initial_candidates(SearchConfig(time_budget_seconds=1, report_path="report.html"))
    sample_counts = {candidate.num_train_samples for candidate in candidates}

    assert sample_counts == {400}


def test_initial_candidates_are_future_prediction_candidates():
    candidates = _initial_candidates(SearchConfig(time_budget_seconds=1, report_path="report.html"))
    ids = [candidate.candidate_id for candidate in candidates]

    assert ids == [
        "identity_recent_h+0",
        "identity_coverage_h+0",
        "trend_default_h+0",
        "window_short_h+0",
        "window_long_h+0",
        "coverage_h+0",
        "identity_recent_h+1",
        "identity_coverage_h+1",
        "trend_default_h+1",
        "window_short_h+1",
        "window_long_h+1",
        "coverage_h+1",
    ]


def test_multi_horizon_search_runs_all_baselines_before_candidates(tmp_path):
    calls: list[tuple[str, str]] = []

    def fake_runner(config: CandidateConfig, holdout: HoldoutInterval) -> HoldoutRunResult:
        calls.append((config.candidate_id, holdout.name))
        return _result(config, holdout)

    run_search(
        _holdouts(),
        SearchConfig(
            time_budget_seconds=1,
            report_path=tmp_path / "report.html",
            prediction_horizons=(3, 5),
        ),
        fake_runner,
    )

    baseline_call_ids = [candidate_id for candidate_id, _holdout_name in calls[:9]]
    assert baseline_call_ids == [
        "baseline_h+0",
        "baseline_h+0",
        "baseline_h+0",
        "baseline_h+3",
        "baseline_h+3",
        "baseline_h+3",
        "baseline_h+5",
        "baseline_h+5",
        "baseline_h+5",
    ]


def test_low_risk_context_candidates_keep_identity_features():
    candidates = _initial_candidates(SearchConfig(time_budget_seconds=1, report_path="report.html"))
    by_id = {candidate.candidate_id: candidate for candidate in candidates}

    assert by_id["identity_recent_h+0"].feature_mode == "identity"
    assert by_id["identity_recent_h+0"].context_policy == "recent"
    assert by_id["identity_coverage_h+0"].feature_mode == "identity"
    assert by_id["identity_coverage_h+0"].context_policy == "coverage"


def test_frequency_candidate_is_opt_in():
    default_ids = [candidate.candidate_id for candidate in _initial_candidates(SearchConfig(1, "report.html"))]
    opt_in_ids = [
        candidate.candidate_id
        for candidate in _initial_candidates(SearchConfig(1, "report.html", include_frequency_candidate=True))
    ]

    assert "frequency_h+0" not in default_ids
    assert "frequency_h+0" in opt_in_ids
    assert "frequency_h+1" in opt_in_ids


def test_zero_time_budget_runs_full_candidate_list(tmp_path):
    calls: list[tuple[str, str]] = []

    def fake_runner(config: CandidateConfig, holdout: HoldoutInterval) -> HoldoutRunResult:
        calls.append((config.candidate_id, holdout.name))
        return _result(config, holdout)

    run_search(
        _holdouts(),
        SearchConfig(time_budget_seconds=0, report_path=tmp_path / "report.html"),
        fake_runner,
    )

    candidate_ids = {call[0] for call in calls}
    assert {candidate.candidate_id for candidate in _initial_candidates(SearchConfig(0, tmp_path / "report.html"))}.issubset(candidate_ids)
