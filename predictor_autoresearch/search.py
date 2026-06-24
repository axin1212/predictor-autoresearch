from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import math
import time
import os

from predictor_autoresearch.holdout import HoldoutInterval, HoldoutPlan
from predictor_autoresearch.model_runner import CandidateConfig, HoldoutRunResult
from predictor_autoresearch.report import CandidateReport, ReportState, write_report
from predictor_autoresearch.scoring import candidate_score


@dataclass(frozen=True)
class SearchConfig:
    time_budget_seconds: float
    report_path: Path
    default_window_minutes: int = 60
    top_features_n: int = 32
    num_train_samples: int = 400
    random_state: int = 42
    include_frequency_candidate: bool = False
    prediction_horizons: tuple[int, ...] = (1,)
    metadata: dict[str, str] | None = None


CandidateRunner = Callable[[CandidateConfig, HoldoutInterval], HoldoutRunResult]


def run_search(
    holdouts: HoldoutPlan,
    config: SearchConfig,
    runner: CandidateRunner,
) -> ReportState:
    _progress("search_start")
    deadline = (
        math.inf
        if config.time_budget_seconds <= 0
        else time.monotonic() + config.time_budget_seconds
    )
    reports: list[CandidateReport] = []

    horizons = _prediction_horizons(config)
    suffix_ids = len(horizons) > 1
    for horizon_step in horizons:
        baseline = _baseline_candidate(config, horizon_step, suffix_ids)
        _progress(f"candidate_start id={baseline.candidate_id} holdouts={len(holdouts.intervals)} horizon={horizon_step}")
        baseline_results = [_safe_run(runner, baseline, holdout) for holdout in holdouts.intervals]
        reports.append(_candidate_report(baseline, baseline_results, len(holdouts.intervals)))
        write_report(config.report_path, _report_state(reports, config))
        _progress(f"candidate_end id={baseline.candidate_id}")

        worst = min(baseline_results, key=_score_for_worst_pick).holdout_name
        quick_holdout = next(holdout for holdout in holdouts.intervals if holdout.name == worst)

        for candidate in _low_risk_candidates_for_horizon(config, horizon_step, suffix_ids):
            if time.monotonic() >= deadline:
                break
            candidate_report = _run_screened_candidate(runner, candidate, quick_holdout, holdouts, baseline_results, deadline)
            reports.append(candidate_report)
            write_report(config.report_path, _report_state(reports, config))
            _progress(f"candidate_end id={candidate.candidate_id}")

    final_state = _report_state(reports, config)
    write_report(config.report_path, final_state)
    _progress("search_end")
    return final_state


def _initial_candidates(config: SearchConfig) -> list[CandidateConfig]:
    horizons = _prediction_horizons(config)
    suffix_ids = len(horizons) > 1
    return [
        candidate
        for horizon_step in horizons
        for candidate in _low_risk_candidates_for_horizon(config, horizon_step, suffix_ids)
    ]


def _baseline_candidate(config: SearchConfig, horizon_step: int, suffix_id: bool) -> CandidateConfig:
    candidate = CandidateConfig(
        candidate_id="baseline",
        window_minutes=config.default_window_minutes,
        context_policy="uniform",
        horizon_step=horizon_step,
        num_train_samples=config.num_train_samples,
        include_frequency=False,
        random_state=config.random_state,
        top_features_n=config.top_features_n,
        feature_mode="identity",
    )
    return _with_horizon(candidate, horizon_step, suffix_id)


def _low_risk_candidates_for_horizon(
    config: SearchConfig,
    horizon_step: int,
    suffix_id: bool,
) -> list[CandidateConfig]:
    return [_with_horizon(candidate, horizon_step, suffix_id) for candidate in _low_risk_candidates(config, horizon_step)]


def _low_risk_candidates(config: SearchConfig, horizon_step: int) -> list[CandidateConfig]:
    candidates = []
    candidates.extend(
        [
            CandidateConfig(
                "identity_recent",
                config.default_window_minutes,
                "recent",
                horizon_step,
                config.num_train_samples,
                top_features_n=config.top_features_n,
                feature_mode="identity",
            ),
            CandidateConfig(
                "identity_coverage",
                config.default_window_minutes,
                "coverage",
                horizon_step,
                config.num_train_samples,
                top_features_n=config.top_features_n,
                feature_mode="identity",
            ),
            CandidateConfig("trend_default", config.default_window_minutes, "uniform", horizon_step, config.num_train_samples, top_features_n=config.top_features_n),
            CandidateConfig("window_short", max(5, config.default_window_minutes // 2), "uniform", horizon_step, config.num_train_samples, top_features_n=config.top_features_n),
            CandidateConfig("window_long", config.default_window_minutes * 2, "uniform", horizon_step, config.num_train_samples, top_features_n=config.top_features_n),
            CandidateConfig("coverage", config.default_window_minutes, "coverage", horizon_step, config.num_train_samples, top_features_n=config.top_features_n),
        ]
    )
    if config.include_frequency_candidate:
        candidates.append(
            CandidateConfig(
                "frequency",
                config.default_window_minutes,
                "uniform",
                horizon_step,
                config.num_train_samples,
                True,
                top_features_n=config.top_features_n,
            )
        )
    return candidates


def _with_horizon(candidate: CandidateConfig, horizon_step: int, suffix_id: bool) -> CandidateConfig:
    if not suffix_id:
        return candidate
    return CandidateConfig(
        candidate_id=f"{candidate.candidate_id}_h+{horizon_step}",
        window_minutes=candidate.window_minutes,
        context_policy=candidate.context_policy,
        horizon_step=horizon_step,
        num_train_samples=candidate.num_train_samples,
        include_frequency=candidate.include_frequency,
        random_state=candidate.random_state,
        top_features_n=candidate.top_features_n,
        feature_mode=candidate.feature_mode,
    )


def _prediction_horizons(config: SearchConfig) -> tuple[int, ...]:
    values = tuple(sorted(set(config.prediction_horizons)))
    if not values:
        raise ValueError("prediction_horizons cannot be empty")
    if any(value <= 0 for value in values):
        raise ValueError("prediction_horizons must be positive")
    return values


def _run_screened_candidate(
    runner: CandidateRunner,
    candidate: CandidateConfig,
    quick_holdout: HoldoutInterval,
    holdouts: HoldoutPlan,
    baseline_results: list[HoldoutRunResult],
    deadline: float,
) -> CandidateReport:
    _progress(f"candidate_start id={candidate.candidate_id} quick_holdout={quick_holdout.name}")
    quick_result = _safe_run(runner, candidate, quick_holdout)
    results: list[HoldoutRunResult] = [quick_result]
    baseline_worst = next(result.rmse_improvement_pct for result in baseline_results if result.holdout_name == quick_holdout.name)
    if quick_result.rmse_improvement_pct >= baseline_worst:
        for holdout in holdouts.intervals:
            if holdout.name == quick_holdout.name or time.monotonic() >= deadline:
                continue
            results.append(_safe_run(runner, candidate, holdout))
    return _candidate_report(candidate, results, len(holdouts.intervals))


def _safe_run(
    runner: CandidateRunner,
    candidate: CandidateConfig,
    holdout: HoldoutInterval,
) -> HoldoutRunResult:
    try:
        return runner(candidate, holdout)
    except Exception as exc:  # noqa: BLE001
        return HoldoutRunResult(
            candidate_id=candidate.candidate_id,
            holdout_name=holdout.name,
            status="error",
            actual=[],
            predictions=[],
            r2=float("nan"),
            rmse=float("nan"),
            naive_rmse=float("nan"),
            rmse_improvement_pct=float("nan"),
            horizon_step=candidate.horizon_step,
            selected_features=[],
            error=repr(exc),
        )


def _candidate_report(
    config: CandidateConfig,
    results: list[HoldoutRunResult],
    total_windows: int,
) -> CandidateReport:
    score = candidate_score([result.rmse_improvement_pct for result in results], total_windows=total_windows)
    status = "complete" if len(results) == total_windows else "partial"
    return CandidateReport(candidate_id=config.candidate_id, score=score, status=status, holdouts=results)


def _rank(reports: list[CandidateReport]) -> list[CandidateReport]:
    return sorted(reports, key=lambda report: report.score, reverse=True)


def _report_state(reports: list[CandidateReport], config: SearchConfig) -> ReportState:
    return ReportState(_rank(reports), metadata=config.metadata or {})


def _progress(message: str) -> None:
    if os.environ.get("PREDICTOR_AUTORESEARCH_PROGRESS", os.environ.get("SOFT_SENSOR_PROGRESS", "1")) == "0":
        return
    print(f"[predictor-autoresearch] {message}", flush=True)


def _score_for_worst_pick(result: HoldoutRunResult) -> float:
    if math.isfinite(result.rmse_improvement_pct):
        return result.rmse_improvement_pct
    return float("-inf")
