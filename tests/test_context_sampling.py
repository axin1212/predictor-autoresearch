from __future__ import annotations

import pandas as pd

from predictor_autoresearch.context_sampling import ContextPolicy, sample_context_indices
from predictor_autoresearch.holdout import HoldoutInterval


def _times() -> pd.Series:
    return pd.Series(pd.date_range("2026-01-01", periods=20, freq="min"))


def _holdout() -> HoldoutInterval:
    return HoldoutInterval(
        name="h",
        start_time=pd.Timestamp("2026-01-01 00:10:00"),
        end_time=pd.Timestamp("2026-01-01 00:12:00"),
        label_indices=[10, 11, 12],
    )


def test_uniform_is_deterministic_with_seed():
    first = sample_context_indices(_times(), _holdout(), ContextPolicy.UNIFORM, n=5, random_state=7)
    second = sample_context_indices(_times(), _holdout(), ContextPolicy.UNIFORM, n=5, random_state=7)

    assert first == second
    assert len(first) == 5


def test_recent_prefers_times_closest_to_holdout():
    selected = sample_context_indices(_times(), _holdout(), ContextPolicy.RECENT, n=3, random_state=0)

    assert selected == [9, 13, 8]


def test_coverage_samples_across_range():
    selected = sample_context_indices(_times(), _holdout(), ContextPolicy.COVERAGE, n=4, random_state=0)

    assert len(selected) == 4
    assert min(selected) < 5
    assert max(selected) > 15


def test_request_above_available_returns_all_available():
    selected = sample_context_indices(_times(), _holdout(), ContextPolicy.UNIFORM, n=100, random_state=0)

    assert selected == [idx for idx in range(20) if idx not in {10, 11, 12}]
