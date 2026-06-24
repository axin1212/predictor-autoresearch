from __future__ import annotations

import pandas as pd
import pytest

from predictor_autoresearch.holdout import build_holdout_plan


def _frame(label_count: int) -> pd.DataFrame:
    rows = 120
    target = [None] * rows
    step = max(1, rows // label_count)
    for i in range(label_count):
        target[min(i * step, rows - 1)] = float(i)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min"),
            "target": target,
        }
    )


def test_build_holdout_plan_returns_three_intervals_for_enough_labels():
    plan = build_holdout_plan(_frame(36), "timestamp", "target")

    assert len(plan.intervals) == 3
    assert plan.confidence == "high"
    assert _label_sets_do_not_overlap(plan)
    assert sum(len(interval.label_indices) for interval in plan.intervals) == 11


def test_build_holdout_plan_degrades_to_two_intervals():
    plan = build_holdout_plan(_frame(18), "timestamp", "target")

    assert len(plan.intervals) == 2
    assert plan.confidence == "medium"
    assert _label_sets_do_not_overlap(plan)
    assert sum(len(interval.label_indices) for interval in plan.intervals) == 5


def test_build_holdout_plan_degrades_to_one_interval():
    plan = build_holdout_plan(_frame(10), "timestamp", "target")

    assert len(plan.intervals) == 1
    assert plan.confidence == "low"
    assert sum(len(interval.label_indices) for interval in plan.intervals) == 3


def test_build_holdout_plan_uses_configurable_validation_fraction():
    plan = build_holdout_plan(_frame(40), "timestamp", "target", validation_fraction=0.40)

    assert len(plan.intervals) == 3
    assert sum(len(interval.label_indices) for interval in plan.intervals) == 16


def test_build_holdout_plan_fails_below_eight_labels():
    with pytest.raises(ValueError, match="at least 8"):
        build_holdout_plan(_frame(7), "timestamp", "target")


def _label_sets_do_not_overlap(plan) -> bool:
    seen: set[int] = set()
    for interval in plan.intervals:
        current = set(interval.label_indices)
        if seen & current:
            return False
        seen |= current
    return True
