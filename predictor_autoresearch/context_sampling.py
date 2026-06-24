from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from predictor_autoresearch.holdout import HoldoutInterval


class ContextPolicy(str, Enum):
    UNIFORM = "uniform"
    RECENT = "recent"
    COVERAGE = "coverage"
    SIMILAR = "similar"


def sample_context_indices(
    label_times: pd.Series,
    holdout: HoldoutInterval,
    policy: ContextPolicy | str,
    n: int,
    random_state: int,
) -> list[int]:
    times = pd.to_datetime(label_times, errors="coerce")
    available = [
        int(idx)
        for idx, value in times.items()
        if pd.notna(value) and not (holdout.start_time <= pd.Timestamp(value) <= holdout.end_time)
    ]
    if n >= len(available):
        return available
    if n <= 0:
        return []

    resolved = ContextPolicy(policy)
    if resolved == ContextPolicy.UNIFORM:
        rng = np.random.default_rng(random_state)
        return sorted(int(idx) for idx in rng.choice(available, size=n, replace=False))
    if resolved == ContextPolicy.RECENT:
        return _recent_indices(times, available, holdout, n)
    if resolved in {ContextPolicy.COVERAGE, ContextPolicy.SIMILAR}:
        return _coverage_indices(available, n)
    raise ValueError(f"unsupported context policy: {policy}")


def _recent_indices(
    times: pd.Series,
    available: list[int],
    holdout: HoldoutInterval,
    n: int,
) -> list[int]:
    center = holdout.start_time + (holdout.end_time - holdout.start_time) / 2
    ranked = sorted(
        available,
        key=lambda idx: (abs(pd.Timestamp(times.loc[idx]) - center), idx),
    )
    return ranked[:n]


def _coverage_indices(available: list[int], n: int) -> list[int]:
    if n == 1:
        return [available[len(available) // 2]]
    positions = np.linspace(0, len(available) - 1, n)
    selected = [available[int(round(pos))] for pos in positions]
    deduped: list[int] = []
    for idx in selected:
        if idx not in deduped:
            deduped.append(idx)
    for idx in available:
        if len(deduped) >= n:
            break
        if idx not in deduped:
            deduped.append(idx)
    return deduped
