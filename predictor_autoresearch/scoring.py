from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def r2_score_np(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if mask.sum() < 2:
        return float("nan")
    y = actual[mask]
    y_hat = predicted[mask]
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def rmse_np(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2)))


def robust_score(
    r2_values: Sequence[float],
    alpha: float = 0.5,
    beta: float = 0.25,
    r2_floor: float = 0.0,
) -> float:
    values = np.asarray([value for value in r2_values if np.isfinite(value)], dtype=float)
    if len(values) == 0:
        return float("-inf")
    return float(np.mean(values) - alpha * np.std(values) - beta * max(0.0, r2_floor - float(np.min(values))))


def candidate_score(r2_values: Sequence[float | None], total_windows: int) -> float:
    present = [value for value in r2_values if value is not None and np.isfinite(value)]
    if len(present) == 0:
        return float("-inf")
    return float(np.mean(present))
