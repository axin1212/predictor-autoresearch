from __future__ import annotations

import numpy as np
import pytest

from predictor_autoresearch.scoring import candidate_score, r2_score_np, rmse_np, robust_score


def test_r2_preserves_negative_values():
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([3.0, 2.0, 1.0])

    assert r2_score_np(actual, predicted) < 0


def test_rmse_np():
    assert rmse_np(np.array([1.0, 3.0]), np.array([1.0, 1.0])) == 2**0.5


def test_robust_score_penalizes_std():
    stable = robust_score([0.5, 0.5, 0.5])
    unstable = robust_score([0.9, 0.5, 0.1])

    assert stable > unstable


def test_candidate_score_is_mean_r2_without_penalties():
    score = candidate_score([0.9, 0.5, 0.1], total_windows=3)

    assert score == pytest.approx(0.5)


def test_candidate_score_ignores_missing_windows_without_penalty():
    full = candidate_score([0.5, 0.4, 0.3], total_windows=3)
    partial = candidate_score([0.5, None, 0.3], total_windows=3)

    assert full == pytest.approx(0.4)
    assert partial == pytest.approx(0.4)
